"""
Event-driven factors: buyback announcements and post-earnings-announcement drift.

These are *overlay* factors — they produce per-symbol scores that are meant to
be blended on top of a base price/volume model (see ``enhance_scores``). When no
events are present they are transparent no-ops, so a pipeline can always call
them safely.

Both classes read from on-disk parquet caches so the scoring path never touches
the network. Cache builders live under ``scripts/`` in the project root.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["BuybackFactor", "PEADFactor"]


def _project_root() -> str:
    # src/quant/factors/events.py -> project root is three levels up.
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def _zscore_overlay(scores: Dict[str, float]) -> Dict[str, float]:
    """Cross-sectional z-score of an event score map (robust to singletons)."""
    vals = np.asarray(list(scores.values()), dtype=float)
    if vals.size < 2:
        return {sym: 0.0 for sym in scores}
    mu, sigma = float(np.mean(vals)), float(np.std(vals))
    if sigma < 1e-8:
        return {sym: 0.0 for sym in scores}
    return {sym: (v - mu) / sigma for sym, v in scores.items()}


class BuybackFactor:
    """
    Buyback event factor with time decay.

    Loads cached buyback events from parquet and computes a per-symbol score of
    ``size_score x decay_factor`` where:

    - ``size_score``: small (3) > medium (2) > large (1). This reverse
      dose-response reflects the empirical finding that small buybacks carry a
      larger announcement-period CAR than large ones.
    - ``decay_factor``: exponential decay ``exp(-days_ago / half_life)`` with a
      default half-life of 10 days.

    If the cached events already carry a ``size_score`` column it is used
    directly; otherwise the score is inferred from a ``size`` / ``amount``
    column when available, defaulting to medium (2).
    """

    # Size buckets (in currency units of the event's reported amount).
    _SMALL_MAX = 5.0e7      # <= 50M  -> small  -> 3
    _MEDIUM_MAX = 3.0e8     # <= 300M -> medium -> 2, else large -> 1

    def __init__(
        self,
        cache_path: Optional[str] = None,
        decay_days: int = 20,
        half_life: int = 10,
    ):
        if cache_path is None:
            cache_path = os.path.join(
                _project_root(), "data", "event_cache", "buyback_events.parquet"
            )
        self.cache_path = cache_path
        self.decay_days = decay_days
        self.half_life = half_life
        self.events: Optional[pd.DataFrame] = None

        if os.path.exists(cache_path):
            events = pd.read_parquet(cache_path)
            events["event_date"] = pd.to_datetime(events["event_date"])
            if "symbol" in events.columns:
                events["symbol"] = events["symbol"].astype(str).str.zfill(6)
            self.events = events

    # ------------------------------------------------------------------
    def _infer_size_score(self, row: pd.Series) -> float:
        """Derive a size score from a raw event row."""
        if "size_score" in row.index and pd.notna(row.get("size_score")):
            return float(row["size_score"])

        amount = row.get("amount", row.get("buyback_amount", np.nan))
        if pd.notna(amount):
            amount = float(amount)
            if amount <= self._SMALL_MAX:
                return 3.0
            if amount <= self._MEDIUM_MAX:
                return 2.0
            return 1.0

        # Explicit categorical size label, if present.
        label = str(row.get("size", "")).lower()
        if label in ("small", "s"):
            return 3.0
        if label in ("large", "l"):
            return 1.0
        return 2.0  # default: medium

    def compute_scores(self, as_of_date) -> Dict[str, float]:
        """
        Return ``{symbol: score}`` for stocks with buybacks in the decay window.

        Only events dated within ``[as_of - decay_days, as_of]`` contribute; if
        a symbol has several recent events, the highest decayed score wins.
        """
        if self.events is None or len(self.events) == 0:
            return {}

        as_of = pd.Timestamp(as_of_date)
        cutoff = as_of - pd.Timedelta(days=self.decay_days)
        mask = (self.events["event_date"] <= as_of) & (self.events["event_date"] >= cutoff)
        recent = self.events[mask]

        scores: Dict[str, float] = {}
        for _, row in recent.iterrows():
            days_ago = (as_of - row["event_date"]).days
            decay = float(np.exp(-days_ago / self.half_life))
            base = self._infer_size_score(row)
            value = base * decay
            sym = str(row["symbol"])
            if sym not in scores or value > scores[sym]:
                scores[sym] = value
        return scores

    def enhance_scores(
        self,
        base_scores: Dict[str, float],
        event_scores: Dict[str, float],
        weight: float = 0.15,
    ) -> Dict[str, float]:
        """
        Overlay z-scored buyback signal onto base model scores.

        Symbols present in ``base_scores`` but without a recent buyback are left
        unchanged.
        """
        if not event_scores:
            return base_scores
        z = _zscore_overlay(event_scores)
        enhanced = dict(base_scores)
        for sym, zv in z.items():
            if sym in enhanced:
                enhanced[sym] = enhanced[sym] + weight * zv
        return enhanced


class PEADFactor:
    """
    Post-Earnings Announcement Drift.

    Detects earnings surprises from cached fundamental forecast data and scores
    each symbol by its standardized unexpected earnings (SUE):

        SUE = (actual - expected) / std

    The drift is known to persist for roughly 60 days, so events older than
    ``decay_days`` (default 60) are ignored. As with :class:`BuybackFactor`, the
    score is intended as an overlay via :meth:`enhance_scores`.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        decay_days: int = 60,
        max_surprise_abs: float = 5.0,
    ):
        if cache_dir is None:
            cache_dir = os.path.join(_project_root(), "data", "pead_cache")
        self.cache_dir = cache_dir
        self.decay_days = decay_days
        self.max_surprise_abs = max_surprise_abs

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------
    def _load_forecasts(self) -> pd.DataFrame:
        """Load and concatenate all cached forecast parquet files."""
        if not os.path.isdir(self.cache_dir):
            return pd.DataFrame()

        frames = []
        for fname in sorted(os.listdir(self.cache_dir)):
            if not fname.endswith(".parquet"):
                continue
            try:
                df = pd.read_parquet(os.path.join(self.cache_dir, fname))
            except Exception:  # noqa: BLE001 - skip unreadable cache files
                continue
            if df is not None and len(df) > 0:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return self._clean_forecasts(pd.concat(frames, ignore_index=True))

    def _clean_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns and compute the earnings surprise (SUE proxy)."""
        df = df.copy()
        col_map = {
            "股票代码": "symbol", "股票简称": "name",
            "公告日期": "ann_date", "预测数值": "forecast_net",
            "上年同期值": "prev_year_net", "业绩变动": "change_type",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "symbol" not in df.columns or "ann_date" not in df.columns:
            return pd.DataFrame()

        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["ann_date"] = pd.to_datetime(df["ann_date"])

        for col in ("forecast_net", "prev_year_net"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "forecast_net" not in df.columns or "prev_year_net" not in df.columns:
            return pd.DataFrame()

        valid = (
            df["forecast_net"].notna()
            & df["prev_year_net"].notna()
            & (df["prev_year_net"] != 0)
        )
        df = df[valid].copy()

        surprise = (df["forecast_net"] - df["prev_year_net"]) / df["prev_year_net"].abs()
        df["surprise"] = surprise.clip(-self.max_surprise_abs, self.max_surprise_abs)
        df["surprise_sign"] = np.where(df["surprise"] > 0, "positive", "negative")
        return df[["symbol", "ann_date", "surprise", "surprise_sign"]]

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------
    def compute_scores(self, as_of_date=None) -> Dict[str, float]:
        """
        Return ``{symbol: surprise_score}`` for symbols with a live earnings
        surprise as of ``as_of_date`` (default: today).

        Only the most recent announcement within the decay window is used per
        symbol.
        """
        if as_of_date is None:
            as_of = pd.Timestamp(datetime.now().date())
        else:
            as_of = pd.Timestamp(as_of_date)

        forecasts = self._load_forecasts()
        if len(forecasts) == 0:
            return {}

        cutoff = as_of - timedelta(days=self.decay_days)
        recent = forecasts[
            (forecasts["ann_date"] <= as_of) & (forecasts["ann_date"] >= cutoff)
        ]
        if len(recent) == 0:
            return {}

        scores: Dict[str, float] = {}
        for sym, group in recent.groupby("symbol"):
            latest = group.sort_values("ann_date").iloc[-1]
            scores[str(sym)] = float(latest["surprise"])
        return scores

    def enhance_scores(
        self,
        base_scores: Dict[str, float],
        event_scores: Optional[Dict[str, float]] = None,
        weight: float = 0.15,
    ) -> Dict[str, float]:
        """
        Overlay z-scored PEAD surprise onto base model scores.

        ``event_scores`` defaults to :meth:`compute_scores` for today. Symbols
        without a live earnings event are passed through unchanged.
        """
        if event_scores is None:
            event_scores = self.compute_scores()
        if not event_scores:
            return base_scores

        z = _zscore_overlay(event_scores)
        enhanced = dict(base_scores)
        for sym, zv in z.items():
            if sym in enhanced:
                enhanced[sym] = enhanced[sym] + weight * zv
        return enhanced

    # ------------------------------------------------------------------
    def get_event_stats(self) -> dict:
        """Summary statistics over the full cached forecast history."""
        forecasts = self._load_forecasts()
        if len(forecasts) == 0:
            return {"has_data": False, "total_events": 0}
        pos = int((forecasts["surprise_sign"] == "positive").sum())
        neg = int((forecasts["surprise_sign"] == "negative").sum())
        return {
            "has_data": True,
            "total_events": int(len(forecasts)),
            "positive": pos,
            "negative": neg,
            "date_range": (
                f"{forecasts['ann_date'].min().date()} ~ "
                f"{forecasts['ann_date'].max().date()}"
            ),
            "mean_surprise": float(forecasts["surprise"].mean()),
        }
