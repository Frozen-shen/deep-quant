"""
Trading calendar module.

Provides a TradingCalendar class that wraps akshare's trade date data
with local caching to avoid repeated network calls. Supports querying
trading days, checking if a date is a trading day, and computing offsets.

Usage:
    from quant.data.calendar import TradingCalendar

    cal = TradingCalendar()
    cal.is_trading_day("2026-07-31")          # True/False
    cal.get_trading_days("2026-01-01", "2026-06-30")  # List[Timestamp]
    cal.offset("2026-07-31", -5)              # 5 trading days back
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_left, bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache location: project_root/data/cache/trading_calendar.csv
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

DateLike = Union[str, date, datetime, pd.Timestamp]


class TradingCalendar:
    """
    Trading calendar for A-share markets.

    Loads trading days from akshare (Sina source) and caches locally.
    Provides efficient lookup via a sorted list with binary search.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to store the cached calendar CSV. Defaults to
        ``<project_root>/data/cache/``.
    force_refresh : bool
        If True, bypass cache and re-fetch from akshare.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        force_refresh: bool = False,
    ):
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_path = self._cache_dir / "trading_calendar.csv"
        self._trading_days: List[pd.Timestamp] = []
        self._trading_days_set: set = set()
        self._loaded = False

        if force_refresh:
            self._fetch_and_cache()
        else:
            self._ensure_loaded()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trading_day(self, dt: Optional[DateLike] = None) -> bool:
        """
        Check whether a given date is a trading day.

        Parameters
        ----------
        dt : date-like, optional
            The date to check. Defaults to today.

        Returns
        -------
        bool
        """
        self._ensure_loaded()
        ts = self._to_timestamp(dt)
        return ts in self._trading_days_set

    def get_trading_days(
        self,
        start: DateLike,
        end: DateLike,
    ) -> List[pd.Timestamp]:
        """
        Get all trading days in [start, end] inclusive.

        Parameters
        ----------
        start, end : date-like
            Range boundaries (inclusive).

        Returns
        -------
        List[pd.Timestamp]
            Sorted list of trading day timestamps.
        """
        self._ensure_loaded()
        start_ts = self._to_timestamp(start)
        end_ts = self._to_timestamp(end)
        left = bisect_left(self._trading_days, start_ts)
        right = bisect_right(self._trading_days, end_ts)
        return self._trading_days[left:right]

    def offset(self, dt: DateLike, n: int) -> pd.Timestamp:
        """
        Offset a date by n trading days.

        Parameters
        ----------
        dt : date-like
            Reference date. If it is not a trading day, the next trading
            day is used as the base.
        n : int
            Number of trading days to move. Positive = forward,
            negative = backward.

        Returns
        -------
        pd.Timestamp
            The resulting trading day.

        Raises
        ------
        ValueError
            If the offset exceeds the available calendar range.
        """
        self._ensure_loaded()
        ts = self._to_timestamp(dt)

        # Find the index of dt (or the next trading day if dt is not one)
        idx = bisect_left(self._trading_days, ts)
        if idx >= len(self._trading_days):
            raise ValueError(
                f"Date {ts.date()} is beyond the calendar range "
                f"(max: {self._trading_days[-1].date()})"
            )

        # If dt is not a trading day, idx points to the next one (that's our base)
        target_idx = idx + n
        if target_idx < 0 or target_idx >= len(self._trading_days):
            raise ValueError(
                f"Offset {n} from {ts.date()} exceeds calendar bounds "
                f"({self._trading_days[0].date()} ~ {self._trading_days[-1].date()})"
            )
        return self._trading_days[target_idx]

    def prev_trading_day(self, dt: Optional[DateLike] = None, n: int = 1) -> pd.Timestamp:
        """
        Get the n-th previous trading day before dt.

        Parameters
        ----------
        dt : date-like, optional
            Reference date. Defaults to today.
        n : int
            How many trading days back. Must be >= 1.

        Returns
        -------
        pd.Timestamp
        """
        self._ensure_loaded()
        ts = self._to_timestamp(dt)
        # bisect_left gives the insertion point; everything before is < ts
        idx = bisect_left(self._trading_days, ts)
        target_idx = idx - n
        if target_idx < 0:
            raise ValueError(
                f"Cannot go {n} trading days before {ts.date()}: "
                f"calendar starts at {self._trading_days[0].date()}"
            )
        return self._trading_days[target_idx]

    def next_trading_day(self, dt: Optional[DateLike] = None, n: int = 1) -> pd.Timestamp:
        """
        Get the n-th next trading day after dt.

        Parameters
        ----------
        dt : date-like, optional
            Reference date. Defaults to today.
        n : int
            How many trading days forward. Must be >= 1.

        Returns
        -------
        pd.Timestamp
        """
        self._ensure_loaded()
        ts = self._to_timestamp(dt)
        # bisect_right gives the index after all entries <= ts
        idx = bisect_right(self._trading_days, ts)
        target_idx = idx + n - 1
        if target_idx >= len(self._trading_days):
            raise ValueError(
                f"Cannot go {n} trading days after {ts.date()}: "
                f"calendar ends at {self._trading_days[-1].date()}"
            )
        return self._trading_days[target_idx]

    @property
    def date_range(self) -> tuple:
        """Return (earliest, latest) trading day in the calendar."""
        self._ensure_loaded()
        if not self._trading_days:
            return (None, None)
        return (self._trading_days[0], self._trading_days[-1])

    @property
    def count(self) -> int:
        """Total number of trading days loaded."""
        self._ensure_loaded()
        return len(self._trading_days)

    def count_trading_days(self, start: DateLike, end: DateLike) -> int:
        """Count trading days in [start, end] inclusive."""
        return len(self.get_trading_days(start, end))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load calendar from cache or fetch from network."""
        if self._loaded:
            return

        if self._cache_path.exists():
            self._load_from_cache()
        else:
            self._fetch_and_cache()

        self._loaded = True

    def _load_from_cache(self) -> None:
        """Load trading days from the local CSV cache."""
        try:
            df = pd.read_csv(self._cache_path, parse_dates=["date"])
            days = sorted(pd.to_datetime(df["date"]).tolist())
            self._trading_days = days
            self._trading_days_set = set(days)
            logger.debug(
                "Loaded %d trading days from cache (%s ~ %s)",
                len(days), days[0].date(), days[-1].date(),
            )
        except Exception as e:
            logger.warning("Failed to load calendar cache: %s. Re-fetching.", e)
            self._fetch_and_cache()

    def _fetch_and_cache(self) -> None:
        """Fetch trading calendar from akshare and persist to CSV."""
        import akshare as ak

        logger.info("Fetching trading calendar from akshare (Sina)...")
        try:
            df = ak.tool_trade_date_hist_sina()
            # akshare returns column 'trade_date'
            if "trade_date" in df.columns:
                df = df.rename(columns={"trade_date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset=["date"]).sort_values("date")

            # Persist
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            df[["date"]].to_csv(self._cache_path, index=False)

            days = sorted(df["date"].tolist())
            self._trading_days = days
            self._trading_days_set = set(days)
            self._loaded = True
            logger.info(
                "Calendar fetched and cached: %d days (%s ~ %s)",
                len(days), days[0].date(), days[-1].date(),
            )
        except Exception as e:
            logger.error("Failed to fetch trading calendar: %s", e)
            # If we have a stale cache, use it
            if self._cache_path.exists():
                logger.warning("Falling back to stale cache.")
                self._load_from_cache()
                self._loaded = True
            else:
                raise RuntimeError(
                    f"Cannot initialize TradingCalendar: fetch failed ({e}) "
                    "and no local cache exists."
                ) from e

    @staticmethod
    def _to_timestamp(dt: Optional[DateLike]) -> pd.Timestamp:
        """Normalize various date inputs to a normalized pd.Timestamp."""
        if dt is None:
            return pd.Timestamp.now().normalize()
        ts = pd.Timestamp(dt)
        return ts.normalize()
