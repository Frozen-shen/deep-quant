"""
Walk-Forward Optimization (WFO) trainer.

WFO is the honest way to evaluate a factor model: it never lets the model see
the future. Data is split into rolling train/test windows that march forward
in time::

    [----train----][--test--]
         [----train----][--test--]
              [----train----][--test--]

For each window the trainer:
    1. Builds a cross-sectionally standardised factor matrix on the train set.
    2. Fits the model (or, for the IC-weighted linear model, sets IC weights).
    3. Generates out-of-sample predictions on the test set.
    4. Measures the realised Rank IC of those predictions.

The trainer is deliberately model-agnostic: it works with any object that
exposes ``fit(X, y, groups)`` and ``predict(X)`` (an optional ``factor_names``
keyword on ``predict`` is forwarded when the model accepts it).

Default windowing matches the project config:
    train_days = 504  (2 years)
    test_days  = 63   (1 quarter)
    step_days  = 63   (roll forward 1 quarter)
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quant.model.linear import RankICCalculator

logger = logging.getLogger(__name__)


def _accepted_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the keyword arguments that ``func`` accepts.

    Models have heterogeneous signatures (the IC-weighted linear model takes
    ``factor_names``; the LightGBM ranker does not). Filtering by signature
    lets the trainer forward shared arguments without crashing on models that
    do not recognise them, while still surfacing genuine errors.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = {
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


@dataclass
class WFOResult:
    """Results from walk-forward optimization."""

    predictions: pd.DataFrame          # date x symbol predicted scores
    ic_series: pd.Series               # realised IC per test period
    weights_history: List[Dict[str, float]] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - convenience only
        lines = ["=== Walk-Forward Result ==="]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key:20s}: {value:+.4f}")
            else:
                lines.append(f"  {key:20s}: {value}")
        lines.append("===========================")
        return "\n".join(lines)


class WalkForwardTrainer:
    """
    Walk-Forward Optimization (WFO) trainer.

    Args:
        model: Any object with ``fit(X, y, groups)`` and ``predict(X)``.
        config: Optional dict overriding the windowing parameters:
            - ``train_days`` (int): training window length in trading days.
            - ``test_days``  (int): out-of-sample window length.
            - ``step_days``  (int): forward step between windows.
            - ``forward_period`` (int): return horizon for IC measurement.
            - ``min_samples`` (int): min cross-section size for IC.
            - ``min_train_samples`` (int): min rows required to fit a window.

    Example:
        >>> from quant.model.linear import ICWeightedLinear
        >>> trainer = WalkForwardTrainer(ICWeightedLinear(), {"train_days": 504})
        >>> result = trainer.run(factor_panels, returns, symbols)
        >>> print(result.metrics["mean_ic"], result.metrics["ic_ir"])
    """

    DEFAULT_CONFIG = {
        "train_days": 504,
        "test_days": 63,
        "step_days": 63,
        "forward_period": 5,
        "min_samples": 10,
        "min_train_samples": 100,
    }

    def __init__(self, model, config: Optional[dict] = None):
        self.model = model
        self.config = dict(self.DEFAULT_CONFIG)
        self.config.update(config or {})
        self._ic_calc = RankICCalculator(
            forward_period=self.config["forward_period"],
            min_samples=self.config["min_samples"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> WFOResult:
        """
        Execute the full walk-forward loop.

        Args:
            factor_panels: Mapping of factor name -> DataFrame indexed by date
                with one column per symbol (date x symbol).
            returns: DataFrame indexed by date, one column per symbol, of
                forward returns aligned to the factor horizon.
            symbols: Optional explicit symbol universe. Defaults to the union
                of columns found across the factor panels and returns.

        Returns:
            A populated :class:`WFOResult`.
        """
        if not factor_panels:
            raise ValueError("factor_panels is empty.")

        factor_names = list(factor_panels.keys())
        returns = returns.sort_index()
        dates = list(returns.index)

        if symbols is None:
            symbol_set = set(returns.columns)
            for panel in factor_panels.values():
                symbol_set.update(panel.columns)
            symbols = sorted(symbol_set)

        train_days = int(self.config["train_days"])
        test_days = int(self.config["test_days"])
        step_days = int(self.config["step_days"])

        prediction_frames: List[pd.DataFrame] = []
        ic_records: Dict = {}
        weights_history: List[Dict[str, float]] = []

        start = 0
        n_dates = len(dates)
        window_idx = 0

        while start + train_days < n_dates:
            train_dates = dates[start : start + train_days]
            test_dates = dates[start + train_days : start + train_days + test_days]
            if not test_dates:
                break

            logger.info(
                "[WFO] window %d: train %s..%s (%d), test %s..%s (%d)",
                window_idx,
                train_dates[0],
                train_dates[-1],
                len(train_dates),
                test_dates[0],
                test_dates[-1],
                len(test_dates),
            )

            # ---- Train ------------------------------------------------
            X_train, y_train, g_train = self._build_matrix(
                factor_panels, returns, factor_names, symbols, train_dates
            )

            trained = False
            if len(X_train) >= int(self.config["min_train_samples"]):
                try:
                    # Ranking models (lambdarank) need integer relevance
                    # labels; regression/IC models are indifferent to the
                    # integer conversion since they rely on rank correlation.
                    y_fit = self._prepare_labels(y_train, g_train)
                    self._train_window(X_train, y_fit, g_train, factor_names)
                    trained = True
                except Exception as exc:  # keep the walk going on failure
                    logger.warning("[WFO] window %d fit failed: %s", window_idx, exc)

            # Record weights if the model exposes them (linear model).
            weights_history.append(self._current_weights(factor_names))

            # ---- Predict + score -------------------------------------
            if trained:
                pred_frame, window_ic = self._predict_window(
                    factor_panels, returns, factor_names, symbols, test_dates
                )
                if pred_frame is not None and not pred_frame.empty:
                    prediction_frames.append(pred_frame)
                if window_ic is not None and np.isfinite(window_ic):
                    # One IC observation per test date, keyed by date.
                    for d in test_dates:
                        ic_records[d] = window_ic

            start += step_days
            window_idx += 1

        predictions = self._combine_predictions(prediction_frames, symbols)
        ic_series = pd.Series(ic_records, dtype=float).sort_index()
        ic_series.name = "ic"
        metrics = self._compute_metrics(ic_series)

        return WFOResult(
            predictions=predictions,
            ic_series=ic_series,
            weights_history=weights_history,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _build_matrix(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        factor_names: List[str],
        symbols: List[str],
        dates: List,
    ):
        """
        Build a long-format (n_obs, n_factors) matrix over the given dates.

        Each observation is a (date, symbol) pair. Factors are cross-
        sectionally z-scored within each date so that columns are on a common
        scale and comparable across time. The returned ``groups`` array is an
        integer date id suitable for the ranking objective / IC computation.
        """
        X_rows: List[np.ndarray] = []
        y_list: List[float] = []
        g_list: List[int] = []

        for group_id, date in enumerate(dates):
            if date not in returns.index:
                continue
            ret_row = returns.loc[date]

            # Collect raw factor values for this cross-section.
            cross = {}
            for name in factor_names:
                panel = factor_panels[name]
                if date in panel.index:
                    cross[name] = panel.loc[date]
                else:
                    cross[name] = pd.Series(dtype=float)

            # Standardise each factor cross-sectionally (z-score).
            z_cross: Dict[str, pd.Series] = {}
            for name in factor_names:
                series = cross[name].reindex(symbols).astype(float)
                z_cross[name] = _cross_sectional_zscore(series)

            for sym in symbols:
                if sym not in ret_row.index:
                    continue
                y_val = ret_row[sym]
                if pd.isna(y_val):
                    continue
                row = [z_cross[name].get(sym, np.nan) for name in factor_names]
                X_rows.append(np.array(row, dtype=float))
                y_list.append(float(y_val))
                g_list.append(group_id)

        if not X_rows:
            return np.empty((0, len(factor_names))), np.empty(0), np.empty(0, dtype=int)

        X = np.array(X_rows, dtype=float)
        # Impute any residual NaN with the neutral value 0.
        X = np.where(np.isnan(X), 0.0, X)
        y = np.array(y_list, dtype=float)
        groups = np.array(g_list, dtype=int)
        return X, y, groups

    def _prepare_labels(
        self, y: np.ndarray, groups: np.ndarray
    ) -> np.ndarray:
        """
        Convert forward returns into integer cross-sectional relevance labels.

        Ranking objectives (LightGBM lambdarank) require non-negative integer
        labels, whereas the IC-weighted linear model relies on rank
        correlation and is therefore invariant to this monotonic transform.
        Converting once here keeps the trainer model-agnostic: every model
        receives the same integer labels (0 = worst within each date).

        Args:
            y: ``(n_obs,)`` forward returns.
            groups: ``(n_obs,)`` integer group (date) ids.

        Returns:
            ``(n_obs,)`` integer labels ranked within each group.
        """
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        labels = np.zeros(len(y), dtype=np.int64)
        frame = pd.DataFrame({"y": y, "g": groups})
        # Rank within each group (ascending: smallest return -> rank 0),
        # using a dense integer labelling suitable for lambdarank.
        frame["label"] = (
            frame.groupby("g")["y"].rank(method="average").sub(1).round()
        )
        labels = frame["label"].clip(lower=0).to_numpy(dtype=np.int64)
        return labels

    def _train_window(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        g_train: np.ndarray,
        factor_names: List[str],
    ) -> None:
        """
        Fit the model on one training window.

        Forwards ``factor_names`` to models that accept it (e.g. the
        IC-weighted linear model) and omits it for models that do not.
        """
        kwargs = _accepted_kwargs(self.model.fit, {"factor_names": factor_names})
        self.model.fit(X_train, y_train, g_train, **kwargs)

    def _predict_window(
        self,
        factor_panels: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        factor_names: List[str],
        symbols: List[str],
        test_dates: List,
    ):
        """
        Generate out-of-sample predictions and measure realised IC.

        Returns a (date x symbol) prediction frame and the mean realised Rank
        IC across the test dates (or None if it cannot be computed).
        """
        records: Dict = {}
        per_date_ic: List[float] = []

        for date in test_dates:
            if date not in returns.index:
                continue
            ret_row = returns.loc[date]

            # Build and standardise the cross-section for this date.
            columns = {}
            for name in factor_names:
                panel = factor_panels[name]
                if date in panel.index:
                    series = panel.loc[date].reindex(symbols).astype(float)
                else:
                    series = pd.Series(np.nan, index=symbols)
                columns[name] = _cross_sectional_zscore(series)

            X = np.column_stack(
                [columns[name].reindex(symbols).to_numpy(dtype=float) for name in factor_names]
            )
            X = np.where(np.isnan(X), 0.0, X)

            scores = self._safe_predict(X, factor_names)

            # Store scores keyed by symbol.
            score_series = pd.Series(scores, index=symbols)
            records[date] = score_series

            # Realised IC: correlation of predicted scores with true returns.
            valid = score_series.to_frame("score").join(
                ret_row.rename("ret"), how="inner"
            ).dropna()
            if len(valid) >= self._ic_calc.min_samples:
                ic = self._ic_calc.compute(valid["score"], valid["ret"])
                per_date_ic.append(ic)

        if not records:
            return None, None

        pred_frame = pd.DataFrame(records).T  # date x symbol
        pred_frame = pred_frame.reindex(columns=symbols)
        mean_ic = float(np.mean(per_date_ic)) if per_date_ic else None
        return pred_frame, mean_ic

    def _safe_predict(self, X: np.ndarray, factor_names: List[str]) -> np.ndarray:
        """Call predict, forwarding factor_names only if the model accepts it."""
        kwargs = _accepted_kwargs(self.model.predict, {"factor_names": factor_names})
        return np.asarray(self.model.predict(X, **kwargs), dtype=float)

    def _current_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """Snapshot the model's factor weights if it exposes them."""
        if hasattr(self.model, "get_weights"):
            try:
                weights = self.model.get_weights()
                return {name: float(weights.get(name, 0.0)) for name in factor_names}
            except Exception:  # pragma: no cover - defensive
                return {}
        return {}

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _combine_predictions(
        frames: List[pd.DataFrame], symbols: List[str]
    ) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=symbols)
        combined = pd.concat(frames, axis=0)
        # If windows overlap, keep the most recent prediction per date.
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index().reindex(columns=symbols)

    @staticmethod
    def _compute_metrics(ic_series: pd.Series) -> Dict:
        """
        Summarise the out-of-sample IC series.

        Reports mean IC, IC volatility, the information-ratio-style ICIR, the
        hit rate (share of periods with positive IC), and a crude annualised
        ICIR assuming quarterly periods.
        """
        ic = ic_series.dropna()
        metrics: Dict = {"n_periods": int(len(ic))}

        if len(ic) == 0:
            metrics.update(
                {
                    "mean_ic": 0.0,
                    "ic_std": 0.0,
                    "ic_ir": 0.0,
                    "ic_ir_annualized": 0.0,
                    "hit_rate": 0.0,
                }
            )
            return metrics

        mean_ic = float(ic.mean())
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
        # Quarterly periods -> ~4 per year.
        ic_ir_ann = ic_ir * np.sqrt(4)

        metrics.update(
            {
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "ic_ir": float(ic_ir),
                "ic_ir_annualized": float(ic_ir_ann),
                "hit_rate": float((ic > 0).mean()),
                "min_ic": float(ic.min()),
                "max_ic": float(ic.max()),
            }
        )
        return metrics


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score a single cross-section (one date). NaNs are preserved so callers
    can decide how to impute; a zero-variance cross-section returns all zeros.
    """
    values = series.astype(float)
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0 or pd.isna(mean):
        return pd.Series(0.0, index=series.index)
    return (values - mean) / std
