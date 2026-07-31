"""
IC-weighted linear factor combination model — the PRIMARY model.

Background
----------
Backtesting proved that every machine-learning model we tried underperformed
a naive equal-weight baseline:

    L0 (equal weight):        +2.79% excess, IR = +0.377   <- BEST
    L3 (LightGBM, 600 trees): -5.50% excess, IR = -0.364   <- WORST
    Ensemble (3x LGB):        -4.6%  excess                <- STILL BAD

Root cause: overfitting. 600 trees x depth 5 trained on ~38k samples simply
memorised noise, and the 30bp transaction cost amplified the high-turnover
losses that resulted.

The IC-weighted linear model is the answer. It is *not* a learned model — it
has no parameters to optimise, so there is nothing to overfit. Each factor is
weighted by its own recent predictive power (Rank IC), so the model adapts as
factor efficacy drifts, but the weights change slowly and stay interpretable.

Formula
-------
    weight_i = decayed_mean(IC_i, lookback)   if decayed_mean > min_ic else 0
    score(s) = sum_i(factor_i(s) * weight_i) / sum_i(weight_i)

Properties
----------
    - Zero overfitting risk (no parameters to learn).
    - Self-adapting (weights update as IC changes).
    - Low turnover (weights move slowly, driven by a long IC lookback).
    - Interpretable (you can inspect exactly which factors drive decisions).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # scipy is a hard dependency but degrade gracefully if absent.
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

logger = logging.getLogger(__name__)


class RankICCalculator:
    """
    Compute Rank IC — the Spearman rank correlation between factor values
    and realised forward returns.

    Rank IC is the standard measure of a factor's cross-sectional predictive
    power. A value of +0.05 means stocks that rank high on the factor tend to
    rank high on subsequent returns. Values above ~0.02-0.03 are generally
    considered useful once they are persistent over time.

    Args:
        forward_period: The return horizon the factor is expected to predict.
            Stored for reference only; the caller is responsible for passing
            returns that already match this horizon.
        min_samples: Minimum number of valid (non-NaN) paired observations
            required to compute a correlation. Below this the result is 0.0.

    Example:
        >>> calc = RankICCalculator(forward_period=5)
        >>> ic = calc.compute(factor_values, forward_returns)
    """

    def __init__(self, forward_period: int = 5, min_samples: int = 10):
        if forward_period < 1:
            raise ValueError("forward_period must be >= 1")
        self.forward_period = forward_period
        self.min_samples = min_samples

    def compute(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """
        Compute the Rank IC (Spearman correlation) for a single factor.

        Args:
            factor_values: Cross-sectional factor values for one date,
                indexed by symbol.
            forward_returns: Realised forward returns aligned to the same
                index (same symbols).

        Returns:
            The Spearman rank correlation in [-1, 1]. Returns 0.0 when there
            is insufficient data or the input is degenerate (zero variance).
        """
        factor_values = pd.Series(factor_values)
        forward_returns = pd.Series(forward_returns)

        # Align on the shared index and drop any NaN on either side.
        combined = pd.concat(
            [factor_values, forward_returns], axis=1, join="inner"
        ).dropna()

        if len(combined) < self.min_samples:
            return 0.0

        x = combined.iloc[:, 0].to_numpy(dtype=float)
        y = combined.iloc[:, 1].to_numpy(dtype=float)

        # Degenerate input (constant factor or constant returns) -> no signal.
        if np.all(x == x[0]) or np.all(y == y[0]):
            return 0.0

        if spearmanr is not None:
            result = spearmanr(x, y)
            # scipy >= 1.9 returns a SignificanceResult; older returns a tuple.
            corr = getattr(result, "correlation", result[0])
        else:  # pragma: no cover - fallback to a manual rank correlation.
            corr = _manual_spearman(x, y)

        if corr is None or not np.isfinite(corr):
            return 0.0
        return float(corr)

    def compute_batch(
        self,
        factor_dict: Dict[str, pd.Series],
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute Rank IC for many factors against the same forward returns.

        Args:
            factor_dict: Mapping of factor name -> cross-sectional values.
            returns: Realised forward returns, indexed by symbol.

        Returns:
            Mapping of factor name -> Rank IC.
        """
        return {
            name: self.compute(values, returns)
            for name, values in factor_dict.items()
        }


class ICWeightedLinear:
    """
    IC-weighted linear factor combination model.

    Core idea::

        weight_i = rolling_mean(IC_i, lookback) if IC_i > 0 else 0
        score(stock) = sum(factor_i * weight_i) / sum(weight_i)

    This is NOT a learned model — it is a simple weighted average where the
    weights are determined by historical predictive power (IC). There are no
    parameters to optimise and therefore nothing to overfit.

    Properties:
        - Zero overfitting (no parameters to optimise).
        - Self-adapting (weights update as IC changes).
        - Low turnover (weights change slowly).
        - Interpretable (can inspect which factors contribute).

    The model exposes the same ``fit`` / ``predict`` duck-typed interface used
    by the other models in this package so it can be driven by
    :class:`~quant.model.trainer.WalkForwardTrainer`.

    Args:
        ic_lookback: Number of recent IC observations used to compute the
            rolling (decayed) mean IC for each factor.
        min_ic: Minimum decayed mean IC for a factor to receive any weight.
            Factors below this threshold are zeroed out. Use 0.0 to include
            every factor with positive IC.
        decay_halflife: Exponential-decay half-life (in IC observations) used
            when averaging IC history. Recent IC observations count more than
            old ones. A larger value means slower adaptation.
    """

    def __init__(
        self,
        ic_lookback: int = 120,
        min_ic: float = 0.02,
        decay_halflife: int = 60,
    ):
        if ic_lookback < 1:
            raise ValueError("ic_lookback must be >= 1")
        if decay_halflife < 1:
            raise ValueError("decay_halflife must be >= 1")

        self.ic_lookback = ic_lookback
        self.min_ic = min_ic
        self.decay_halflife = decay_halflife

        # factor_name -> list of (date, ic_value), in chronological order.
        self._ic_history: Dict[str, List[float]] = {}
        self._ic_dates: Dict[str, List] = {}
        # factor_name -> most recent weight (populated by get_weights()).
        self._weights: Dict[str, float] = {}
        # factor_name -> contribution to the most recent predict() call.
        self._attribution: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # IC ingestion
    # ------------------------------------------------------------------
    def update_ic(self, factor_name: str, ic_value: float, date=None) -> None:
        """
        Record a single daily IC observation for a factor.

        Args:
            factor_name: The factor the IC was computed for.
            ic_value: The realised Rank IC for this period.
            date: Optional date label (used for bookkeeping / debugging).
        """
        if ic_value is None or not np.isfinite(ic_value):
            return
        self._ic_history.setdefault(factor_name, []).append(float(ic_value))
        self._ic_dates.setdefault(factor_name, []).append(date)

    def update_ic_batch(self, ic_dict: Dict[str, float], date=None) -> None:
        """
        Record IC observations for many factors at once (one date).

        Args:
            ic_dict: Mapping of factor name -> IC value.
            date: Optional date label shared by all observations.
        """
        for factor_name, ic_value in ic_dict.items():
            self.update_ic(factor_name, ic_value, date)

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------
    def _decayed_mean(self, ic_values: Sequence[float]) -> float:
        """
        Exponentially-decayed mean of an IC series.

        The most recent observation has weight 1.0 and weights halve every
        ``decay_halflife`` steps going backwards. This makes the average
        respond to recent changes in factor efficacy while still smoothing
        out single-period noise.
        """
        n = len(ic_values)
        if n == 0:
            return 0.0

        arr = np.asarray(ic_values, dtype=float)
        if self.decay_halflife <= 0:  # uniform mean fallback
            return float(np.mean(arr))

        # Positions measured backwards from the most recent observation (0).
        ages = np.arange(n - 1, -1, -1, dtype=float)
        decay = np.exp(-math.log(2.0) * ages / self.decay_halflife)

        total_weight = decay.sum()
        if total_weight <= 0:
            return 0.0
        return float((arr * decay).sum() / total_weight)

    def get_weights(self) -> Dict[str, float]:
        """
        Current factor weights based on rolling (decayed) IC.

        Only factors whose decayed mean IC exceeds ``min_ic`` receive a
        positive weight; all others are set to zero. The returned weights are
        the decayed mean IC values themselves (not normalised to sum to one);
        :meth:`predict` normalises internally so that the absolute scale of
        the weights is irrelevant to the resulting ranking.

        Returns:
            Mapping of factor name -> weight (>= 0).
        """
        weights: Dict[str, float] = {}
        for factor_name, history in self._ic_history.items():
            window = history[-self.ic_lookback:]
            mean_ic = self._decayed_mean(window)
            weights[factor_name] = mean_ic if mean_ic > self.min_ic else 0.0

        self._weights = weights
        return weights

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        factor_matrix: np.ndarray,
        factor_names: List[str],
    ) -> np.ndarray:
        """
        Predict cross-sectional scores.

        Args:
            factor_matrix: ``(n_stocks, n_factors)`` normalised factor values.
                NaN entries are imputed with 0 (the neutral value for
                cross-sectionally standardised factors).
            factor_names: Factor names corresponding to the columns of
                ``factor_matrix``.

        Returns:
            ``(n_stocks,)`` predicted ranking scores. Higher is better.
        """
        factor_matrix = np.asarray(factor_matrix, dtype=float)
        if factor_matrix.ndim != 2:
            raise ValueError(
                f"factor_matrix must be 2D, got shape {factor_matrix.shape}"
            )
        n_stocks, n_factors = factor_matrix.shape
        if len(factor_names) != n_factors:
            raise ValueError(
                f"factor_names length ({len(factor_names)}) does not match "
                f"factor_matrix columns ({n_factors})"
            )

        # Ensure weights reflect the latest IC history.
        all_weights = self.get_weights()

        # Build an aligned weight vector for the requested columns.
        w = np.array(
            [all_weights.get(name, 0.0) for name in factor_names], dtype=float
        )
        weight_sum = w.sum()

        # Impute NaN with the neutral value (0 for standardised factors).
        x = np.where(np.isnan(factor_matrix), 0.0, factor_matrix)

        if weight_sum <= 0:
            # No factor currently passes the IC threshold -> neutral scores.
            self._attribution = {name: 0.0 for name in factor_names}
            return np.zeros(n_stocks, dtype=float)

        scores = (x * w).sum(axis=1) / weight_sum

        # Attribution: share of the (mean absolute) score driven by each
        # factor. Summed across stocks this shows which factors dominate.
        contribution = np.abs(x * w).sum(axis=0) / weight_sum
        contrib_total = contribution.sum()
        if contrib_total > 0:
            self._attribution = {
                name: float(contribution[i] / contrib_total)
                for i, name in enumerate(factor_names)
            }
        else:
            self._attribution = {name: 0.0 for name in factor_names}

        return scores

    def get_attribution(self) -> Dict[str, float]:
        """
        Factor contribution to the most recent :meth:`predict` call.

        Returns:
            Mapping of factor name -> fractional contribution in [0, 1].
            The values sum to ~1 across the factors that were present in the
            last prediction. Empty if :meth:`predict` has not been called.
        """
        return dict(self._attribution)

    # ------------------------------------------------------------------
    # Trainer-facing duck-typed interface
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        val_ratio: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None,
        factor_names: Optional[List[str]] = None,
        calc: Optional[RankICCalculator] = None,
    ) -> "ICWeightedLinear":
        """
        Set IC weights directly from a training matrix.

        There are no parameters to learn, but this method lets the model plug
        into the same walk-forward trainer as the ML models. It computes the
        Rank IC of each feature column against ``y`` (grouped by ``groups``
        when provided) and feeds the results through :meth:`update_ic`.

        Args:
            X: ``(n_samples, n_features)`` normalised factor values.
            y: ``(n_samples,)`` forward returns or ranking labels.
            groups: ``(n_samples,)`` group ids (e.g. date ids). IC is computed
                per group and then aggregated. If None, a single global IC is
                computed per feature.
            val_ratio: Ignored (present for interface compatibility).
            sample_weight: Ignored (present for interface compatibility).
            factor_names: Names for the feature columns. Defaults to
                ``f_0, f_1, ...``.
            calc: Optional :class:`RankICCalculator`. A default one is created
                if not supplied.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]

        if factor_names is None:
            factor_names = [f"f_{i}" for i in range(n_features)]
        if len(factor_names) != n_features:
            raise ValueError(
                f"factor_names length ({len(factor_names)}) does not match "
                f"feature count ({n_features})"
            )

        calculator = calc or RankICCalculator()

        if groups is None:
            # Single global IC per feature.
            ic_by_factor: Dict[str, float] = {}
            for i, name in enumerate(factor_names):
                col = pd.Series(X[:, i])
                ic_by_factor[name] = calculator.compute(col, pd.Series(y))
            self.update_ic_batch(ic_by_factor)
        else:
            # IC per group, then record each group's IC as one observation.
            groups = np.asarray(groups)
            for g in np.unique(groups):
                mask = groups == g
                if mask.sum() < calculator.min_samples:
                    continue
                y_g = pd.Series(y[mask])
                ic_by_factor = {}
                for i, name in enumerate(factor_names):
                    col = pd.Series(X[mask, i])
                    ic_by_factor[name] = calculator.compute(col, y_g)
                self.update_ic_batch(ic_by_factor, date=g)

        # Refresh cached weights.
        self.get_weights()
        return self

    def reset(self) -> None:
        """Clear all accumulated IC history, weights, and attribution."""
        self._ic_history.clear()
        self._ic_dates.clear()
        self._weights.clear()
        self._attribution.clear()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def n_observations(self) -> int:
        """Total number of IC observations recorded across all factors."""
        return sum(len(h) for h in self._ic_history.values())

    def ic_summary(self) -> pd.DataFrame:
        """
        Summarise the IC history for every factor.

        Returns:
            A DataFrame indexed by factor name with columns: ``n`` (number of
            observations), ``mean_ic``, ``decayed_ic`` (the value used for
            weighting), and ``weight`` (after thresholding).
        """
        weights = self.get_weights()
        rows = []
        for name, history in self._ic_history.items():
            window = history[-self.ic_lookback:]
            rows.append(
                {
                    "factor": name,
                    "n": len(history),
                    "mean_ic": float(np.mean(window)) if window else 0.0,
                    "decayed_ic": self._decayed_mean(window),
                    "weight": weights.get(name, 0.0),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["factor", "n", "mean_ic", "decayed_ic", "weight"]
            )
        return pd.DataFrame(rows).set_index("factor").sort_values(
            "weight", ascending=False
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        n_active = sum(1 for v in self._weights.values() if v > 0)
        return (
            f"ICWeightedLinear(lookback={self.ic_lookback}, "
            f"min_ic={self.min_ic}, halflife={self.decay_halflife}, "
            f"factors={len(self._ic_history)}, active={n_active})"
        )


def _manual_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of ranks — a scipy-free Spearman fallback."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank ranking (ties get the mean rank)."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Resolve ties by averaging the ranks of equal values.
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks
