"""
EXTREMELY regularized LightGBM ranker — a FALLBACK model only.

Why this exists
---------------
Our backtests showed that the old LightGBM configuration destroyed alpha:

    Old (L3):  n_estimators=600, max_depth=5, min_data_in_leaf=40,
               learning_rate=0.03, lambda_l1=0.3
               -> -5.50% excess, IR=-0.364  (catastrophic overfitting)

On ~38k samples, 600 trees of depth 5 memorise noise, and the resulting
high-turnover predictions are then decimated by 30bp transaction costs.

This class is the opposite philosophy. Every knob is turned towards
under-fitting rather than over-fitting:

    n_estimators:       50   (was 600)  <- 12x fewer trees
    max_depth:           2   (was 5)    <- no complex interactions
    min_data_in_leaf:  500   (was 40)   <- 12x more conservative leaves
    learning_rate:    0.01   (was 0.03) <- slower, gentler learning
    lambda_l1:         2.0   (was 0.3)  <- much stronger L1 sparsity
    feature_fraction:  0.5              <- random feature subset per tree
    bagging_fraction:  0.7              <- random sample subset per iteration
    early_stopping:   10               <- stop the moment gains dry up

It should ONLY be used if the IC-weighted linear model underperforms, and
even then it is worth asking whether equal weight would be better still.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

logger = logging.getLogger(__name__)


# Hyper-parameters tuned to *under-fit* on purpose. See module docstring.
DEFAULT_PARAMS: Dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],
    "max_position": 80,
    "lambdarank_truncation_level": 80,
    "boosting_type": "gbdt",
    # --- extreme regularisation ---
    "n_estimators": 50,
    "num_leaves": 4,          # depth 2 -> at most 4 leaves
    "max_depth": 2,
    "learning_rate": 0.01,
    "min_data_in_leaf": 500,
    "lambda_l1": 2.0,
    "lambda_l2": 1.0,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "early_stopping_rounds": 10,
    # --- reproducibility ---
    "verbose": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "deterministic": True,
}


def _group_counts(groups: np.ndarray) -> np.ndarray:
    """
    Convert an array of group ids into the per-group sample counts that
    LightGBM's ranking objective expects (one count per group, in order).
    """
    # Preserve first-appearance order so counts line up with the data rows.
    _, counts = np.unique(groups, return_counts=True)
    return counts


class LightGBMRanker:
    """
    EXTREMELY regularized LightGBM model for cross-sectional ranking.

    Key differences from the old system (which overfit catastrophically):
        - n_estimators: 50 (was 600)   <- 12x reduction
        - max_depth: 2 (was 5)         <- prevents complex interactions
        - min_data_in_leaf: 500 (was 40) <- 12x more conservative
        - learning_rate: 0.01 (was 0.03) <- slower learning
        - lambda_l1: 2.0 (was 0.3)     <- stronger L1
        - feature_fraction: 0.5        <- random subset per tree
        - bagging_fraction: 0.7        <- random subset per iteration
        - early_stopping_rounds: 10    <- stop when no improvement

    This model should ONLY be used if IC-weighted linear underperforms.

    Args:
        config: Optional dict of LightGBM parameters that override
            :data:`DEFAULT_PARAMS`. A ``feature_names`` key may also be
            supplied to label the feature columns.

    Raises:
        ImportError: If ``lightgbm`` is not installed.
    """

    def __init__(self, config: Optional[dict] = None):
        if lgb is None:  # pragma: no cover
            raise ImportError(
                "lightgbm is required for LightGBMRanker but is not installed."
            )

        config = dict(config or {})
        self.feature_names: List[str] = list(config.pop("feature_names", []))

        # Merge caller overrides on top of the conservative defaults.
        self.params: Dict = dict(DEFAULT_PARAMS)
        self.params.update(config)

        self.model: Optional["lgb.Booster"] = None
        self._feature_importance: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> "LightGBMRanker":
        """
        Fit the ranking model.

        Args:
            X: ``(n_samples, n_features)`` factor values. NaNs are handled
                natively by LightGBM.
            y: ``(n_samples,)`` relevance labels. For lambdarank these are
                typically integer cross-sectional ranks (0 = worst).
            groups: ``(n_samples,)`` group ids (e.g. date ids). Samples from
                the same date must share a group so that ranking is learned
                within, not across, cross-sections.
            X_val, y_val, groups_val: Optional held-out validation set used
                for early stopping. If omitted, a validation slice is carved
                from the tail of the training data (by group boundary) so no
                cross-section is split across train/validation.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        groups = np.asarray(groups)

        n_features = X.shape[1]
        if not self.feature_names:
            self.feature_names = [f"f_{i}" for i in range(n_features)]

        # --- Build train / validation split -------------------------------
        if X_val is None:
            X_train, y_train, g_train, X_valid, y_valid, g_valid = (
                self._time_split(X, y, groups)
            )
        else:
            X_train, y_train, g_train = X, y, groups
            X_valid = np.asarray(X_val, dtype=float)
            y_valid = np.asarray(y_val)
            g_valid = np.asarray(groups_val)

        # Re-encode group ids so they are dense and ordered from 0.
        g_train_enc = self._encode_groups(g_train)
        g_valid_enc = self._encode_groups(g_valid)

        logger.info(
            "[LightGBMRanker] train=%d samples/%d groups, "
            "valid=%d samples/%d groups",
            len(g_train_enc),
            len(np.unique(g_train_enc)),
            len(g_valid_enc),
            len(np.unique(g_valid_enc)),
        )

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=_group_counts(g_train_enc),
            feature_name=self.feature_names,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=_group_counts(g_valid_enc),
            feature_name=self.feature_names,
            reference=train_data,
            free_raw_data=False,
        )

        # --- Train --------------------------------------------------------
        params = {k: v for k, v in self.params.items()}
        num_boost_round = int(params.pop("n_estimators", 50))
        early_stopping = params.pop("early_stopping_rounds", 10)

        # LightGBM's lambdarank maps each integer label to a gain. The default
        # mapping table is too small for large cross-sections (e.g. 40+ symbols
        # produce labels 0..39), so we size it to the data explicitly using the
        # standard exponential gain 2^label - 1.
        max_label = int(np.max(y_train)) if len(y_train) else 0
        params["label_gain"] = [float(2 ** i - 1) for i in range(max_label + 1)]
        # max_position must be able to address the largest relevance grade.
        params["max_position"] = max(
            int(params.get("max_position", 80)), max_label + 1
        )

        callbacks = [lgb.log_evaluation(period=-1)]
        if early_stopping and early_stopping > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        # --- Feature importance ------------------------------------------
        self._record_importance()
        return self

    @staticmethod
    def _time_split(
        X: np.ndarray, y: np.ndarray, groups: np.ndarray, val_ratio: float = 0.15
    ):
        """
        Split by group (date) boundary so no cross-section straddles the
        train/validation divide. The most recent ``val_ratio`` of groups is
        held out for validation.
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        split_idx = max(1, int(round(n_groups * (1 - val_ratio))))

        train_groups = set(unique_groups[:split_idx].tolist())
        valid_groups = set(unique_groups[split_idx:].tolist())

        train_mask = np.array([g in train_groups for g in groups])
        valid_mask = np.array([g in valid_groups for g in groups])

        # Guarantee a non-empty validation set even with very few groups.
        if not valid_mask.any():
            valid_mask = train_mask.copy()
            train_mask = ~valid_mask

        return (
            X[train_mask],
            y[train_mask],
            groups[train_mask],
            X[valid_mask],
            y[valid_mask],
            groups[valid_mask],
        )

    @staticmethod
    def _encode_groups(groups: np.ndarray) -> np.ndarray:
        """Map arbitrary group ids to dense integers preserving order."""
        return pd.Series(groups).astype(str).factorize()[0].astype(np.int32)

    def _record_importance(self) -> None:
        if self.model is None:
            self._feature_importance = {}
            return
        gains = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name() or self.feature_names
        self._feature_importance = {
            names[i]: float(gains[i]) for i in range(len(gains))
        }
        nonzero = sum(1 for v in gains if v > 0)
        logger.info(
            "[LightGBMRanker] trained with %d boosting rounds, "
            "%d/%d features used",
            self.model.num_trees(),
            nonzero,
            len(gains),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ranking scores. Higher is better.

        Args:
            X: ``(n_samples, n_features)`` factor values. NaNs are handled
                natively by LightGBM.

        Returns:
            ``(n_samples,)`` predicted scores.
        """
        if self.model is None:
            raise ValueError("Model is not trained; call fit() first.")
        X = np.asarray(X, dtype=float)
        return self.model.predict(X)

    def feature_importance(self) -> Dict[str, float]:
        """
        Gain-based feature importance from the fitted model.

        Returns:
            Mapping of feature name -> total gain. Empty if untrained.
        """
        return dict(self._feature_importance)

    def get_params(self) -> dict:
        """
        Return the effective parameter set (defaults merged with overrides),
        including ``feature_names`` for round-tripping.
        """
        out = dict(self.params)
        out["n_estimators"] = self.params.get("n_estimators", 50)
        out["feature_names"] = list(self.feature_names)
        return out

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        trained = self.model is not None
        return (
            f"LightGBMRanker(n_estimators={self.params.get('n_estimators')}, "
            f"max_depth={self.params.get('max_depth')}, "
            f"min_data_in_leaf={self.params.get('min_data_in_leaf')}, "
            f"trained={trained})"
        )
