"""
Simple ensemble blending for combining multiple ranking models.

Design note
-----------
Our experiments showed that ensembling three LightGBM models (3x LGB) still
produced -4.6% excess return — ensembling overfit models just averages the
overfitting. The value of this ensemble is therefore to blend the *robust*
IC-weighted linear model with a (lightly weighted) regularized ranker, not to
stack many ML models.

Blend method: each model's raw scores are converted to cross-sectional rank
percentiles (so models on different scales become comparable), then combined
with a weighted average.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _accepted_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the keyword arguments that ``func`` actually accepts.

    Member models have heterogeneous ``fit`` / ``predict`` signatures (the
    IC-weighted linear model takes ``factor_names``; the LightGBM ranker does
    not). Filtering by signature lets the ensemble forward shared arguments
    without crashing on models that do not recognise them, while still
    surfacing genuine errors for arguments a model does accept.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins, C funcs
        return dict(kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)  # accepts **kwargs -> pass everything

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


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Convert raw scores to cross-sectional rank percentiles in [0, 1].

    This puts every model on the same scale before blending. NaN scores are
    assigned the neutral value 0.5.
    """
    scores = np.asarray(scores, dtype=float)
    series = pd.Series(scores)
    ranked = series.rank(pct=True, na_option="keep")
    return ranked.fillna(0.5).to_numpy(dtype=float)


class SimpleEnsemble:
    """
    Blend multiple models with fixed weights.

    Supports any combination of models that expose a ``predict`` method —
    typically :class:`~quant.model.linear.ICWeightedLinear` plus a
    :class:`~quant.model.ranker.LightGBMRanker`. The blend is a weighted
    average of each model's rank-normalised scores.

    Args:
        models: The member models to blend.
        weights: Optional blending weights, one per model. Defaults to equal
            weighting. Weights are normalised to sum to one.

    Raises:
        ValueError: If ``models`` is empty or ``weights`` has the wrong length.

    Example:
        >>> from quant.model.linear import ICWeightedLinear
        >>> from quant.model.ranker import LightGBMRanker
        >>> ens = SimpleEnsemble(
        ...     models=[ICWeightedLinear(), LightGBMRanker()],
        ...     weights=[0.8, 0.2],   # trust the linear model far more
        ... )
    """

    def __init__(
        self,
        models: List,
        weights: Optional[Sequence[float]] = None,
    ):
        if not models:
            raise ValueError("At least one model is required.")
        self.models = list(models)

        if weights is None:
            weights = np.ones(len(self.models), dtype=float)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(self.models):
            raise ValueError(
                f"weights length ({len(weights)}) does not match number of "
                f"models ({len(self.models)})."
            )
        if (weights < 0).any():
            raise ValueError("Blending weights must be non-negative.")

        total = weights.sum()
        if total <= 0:
            raise ValueError("Blending weights must sum to a positive value.")
        self.weights: np.ndarray = weights / total

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, *args, **kwargs) -> "SimpleEnsemble":
        """
        Fit every member model with the same arguments.

        Any ``*args`` / ``**kwargs`` are forwarded to each model's ``fit``.
        Models without a ``fit`` method (e.g. the IC-weighted linear model,
        which has no parameters to learn but still provides a ``fit`` that
        sets IC weights) are handled transparently.
        """
        for model in self.models:
            if hasattr(model, "fit"):
                model.fit(*args, **_accepted_kwargs(model.fit, kwargs))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Blend member predictions into a single score.

        Args:
            X: ``(n_samples, n_features)`` factor matrix.
            **kwargs: Extra keyword arguments (e.g. ``factor_names`` for the
                IC-weighted linear model) forwarded to each member's
                ``predict``. Members that do not accept a keyword silently
                fall back to ``predict(X)``.

        Returns:
            ``(n_samples,)`` blended scores. Higher is better.
        """
        if not self.models:
            raise ValueError("Ensemble has no member models.")

        blended = None
        for model, weight in zip(self.models, self.weights):
            accepted = _accepted_kwargs(model.predict, kwargs)
            raw = np.asarray(model.predict(X, **accepted), dtype=float)
            normed = _rank_normalize(raw)
            contribution = weight * normed
            blended = contribution if blended is None else blended + contribution

        return blended

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_weights(self) -> List[float]:
        """Return the normalised blending weights."""
        return [float(w) for w in self.weights]

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        names = [type(m).__name__ for m in self.models]
        weights = ", ".join(f"{w:.2f}" for w in self.weights)
        return f"SimpleEnsemble(models={names}, weights=[{weights}])"
