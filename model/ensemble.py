"""Ensemble model: blend multiple rankers for robustness."""
import numpy as np
from typing import List, Dict, Optional


class EnsembleRanker:
    """Stacking ensemble of multiple rankers with different seeds/types."""

    def __init__(self, members: List[str] = None, blend_method: str = 'equal'):
        self.members_config = members or ['lgb', 'lgb', 'lgb']
        self.blend_method = blend_method
        self.models = []
        self.weights = None
        self._seeds = [42, 123, 456, 789, 1024]

    def fit(self, X, y, groups, val_ratio=0.15, sample_weight=None):
        from ml_ranker import MLRanker
        self.models = []

        for i, mtype in enumerate(self.members_config):
            if mtype == 'lgb':
                model = MLRanker(
                    n_estimators=400, max_depth=4,
                    learning_rate=0.03, lambda_l1=0.5,
                    min_data_in_leaf=50
                )
            elif mtype == 'l2':
                try:
                    from model.baselines import L2LinearRanker
                    model = L2LinearRanker(alpha=1.0)
                except ImportError:
                    # sklearn not available, use another lgb
                    model = MLRanker(
                        n_estimators=300, max_depth=3,
                        learning_rate=0.05, lambda_l1=1.0,
                        min_data_in_leaf=80
                    )
            else:
                model = MLRanker(n_estimators=400, max_depth=5,
                                 learning_rate=0.03, lambda_l1=0.3,
                                 min_data_in_leaf=40)

            model.fit(X, y, groups, val_ratio=val_ratio, sample_weight=sample_weight)
            self.models.append(model)

        if self.blend_method == 'equal':
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        return self

    def predict(self, X) -> np.ndarray:
        if not self.models:
            return np.zeros(X.shape[0])
        preds = np.array([m.predict(X) for m in self.models])
        if self.weights is None:
            return preds.mean(axis=0)
        return (preds * self.weights[:, None]).sum(axis=0)

    def feature_importance(self) -> Dict[str, int]:
        agg = {}
        for m in self.models:
            if hasattr(m, 'feature_importance'):
                fi = m.feature_importance()
                if fi:
                    for k, v in fi.items():
                        agg[k] = agg.get(k, 0) + v
        return agg
