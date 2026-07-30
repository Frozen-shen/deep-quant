"""
基线模型阶梯 — L0/L1/L2 用于与 LightGBM (L3) 对打

接口对齐 MLRanker:
  - fit(X, y, groups, val_ratio, sample_weight)
  - predict(X) → scores (越高越好)
"""
import numpy as np


class L0EqualWeight:
    """L0: 等权持有 — 最蠢的基线。不训练，返回随机扰动确保排名分散。"""
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.feature_names = []

    def fit(self, X, y, groups=None, val_ratio=None, sample_weight=None):
        return self

    def predict(self, X):
        # 微弱随机扰动 → 截面排名会有微弱差异 → PortfolioRanker 选 top-k
        # 等价于"随机选股"，用作绝对零基线
        return self.rng.randn(len(X)) * 0.01


class L1SingleFactor:
    """L1: 单因子 Top-K — 最简因子模型。使用 IC 最强的单个因子。"""
    def __init__(self, factor_idx=0):
        """
        Args:
          factor_idx: 用第几个特征列 (默认第0列=skew_20d, ICIR最高)
        """
        self.factor_idx = factor_idx
        self.direction = 1  # 1=因子值越大越好, -1=越小越好
        self.feature_names = []

    def fit(self, X, y, groups=None, val_ratio=None, sample_weight=None):
        # 从训练数据自动推断方向: 计算因子与标签的相关系数
        if len(X) > 0:
            fv = X[:, self.factor_idx]
            corr = np.corrcoef(fv, y)[0, 1]
            self.direction = 1 if corr > 0 else -1
        return self

    def predict(self, X):
        return X[:, self.factor_idx] * self.direction


class L2LinearRanker:
    """L2: Ridge 回归 + 截面排名 — 与 LightGBM 直接对标。"""
    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.feature_names = []

    def fit(self, X, y, groups=None, val_ratio=None, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)
