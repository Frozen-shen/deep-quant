"""
基线模型阶梯 — L0/L1/L2 用于与 LightGBM (L3) 对打

接口对齐 MLRanker:
  - fit(X, y, groups, val_ratio, sample_weight)
  - predict(X) → scores (越高越好)
"""
import numpy as np


class L0EqualWeight:
    """L0: 等权持有 — 最蠢的基线。买入后永不调仓。"""
    def __init__(self):
        self.feature_names = []

    def fit(self, X, y, groups=None, val_ratio=None, sample_weight=None):
        return self

    def predict(self, X):
        # 全零分数 → 排名由dict稳定顺序决定 → 同批股票永远在前 → 永不调仓
        return np.zeros(len(X))


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
