"""
ML 排序模型 — LightGBM lambdarank 替代固定权重求和 (参考 Qlib LGBModel 87行)

用法:
  from ml_ranker import MLRanker
  ranker = MLRanker()
  ranker.fit(factor_panels, return_panels)        # 训练
  scores = ranker.predict(stock_data_today)        # 预测今日排名
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


class MLRanker:
    """
    LightGBM Lambdarank 排序器 (修复版)。

    学习目标: 给定N只股票的因子值,预测它们的截面排名。
    包含: 时序训练/验证分离 + 早停 + 特征重要性
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = []
        self.feature_importance = {}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None,
            val_ratio: float = 0.2):
        """
        训练排序模型 (时序分割: 前80%训练,后20%验证)。

        参数:
          X: (n_samples, n_features)
          y: (n_samples,) 未来收益率
          groups: (n_samples,) 日期分组
          val_ratio: 验证集比例
        """
        self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        # 时序分割
        n = len(X)
        split_idx = int(n * (1 - val_ratio))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_valid, y_valid = X[split_idx:], y[split_idx:]

        if groups is None:
            groups = np.arange(n) // 10
        g_train = groups[:split_idx]
        g_valid = groups[split_idx:] - groups[split_idx]  # 从0开始

        train_data = lgb.Dataset(X_train, label=y_train, group=_count_groups(g_train))
        valid_data = lgb.Dataset(X_valid, label=y_valid, group=_count_groups(g_valid),
                                  reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 2 ** self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "verbose": -1,
            "early_stopping_rounds": 20,
        }

        self.model = lgb.train(params, train_data, valid_sets=[valid_data])

        # 特征重要性
        if self.model is not None:
            imp = self.model.feature_importance(importance_type="gain")
            self.feature_importance = {
                self.feature_names[i]: imp[i]
                for i in range(min(len(imp), len(self.feature_names)))
            }
            top5 = sorted(self.feature_importance.items(), key=lambda x: -x[1])[:5]
            print(f"  [LightGBM] 训练完成, Top5因子: {[(n, f'{v:.0f}') for n,v in top5]}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数 (可用于排名)。"""
        if self.model is None:
            raise ValueError("模型未训练,请先调用 fit()")
        return self.model.predict(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        """返回排名 (0~1,越大越好)。"""
        scores = self.predict(X)
        return pd.Series(scores).rank(pct=True).values

    def save(self, path: str):
        """保存模型。"""
        if self.model is None:
            return
        import joblib
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    def load(self, path: str):
        """加载模型。"""
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]

    @classmethod
    def from_factor_data(cls, factor_panels: dict, return_panel: pd.DataFrame,
                         **kwargs) -> "MLRanker":
        """
        从因子面板数据训练。

        参数:
          factor_panels: {factor_name: DataFrame(date × symbol)}
          return_panel: DataFrame(date × symbol) 未来收益

        返回: 训练好的 MLRanker
        """
        X_list, y_list, groups_list = [], [], []

        factor_names = list(factor_panels.keys())
        if not factor_names:
            raise ValueError("无有效因子")

        # 找一个参考面板获取日期和股票
        ref = list(factor_panels.values())[0]
        common_dates = ref.index.intersection(return_panel.index)

        group_id = 0
        for d in common_dates:
            # 获取当天的所有股票因子和收益
            stock_data = {}
            for fn in factor_names:
                if d in factor_panels[fn].index:
                    row = factor_panels[fn].loc[d]
                    for sym in row.index:
                        if sym not in stock_data:
                            stock_data[sym] = {}
                        stock_data[sym][fn] = row[sym]

            if d not in return_panel.index:
                continue
            ret_row = return_panel.loc[d]

            features = []
            targets = []
            for sym, factors in stock_data.items():
                if sym not in ret_row.index:
                    continue
                vals = [factors.get(fn, np.nan) for fn in factor_names]
                if any(np.isnan(v) for v in vals):
                    continue
                features.append(vals)
                targets.append(ret_row[sym])

            if len(features) >= 5:
                X_list.extend(features)
                y_list.extend(targets)
                groups_list.extend([group_id] * len(features))
                group_id += 1

        if len(X_list) < 100:
            raise ValueError(f"训练样本不足 ({len(X_list)}), 需要更多数据")

        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups_list)

        ranker = cls(**kwargs)
        ranker.feature_names = factor_names
        ranker.fit(X, y, groups)
        return ranker


def _count_groups(groups: np.ndarray) -> np.ndarray:
    """LightGBM要求的group计数。"""
    _, counts = np.unique(groups, return_counts=True)
    return counts


def demo():
    """演示: 用随机数据训练 → 预测。"""
    print("MLRanker 演示...")
    np.random.seed(42)
    X = np.random.randn(500, 10)  # 500样本,10个因子
    y = X[:, 0] * 0.3 + X[:, 1] * 0.2 + np.random.randn(500) * 0.5
    groups = np.repeat(np.arange(50), 10)  # 50天,每天10只股票

    ranker = MLRanker(n_estimators=50, max_depth=4)
    ranker.fit(X, y, groups)

    pred = ranker.predict(X[:10])
    print(f"  训练完成, 预测前10: {pred.round(3)}")
    print(f"  排名: {ranker.rank(X[:10]).round(2)}")
    print("✅ MLRanker 正常")


if __name__ == "__main__":
    demo()
