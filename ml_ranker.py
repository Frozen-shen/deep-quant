"""
ML 排序模型 — LightGBM Lambdarank (honest)

学习目标: 给定每日截面股票因子值, 预测截面排名 (Top-K选股)。
关键改进:
  1. objective=lambdarank (直接优化排序,N个pairwise loss)
  2. 标签=截面排名 (整数, 0=最差, N-1=最好)
  3. 按日期分组切分训练/验证 (保证同一天股票不跨split)
  4. L1正则化 + min_data_in_leaf 防过拟合

用法:
  from ml_ranker import MLRanker
  ranker = MLRanker()
  ranker.fit(X, y, groups)        # 训练
  scores = ranker.predict(X_new)  # 预测分数
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


class MLRanker:
    """
    LightGBM Lambdarank 截面排序器。

    标签约定:
      y: 整数截面排名 (0=最差, N-1=最好), 用于 lambdarank
      groups: 日期ID, 同一天的样本属于同一个 ranking group

    排序逻辑:
      - 分数越高 → 排名越靠前 → 优先买入
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, lambda_l1: float = 0.5,
                 min_data_in_leaf: int = 30):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.min_data_in_leaf = min_data_in_leaf
        self.model = None
        self.feature_names = []
        self.feature_importance = {}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None,
            val_ratio: float = 0.2, sample_weight: np.ndarray = None):
        """
        训练 Lambdarank 模型。

        参数:
          X: (n_samples, n_features) 截面标准化后的因子值
          y: (n_samples,) 整数截面排名 (0=最差, N-1=最好)
          groups: (n_samples,) 日期分组ID, 同一天 = 同一 ranking group
          val_ratio: 按日期数 (而非样本数) 分割的验证比例
          sample_weight: (n_samples,) 样本权重 (DEnsemble迭代重训练用)

        NOTE: 切分按日期组边界, 而非样本索引。
              保证同一天的所有股票不在 train/valid 之间分裂。
        """
        if not self.feature_names:
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        if groups is None:
            groups = np.arange(len(X)) // 10

        # ── 按 group (日期) 边界切分 ──
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        split_g = int(n_groups * (1 - val_ratio))

        train_groups_set = unique_groups[:split_g]
        valid_groups_set = unique_groups[split_g:]

        train_mask = np.isin(groups, train_groups_set)
        valid_mask = np.isin(groups, valid_groups_set)

        X_train, y_train = X[train_mask], y[train_mask]
        X_valid, y_valid = X[valid_mask], y[valid_mask]
        g_train_raw = groups[train_mask]
        g_valid_raw = groups[valid_mask]

        # 样本权重 (如果有)
        train_w = sample_weight[train_mask] if sample_weight is not None else None
        valid_w = sample_weight[valid_mask] if sample_weight is not None else None

        # 重新编码 group ID (从0开始连续)
        g_train = pd.Series(g_train_raw).astype(str).factorize()[0]
        g_valid = pd.Series(g_valid_raw).astype(str).factorize()[0]

        print(f"  [MLRanker] 训练组: {len(g_train)}样本/{len(np.unique(g_train))}天, "
              f"验证组: {len(g_valid)}样本/{len(np.unique(g_valid))}天")

        train_data = lgb.Dataset(X_train, label=y_train,
                                 group=_count_groups(g_train),
                                 weight=train_w if train_w is not None else None)
        valid_data = lgb.Dataset(X_valid, label=y_valid,
                                 group=_count_groups(g_valid),
                                 weight=valid_w if valid_w is not None else None,
                                 reference=train_data)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "boosting_type": "gbdt",
            "num_leaves": 2 ** self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "lambda_l1": self.lambda_l1,
            "min_data_in_leaf": self.min_data_in_leaf,
            "verbose": -1,
            "early_stopping_rounds": 20,
            "seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "deterministic": True,
        }

        self.model = lgb.train(params, train_data, valid_sets=[valid_data])

        # ── 特征重要性 ──
        if self.model is not None:
            imp = self.model.feature_importance(importance_type="gain")
            self.feature_importance = {
                self.feature_names[i]: imp[i]
                for i in range(min(len(imp), len(self.feature_names)))
            }
            top5 = sorted(self.feature_importance.items(),
                          key=lambda x: -x[1])[:5]
            nonzero = sum(1 for v in imp if v > 0)
            print(f"  [MLRanker] 完成, 非零重要性因子: {nonzero}/{len(imp)}, "
                  f"Top5: {[(n, f'{v:.0f}') for n, v in top5]}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测分数 (越高越好, 用于排名选股)。"""
        if self.model is None:
            raise ValueError("模型未训练, 请先调用 fit()")
        return self.model.predict(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        """返回排名 (0~1, 越大越好)。"""
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
        从因子面板数据训练 (兼容旧接口)。

        参数:
          factor_panels: {factor_name: DataFrame(date × symbol)}
          return_panel: DataFrame(date × symbol) 未来收益

        返回: 训练好的 MLRanker
        """
        X_list, y_list, groups_list = [], [], []

        factor_names = list(factor_panels.keys())
        if not factor_names:
            raise ValueError("无有效因子")

        ref = list(factor_panels.values())[0]
        common_dates = ref.index.intersection(return_panel.index)

        group_id = 0
        for d in common_dates:
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

            features, targets, syms = [], [], []
            for sym, factors in stock_data.items():
                if sym not in ret_row.index:
                    continue
                vals = [factors.get(fn, np.nan) for fn in factor_names]
                if any(np.isnan(v) for v in vals):
                    continue
                features.append(vals)
                targets.append(ret_row[sym])
                syms.append(sym)

            if len(features) >= 5:
                # 截面特征标准化
                feats = np.array(features)
                mean = feats.mean(axis=0, keepdims=True)
                std = feats.std(axis=0, keepdims=True)
                std[std == 0] = 1.0
                feats = (feats - mean) / std

                # 截面排名标签 (lambdarank)
                from scipy.stats import rankdata
                labels = rankdata(np.array(targets)) - 1  # 0~N-1

                X_list.extend(feats.tolist())
                y_list.extend(labels.tolist())
                groups_list.extend([group_id] * len(features))
                group_id += 1

        if len(X_list) < 100:
            raise ValueError(f"训练样本不足 ({len(X_list)}), 需要更多数据")

        X = np.array(X_list)
        y = np.array(y_list, dtype=int)
        groups = np.array(groups_list)

        ranker = cls(**kwargs)
        ranker.feature_names = factor_names
        ranker.fit(X, y, groups)
        return ranker


def _count_groups(groups: np.ndarray) -> np.ndarray:
    """LightGBM 要求的 group 计数 (每个 group 的样本数)。"""
    _, counts = np.unique(groups, return_counts=True)
    return counts


# ════════════════════════════════════
#  DEnsemble: 迭代样本重加权 (Qlib DEnsembleModel)
# ════════════════════════════════════

class DEnsembleRanker:
    """
    迭代集成排序器: N个子模型, 每次重训时给"难样本"更高权重。
    
    原理:
      1. Train model_0 with equal weights
      2. Predict → compute per-sample rank error within each group
      3. Assign higher weight to samples with larger errors
      4. Train model_1 with updated weights
      5. Repeat N times → weighted average prediction
    """

    def __init__(self, n_models: int = 3, alpha: float = 0.5,
                 **model_kwargs):
        self.n_models = n_models
        self.alpha = alpha  # 难样本权重放大系数
        self.model_kwargs = model_kwargs
        self.models = []
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None,
            val_ratio: float = 0.2):
        """迭代训练N个子模型。"""
        n = len(X)
        weights = np.ones(n)

        for i in range(self.n_models):
            print(f"  [DEnsemble] 训练子模型 {i+1}/{self.n_models}...")
            m = MLRanker(**self.model_kwargs)
            m.feature_names = self.feature_names or [f"f_{j}" for j in range(X.shape[1])]
            m.fit(X, y, groups, val_ratio, sample_weight=weights)
            self.models.append(m)
            self.feature_names = m.feature_names

            if i == self.n_models - 1:
                break  # 最后一轮不需要更新权重

            # ── 计算难样本权重 ──
            preds = m.predict(X)
            # 在每组内计算排名误差
            unique_g = np.unique(groups)
            errors = np.zeros(n)
            for g in unique_g:
                mask = groups == g
                if mask.sum() < 2:
                    continue
                group_preds = preds[mask]
                group_labels = y[mask]
                # 预测排名 vs 真实排名 的绝对差异
                pred_rank = pd.Series(group_preds).rank(pct=True).values
                true_rank = pd.Series(group_labels).rank(pct=True).values
                errors[mask] = np.abs(pred_rank - true_rank)

            # 归一化误差 → 新权重
            if errors.max() > 0:
                errors = errors / errors.max()
                weights = 1.0 + self.alpha * errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权平均预测。"""
        if not self.models:
            raise ValueError("模型未训练")
        preds = np.zeros(len(X))
        for m in self.models:
            preds += m.predict(X)
        return preds / len(self.models)

    @property
    def model(self):
        """兼容单模型接口: 返回最后一个子模型。"""
        return self.models[-1].model if self.models else None

    @property
    def feature_importance(self):
        """返回最后一个子模型的特征重要性。"""
        return self.models[-1].feature_importance if self.models else {}


def demo():
    """演示 Lambdarank 训练。"""
    print("MLRanker (Lambdarank) 演示...")
    np.random.seed(42)

    # 模拟数据: 50天 × 10只股票 × 5个因子
    X_list, y_list, g_list = [], [], []
    for day in range(50):
        X_day = np.random.randn(10, 5)
        # 第一个因子有微弱预测力
        true_score = X_day[:, 0] * 0.5 + np.random.randn(10) * 0.5
        from scipy.stats import rankdata
        labels = rankdata(true_score) - 1  # 0~9
        X_list.extend(X_day.tolist())
        y_list.extend(labels.tolist())
        g_list.extend([day] * 10)

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    groups = np.array(g_list)

    ranker = MLRanker(n_estimators=50, max_depth=4, lambda_l1=0.0)
    ranker.fit(X, y, groups)

    pred = ranker.predict(X[:10])
    print(f"  训练完成, 预测前10: {pred.round(3)}")
    print(f"  排名: {ranker.rank(X[:10]).round(2)}")
    print("✅ MLRanker (Lambdarank) 正常")


if __name__ == "__main__":
    demo()
