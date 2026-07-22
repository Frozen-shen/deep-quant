"""
超参数网格搜索 — 在窗口1上搜索最优 LightGBM 参数

搜索空间:
  - max_depth: [4, 5, 6, 7]
  - learning_rate: [0.01, 0.03, 0.05]
  - lambda_l1: [0, 0.5, 1.0]
  - min_data_in_leaf: [20, 30, 50]

评估指标: 验证集 NDCG@3 + NDCG@5

用法: python hyper_search.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from itertools import product
from data_cache import load_all
from factor_scorer import FactorScorer
from ml_ranker import MLRanker

# ════════════════════════════════════
#  配置
# ════════════════════════════════════

SYMBOLS = [
    "688981","002371","002049",
    "002230","300033",
    "300750","002594",
    "601012","300274",
    "600519","000858",
    "601318","600036",
    "300760","600276",
    "600760","601668","300122",
]

TRAIN_START = "2018-01-01"
TRAIN_END = "2020-12-31"

# 搜索空间
PARAM_GRID = {
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05],
    "lambda_l1": [0.0, 0.5, 1.0],
    "min_data_in_leaf": [20, 30, 50],
}

print("=" * 60)
print("  LightGBM 超参数网格搜索")
print("=" * 60)
print(f"  搜索空间: {len(list(product(*PARAM_GRID.values())))} 组合")

# ════════════════════════════════════
#  加载 & 构建训练数据
# ════════════════════════════════════

print("\n[加载] 数据...")
ALL_DATA = load_all(SYMBOLS)
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))

scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())

train_days = [d for d in all_days
              if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)]
train_days = train_days[::3]
print(f"  训练日: {len(train_days)}天")

# 构建全量训练数据 (一次性)
print("[构建] 训练数据...")
X_all, y_all, g_all = [], [], []

for today in train_days:
    sd = {}
    for sym in SYMBOLS:
        if sym not in ALL_DATA:
            continue
        dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
        if len(dt) >= 60:
            sd[sym] = dt

    if len(sd) < 5:
        continue

    # 计算原始因子 + 截面标准化 + 排名标签
    day_feats, day_rets = {}, {}
    for sym, df in sd.items():
        f = scorer.compute_factors(df)
        if len(f) == 0:
            continue
        row = f.iloc[-1]
        close = df["close"].values
        if len(close) < 6:
            continue
        feats = [float(row.get(fn, 0)) for fn in factor_names]
        fwd = close[-1] / close[-6] - 1
        day_feats[sym] = feats
        day_rets[sym] = fwd

    syms = list(day_feats.keys())
    if len(syms) < 5:
        continue

    feats_raw = np.array([day_feats[s] for s in syms])
    mean = feats_raw.mean(axis=0, keepdims=True)
    std = feats_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats_norm = (feats_raw - mean) / std

    rets = np.array([day_rets[s] for s in syms])
    labels = rankdata(rets).astype(int) - 1

    n = len(syms)
    X_all.extend(feats_norm.tolist())
    y_all.extend(labels.tolist())
    g_all.extend([today] * n)

X_all = np.array(X_all)
y_all = np.array(y_all, dtype=int)
g_all = pd.Series(g_all).astype(str).factorize()[0]

print(f"  总样本: {len(X_all)}, 日组数: {len(np.unique(g_all))}")

# ════════════════════════════════════
#  网格搜索
# ════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  网格搜索中...")

best_score = -999
best_params = None
results = []

# 枚举所有组合
keys = list(PARAM_GRID.keys())
values = list(PARAM_GRID.values())
total_combos = 1
for v in values:
    total_combos *= len(v)

combo_idx = 0
for combo in product(*values):
    params = dict(zip(keys, combo))
    combo_idx += 1

    try:
        model = MLRanker(
            n_estimators=200,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            lambda_l1=params["lambda_l1"],
            min_data_in_leaf=params["min_data_in_leaf"],
        )
        model.feature_names = factor_names
        model.fit(X_all, y_all, g_all, val_ratio=0.2)

        # 从模型获取验证集最佳 NDCG
        if model.model is not None:
            evals = model.model.best_score
            if "valid_0" in evals:
                ndcg1 = evals["valid_0"].get("ndcg@1", 0)
                ndcg3 = evals["valid_0"].get("ndcg@3", 0)
                ndcg5 = evals["valid_0"].get("ndcg@5", 0)
                score = ndcg3 + ndcg5  # 综合评分
            else:
                score = -999
        else:
            score = -999

        results.append({**params, "ndcg@1": ndcg1, "ndcg@3": ndcg3,
                        "ndcg@5": ndcg5, "score": score})

        mark = "★" if score > best_score else " "
        print(f"  [{combo_idx}/{total_combos}]{mark} depth={params['max_depth']} "
              f"lr={params['learning_rate']} l1={params['lambda_l1']} "
              f"leaf={params['min_data_in_leaf']} → NDCG@3={ndcg3:.4f} "
              f"NDCG@5={ndcg5:.4f}")

        if score > best_score:
            best_score = score
            best_params = params

    except Exception as e:
        print(f"  [{combo_idx}/{total_combos}] ✗ {params}: {e}")
        results.append({**params, "ndcg@1": -1, "ndcg@3": -1,
                        "ndcg@5": -1, "score": -999})

# ════════════════════════════════════
#  结果
# ════════════════════════════════════

df = pd.DataFrame(results)
df = df.sort_values("score", ascending=False)

print(f"\n{'=' * 70}")
print(f"  网格搜索结果 (Top 10)")
print(f"{'=' * 70}")
print(f"{'rank':<6} {'depth':<6} {'lr':<6} {'l1':<6} {'leaf':<6} "
      f"{'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@5':>8} {'score':>8}")
print("-" * 70)

for i, (_, r) in enumerate(df.head(10).iterrows()):
    print(f"{i+1:<6} {r['max_depth']:<6} {r['learning_rate']:<6.3f} "
          f"{r['lambda_l1']:<6.1f} {r['min_data_in_leaf']:<6} "
          f"{r['ndcg@1']:>8.4f} {r['ndcg@3']:>8.4f} {r['ndcg@5']:>8.4f} "
          f"{r['score']:>8.4f}")

print(f"\n  ★ 最优参数:")
print(f"     n_estimators=200")
print(f"     max_depth={best_params['max_depth']}")
print(f"     learning_rate={best_params['learning_rate']}")
print(f"     lambda_l1={best_params['lambda_l1']}")
print(f"     min_data_in_leaf={best_params['min_data_in_leaf']}")

# 保存结果
out_path = os.path.join(os.path.dirname(__file__), "hyper_search_results.csv")
df.to_csv(out_path, index=False)
print(f"\n  结果已保存: {out_path}")
print("✅ 搜索完成")
