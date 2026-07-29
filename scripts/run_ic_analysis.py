"""
因子 IC 分析 v2 — 用 FactorCache 预计算加速

用法: python scripts/run_ic_analysis.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_cache import get_cached_symbols, load_all
from factor_scorer import FactorScorer
from factor_cache import FactorCache

START, END = "2018-01-01", "2022-12-31"
LABEL_HORIZON = 20

print("=" * 60)
print("  IC分析 v2 (FactorCache加速)")
print("=" * 60)

syms = get_cached_symbols()
print(f"股票池: {len(syms)}")
all_data = load_all(syms)
print(f"有效: {len(all_data)}")

# ★ 预计算所有因子
scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())
cache = FactorCache(scorer, factor_names)
cache.precompute(all_data)
print(f"因子: {len(factor_names)} 个, 预计算完成")

# 收集研究期日期
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
research_days = [d for d in all_days if pd.Timestamp(START) <= d <= pd.Timestamp(END)][::5]
print(f"研究日: {len(research_days)}")

# ── 对每个因子, 收集所有日的IC ──
ic_data = {f: {"values": [], "daily_count": 0} for f in factor_names}

for di, today in enumerate(research_days):
    # 前瞻收益
    rets = {}
    for sym in all_data:
        df = all_data[sym]
        mask = df["date"] == today
        if not mask.any(): continue
        ip = df.index.get_loc(df.index[mask][0])
        if ip + LABEL_HORIZON >= len(df): continue
        fwd = df.iloc[ip + LABEL_HORIZON]["close"] / df.iloc[ip]["close"] - 1
        rets[sym] = fwd
    if len(rets) < 10: continue

    # 收集所有因子的当日值 (从预计算缓存)
    fvals_all = {f: [] for f in factor_names}
    valid_syms = []
    for sym in rets:
        feats = cache.get_features(sym, today)
        if feats is None: continue
        valid_syms.append(sym)
        for fi, fname in enumerate(factor_names):
            fvals_all[fname].append(feats[fi])
    if len(valid_syms) < 10: continue

    ret_arr = np.array([rets[s] for s in valid_syms])

    # 对每个因子计算 IC
    for fname in factor_names:
        fv = np.array(fvals_all[fname])
        if np.std(fv) < 1e-9: continue  # 常数跳过
        try:
            ic, _ = spearmanr(fv, ret_arr)
            ic_data[fname]["values"].append(ic)
            ic_data[fname]["daily_count"] += 1
        except: pass

    if (di + 1) % 50 == 0:
        print(f"  {di+1}/{len(research_days)} d")

# ── 汇总 ──
results = []
for fname in factor_names:
    ics = ic_data[fname]["values"]
    if len(ics) < 50: continue
    mean_ic = np.mean(ics); std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    results.append({
        "factor": fname,
        "n_days": len(ics),
        "ic_mean": round(float(mean_ic), 6),
        "ic_std": round(float(std_ic), 6),
        "icir": round(float(icir), 4),
        "pos_ratio": round(float(sum(1 for x in ics if x > 0) / len(ics)), 4),
        "abs_ic_mean": round(float(abs(mean_ic)), 6),
    })

results.sort(key=lambda x: -abs(x["icir"]))

os.makedirs("data", exist_ok=True)
with open("data/ic_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n  完成: {len(results)} 因子")
print(f"  Top-10 (ICIR):")
for r in results[:10]:
    sgn = "+" if r["ic_mean"] > 0 else ""
    print(f"  {r['factor']:<25} IC={sgn}{r['ic_mean']:+.4f}  ICIR={sgn}{r['icir']:+.2f}  pos={r['pos_ratio']:.0%}")
print(f"\n  输出: data/ic_results.json")
