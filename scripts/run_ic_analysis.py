"""
因子 IC 分析 — 仅用研究期数据 (2018-2022), 输出 ic_results.json

用法: python scripts/run_ic_analysis.py
输出: data/ic_results.json (按ICIR排序的因子列表)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_cache import get_cached_symbols, load_all
from factor_engine import FactorLibrary
from factor_library import get_all_factors

# ═══════════════════════════════
# ★ 研究期: 2018-2022 (因子选择只能用这个区间)
# ═══════════════════════════════
START, END = "2018-01-01", "2022-12-31"

print("=" * 60)
print("  因子 IC 分析 — 研究期 2018~2022")
print("=" * 60)

# 加载数据
syms = get_cached_symbols()
print(f"股票池: {len(syms)} 只")
all_data = load_all(syms)
print(f"有效: {len(all_data)} 只")

# 构建因子库 (所有预定义因子)
lib = get_all_factors()
factor_names = sorted(lib.factors.keys())
print(f"因子数: {len(factor_names)}")

# 收集所有日期
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
research_days = [d for d in all_days if pd.Timestamp(START) <= d <= pd.Timestamp(END)][::5]  # 每5天采样
print(f"研究日: {len(research_days)}")

# ── 计算每个因子的每日 Rank IC ──
ic_records = {f: [] for f in factor_names}

for di, today in enumerate(research_days):
    # 收集当天可用股票的前瞻收益
    rets = {}
    for sym in all_data:
        df = all_data[sym]
        mask = df["date"] == today
        if not mask.any(): continue
        pos = df.index[mask][0]
        iloc = df.index.get_loc(pos)
        if iloc + 20 >= len(df): continue  # T+20 前瞻
        fwd = df.iloc[iloc + 20]["close"] / df.iloc[iloc]["close"] - 1
        rets[sym] = fwd

    if len(rets) < 10: continue

    # 计算每个因子的当日值
    syms_today = list(rets.keys())
    ret_arr = np.array([rets[s] for s in syms_today])

    for fname in factor_names:
        factor = lib.factors[fname]
        try:
            # 对每只股票计算因子值 (用120行历史)
            fvals = []
            valid_syms = []
            for sym in syms_today:
                df = all_data[sym]
                mask = df["date"] <= today
                hist = df[mask].tail(120)
                if len(hist) < 60: continue
                fv = factor.evaluate(hist).iloc[-1]
                if pd.isna(fv): continue
                fvals.append(float(fv))
                valid_syms.append(sym)

            if len(fvals) < 10: continue

            fv_arr = np.array(fvals)
            rv_arr = np.array([rets[s] for s in valid_syms])

            # Rank IC (Spearman)
            ic, _ = spearmanr(fv_arr, rv_arr)
            ic_records[fname].append(ic)
        except:
            pass

    if (di + 1) % 50 == 0:
        print(f"  {di+1}/{len(research_days)} days done")

# ── 汇总 ──
results = []
for fname in factor_names:
    ics = ic_records[fname]
    if len(ics) < 50: continue
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    pos_ratio = sum(1 for x in ics if x > 0) / len(ics)
    results.append({
        "factor": fname,
        "n_days": len(ics),
        "ic_mean": round(float(mean_ic), 6),
        "ic_std": round(float(std_ic), 6),
        "icir": round(float(icir), 4),
        "pos_ratio": round(float(pos_ratio), 4),
        "abs_ic_mean": round(float(abs(mean_ic)), 6),
    })

# 按 ICIR 绝对值排序
results.sort(key=lambda x: -abs(x["icir"]))

# 保存
os.makedirs("data", exist_ok=True)
with open("data/ic_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n  完成: {len(results)} 个因子有足够数据")
print(f"\n  Top-20 因子 (ICIR):")
for r in results[:20]:
    sign = "+" if r["ic_mean"] > 0 else ""
    print(f"    {r['factor']:<25} IC={sign}{r['ic_mean']:+.4f}  "
          f"ICIR={sign}{r['icir']:+.2f}  pos={r['pos_ratio']:.0%}")

print(f"\n  输出: data/ic_results.json")
