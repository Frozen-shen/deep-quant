"""
Alpha158 补全因子 IC 验证 — P3

对新增的 Qlib Alpha158 价量因子 (BETA/RSQR/RESI/IMAX/IMIN/IMXD/WVMA/CORD/
SUMP/SUMN/SUMD/VMA/VSTD) 计算多 horizon 的 Spearman 秩 IC, 验证其预测力。

实现要点:
  - 每只股票的因子只向量化计算一次。
  - 构建 (日期 × 股票) 的宽表, 逐因子/逐 horizon 做截面 Spearman IC,
    避免在 (日期 × 股票) 双层循环里做 DataFrame 查找。

用法: python scripts/run_alpha158_ic.py
输出: data/ic_validation/p3_alpha158_ic.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from data_cache import get_cached_symbols, load_all
from factor_engine import FactorLibrary
from factor_library import ALPHA158_FACTORS

START, END = "2018-01-01", "2022-12-31"
HORIZONS = [5, 10, 20]
MIN_CROSS_SECTION = 10   # 每日最少股票数
MIN_DAYS = 50            # 因子最少有效天数

print("=" * 60)
print("  Alpha158 补全因子 IC 验证 (P3)")
print("=" * 60)

syms = get_cached_symbols()
print(f"股票池: {len(syms)}")
all_data = load_all(syms)
print(f"有效: {len(all_data)}")
symbols = sorted(all_data.keys())

# ── 预计算所有 Alpha158 因子 (向量化, 每只股票一次) ──
factor_names = sorted(ALPHA158_FACTORS.keys())
lib = FactorLibrary.from_config(ALPHA158_FACTORS)
print(f"因子: {len(factor_names)} 个, 预计算中...")

factor_cache = {}   # {symbol: DataFrame(date 索引 × factor)}
for sym in symbols:
    df = all_data[sym]
    try:
        feats = lib.evaluate_all(df)
        feats["date"] = pd.to_datetime(feats["date"])
        feats = feats.set_index("date", drop=False)
        factor_cache[sym] = feats
    except Exception as e:
        print(f"  {sym} 因子计算失败: {e}")
print(f"预计算完成: {len(factor_cache)} 只")
valid_symbols = sorted(factor_cache.keys())

# ── 构建前瞻收益宽表: ret_wide[h] = DataFrame(日期 × 股票) ──
print("构建前瞻收益宽表...")
close_wide = pd.DataFrame({
    sym: all_data[sym].set_index(pd.to_datetime(all_data[sym]["date"]))["close"]
    for sym in valid_symbols
})
ret_wide = {h: close_wide.shift(-h) / close_wide - 1 for h in HORIZONS}

# ── 研究期日期 (每5日采样一次, 降低自相关) ──
all_days = sorted(close_wide.index)
research_days = [d for d in all_days if pd.Timestamp(START) <= d <= pd.Timestamp(END)][::5]
research_idx = pd.DatetimeIndex(research_days)
print(f"研究日: {len(research_days)}")

# ── 逐因子构建宽表并计算各 horizon 的截面 IC ──
# ic_data[horizon][factor] = [ic_per_day, ...]
ic_data = {h: {f: [] for f in factor_names} for h in HORIZONS}

for fi, fname in enumerate(factor_names):
    # 该因子的 (日期 × 股票) 宽表
    fwide = pd.DataFrame({sym: factor_cache[sym][fname] for sym in valid_symbols
                          if fname in factor_cache[sym].columns})
    fsub = fwide.loc[fwide.index.intersection(research_idx)]
    for h in HORIZONS:
        rsub = ret_wide[h].reindex(fsub.index)
        for day in fsub.index:
            fv = fsub.loc[day].to_numpy()
            rv = rsub.loc[day].to_numpy()
            keep = ~(np.isnan(fv) | np.isnan(rv))
            if keep.sum() < MIN_CROSS_SECTION:
                continue
            fvk, rvk = fv[keep], rv[keep]
            if np.std(fvk) < 1e-9:
                continue
            try:
                ic, _ = spearmanr(fvk, rvk)
            except Exception:
                continue
            if not np.isnan(ic):
                ic_data[h][fname].append(ic)
    if (fi + 1) % 13 == 0:
        print(f"  因子 {fi+1}/{len(factor_names)}")

# ── 汇总 ──
results = []
for h in HORIZONS:
    for fname in factor_names:
        ics = ic_data[h][fname]
        if len(ics) < MIN_DAYS:
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        results.append({
            "factor": fname,
            "horizon": h,
            "n_days": len(ics),
            "ic_mean": round(mean_ic, 6),
            "ic_std": round(std_ic, 6),
            "icir": round(icir, 4),
            "pos_ratio": round(float(sum(1 for x in ics if x > 0) / len(ics)), 4),
            "abs_ic_mean": round(abs(mean_ic), 6),
        })

# 按 horizon 分组, 各组内按 |ICIR| 排序
results.sort(key=lambda x: (x["horizon"], -abs(x["icir"])))

out_dir = os.path.join("data", "ic_validation")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "p3_alpha158_ic.json")
payload = {
    "meta": {
        "start": START, "end": END,
        "horizons": HORIZONS,
        "n_symbols": len(valid_symbols),
        "n_research_days": len(research_days),
        "n_factors": len(factor_names),
    },
    "results": results,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

# ── 打印 Top 因子 (各 horizon) ──
for h in HORIZONS:
    rows = [r for r in results if r["horizon"] == h]
    print(f"\n  Horizon={h}d  Top-10 (|ICIR|):")
    for r in rows[:10]:
        sgn = "+" if r["ic_mean"] > 0 else ""
        print(f"    {r['factor']:<12} IC={sgn}{r['ic_mean']:+.4f}  "
              f"ICIR={r['icir']:+.2f}  pos={r['pos_ratio']:.0%}  n={r['n_days']}")

print(f"\n  完成: {len(results)} 条 (因子×horizon)")
print(f"  输出: {out_path}")
