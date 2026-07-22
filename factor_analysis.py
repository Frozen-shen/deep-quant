"""
因子 IC/ICIR 分析 v2 — 使用 data_cache 加速

Spearman Rank IC: 因子值与未来N日收益的截面相关性
ICIR = IC均值 / IC标准差

输出: 每个因子- horizon 组合的预测力, 用于因子筛选

用法:
  python factor_analysis.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy import stats
from data_cache import load_all
from factor_scorer import FactorScorer

# ════════════════════════════════════
#  配置
# ════════════════════════════════════

# 使用全部30只缓存股票
SYMBOLS = [
    "688981","002371","603986","002049","688012","300782","688396",
    "300033","002230","688111","300454","688561","300750","002594","601012",
    "300274","688005","600519","000858","000568","002714","601318","600036",
    "000001","300760","600276","300122","688180","600760","601668",
]
START, END = "2024-01-01", "2026-07-10"
FORWARD_RETURNS = [3, 5, 10, 20]  # 未来N日收益


def main():
    print("=" * 60)
    print("  因子 IC/ICIR 分析 v2 (data_cache加速)")
    print(f"  股票: {len(SYMBOLS)}只 | 区间: {START}~{END}")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/3] 加载缓存数据...")
    ALL_DATA = load_all(SYMBOLS)
    print(f"  成功加载: {len(ALL_DATA)}/{len(SYMBOLS)} 只")

    # 获取所有交易日
    all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))
    start_dt = pd.Timestamp(START)
    end_dt = pd.Timestamp(END)
    trading_days = [d for d in all_days if start_dt <= d <= end_dt]
    print(f"  交易日: {len(trading_days)}天")

    scorer = FactorScorer.from_preset("ic_optimized")
    factor_names = sorted(scorer.factor_weights.keys())
    print(f"  因子: {len(factor_names)}个")

    # ── 构建因子面板 ──
    print("\n[2/3] 构建因子面板 + 未来收益...")
    # factor_panels: {factor_name: [(date, symbol, value), ...]}
    factor_panels = {fn: [] for fn in factor_names}
    # 未来收益: {horizon: [(date, symbol, value), ...]}
    fwd_panels = {h: [] for h in FORWARD_RETURNS}

    total = len(trading_days)
    for di, today in enumerate(trading_days):
        if di % 50 == 0:
            print(f"  ... {di}/{total}")

        # 收集当天可用股票
        sd = {}
        for sym in SYMBOLS:
            if sym not in ALL_DATA:
                continue
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 50:
                sd[sym] = dt

        if len(sd) < 5:
            continue

        # 计算每个股票的因子值
        for sym, df in sd.items():
            f = scorer.compute_factors(df)
            if len(f) == 0:
                continue
            row = f.iloc[-1]
            for fn in factor_names:
                val = row.get(fn)
                if not (val is None or (isinstance(val, float) and np.isnan(val))):
                    factor_panels[fn].append({"date": today, "symbol": sym, "value": float(val)})

        # 计算未来收益
        for sym, df in sd.items():
            close_vals = df["close"].values
            if len(close_vals) < 21:
                continue
            today_close = close_vals[-1]
            for h in FORWARD_RETURNS:
                if len(close_vals) > h:
                    fwd_ret = close_vals[-1] / close_vals[-(h+1)] - 1
                    if not np.isnan(fwd_ret):
                        fwd_panels[h].append({"date": today, "symbol": sym, "value": fwd_ret})

    print(f"  收集完成: {len(trading_days)}天")

    # ── 转为 DataFrame ──
    print("\n[3/3] 计算 IC...")

    # 因子面板 → {name: DataFrame(date × symbol)}
    fp_dict = {}
    for fn in factor_names:
        if len(factor_panels[fn]) < 100:
            continue
        dfp = pd.DataFrame(factor_panels[fn])
        dfp["date"] = pd.to_datetime(dfp["date"])
        fp_dict[fn] = dfp.pivot(index="date", columns="symbol", values="value")

    # 收益面板 → {horizon: DataFrame(date × symbol)}
    rp_dict = {}
    for h, data in fwd_panels.items():
        dfr = pd.DataFrame(data)
        dfr["date"] = pd.to_datetime(dfr["date"])
        rp_dict[h] = dfr.pivot(index="date", columns="symbol", values="value")

    print(f"  有效因子面板: {len(fp_dict)}")
    print(f"  收益率面板: {len(rp_dict)}")

    # ── 计算 IC ──
    all_ic_results = []

    for fname, fpanel in fp_dict.items():
        for h, rpanel in rp_dict.items():
            ic_values = []
            common_dates = fpanel.index.intersection(rpanel.index)

            for d in common_dates:
                f_row = fpanel.loc[d].dropna()
                r_row = rpanel.loc[d].dropna()
                common_syms = f_row.index.intersection(r_row.index)
                if len(common_syms) < 5:
                    continue

                ic, _ = stats.spearmanr(f_row[common_syms], r_row[common_syms])
                if not np.isnan(ic):
                    ic_values.append(ic)

            if len(ic_values) < 50:
                continue

            ic_arr = np.array(ic_values)
            ic_mean = ic_arr.mean()
            ic_std = ic_arr.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0

            all_ic_results.append({
                "factor": fname,
                "horizon": f"{h}d",
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_pos_ratio": (ic_arr > 0).mean(),
                "abs_ic_mean": abs(ic_mean),
                "n_days": len(ic_arr),
            })

    df_results = pd.DataFrame(all_ic_results)
    df_results = df_results.sort_values("abs_ic_mean", ascending=False)

    # ════════════════════════════════════
    #  输出
    # ════════════════════════════════════

    # ── 详细表格 ──
    print(f"\n{'=' * 85}")
    print(f"  IC 分析结果 (按 |IC| 降序, 前30)")
    print(f"{'=' * 85}")
    header = f"{'因子':<25} {'h':>5} {'IC均值':>8} {'ICIR':>8} {'IC>0%':>7} {'N':>6}"
    print(header)
    print("-" * 65)

    for _, r in df_results.head(30).iterrows():
        mark = "✅" if abs(r["icir"]) > 0.5 else ("⚠️" if abs(r["icir"]) > 0.3 else "  ")
        print(f"  {mark} {r['factor']:<25} {r['horizon']:>5} "
              f"{r['ic_mean']:>+8.4f} {r['icir']:>+8.3f} "
              f"{r['ic_pos_ratio']*100:>6.0f}% {r['n_days']:>5}")

    # ── 每个因子的最佳 horizon ──
    print(f"\n{'=' * 85}")
    print(f"  因子最佳 horizon 汇总 (按 ICIR)")
    print(f"{'=' * 85}")
    best_per_factor = df_results.loc[
        df_results.groupby("factor")["abs_ic_mean"].idxmax()
    ].sort_values("abs_ic_mean", ascending=False)

    print(f"{'因子':<25} {'best_h':>6} {'IC均值':>8} {'ICIR':>8} {'IC>0%':>7} {'判定':>4}")
    print("-" * 65)

    kept_factors = []
    dropped_factors = []

    for _, r in best_per_factor.iterrows():
        abs_icir = abs(r["icir"])

        if abs_icir > 0.3:
            mark = "✅ 强"
            kept_factors.append(r["factor"])
        elif abs_icir > 0.10:
            mark = "⚠️ 中"
            kept_factors.append(r["factor"])
        else:
            mark = "❌ 弱"
            dropped_factors.append(r["factor"])

        print(f"  {mark:<5} {r['factor']:<25} {r['horizon']:>6} "
              f"{r['ic_mean']:>+8.4f} {r['icir']:>+8.3f} "
              f"{r['ic_pos_ratio']*100:>6.0f}%")

    # ── 因子保留/剔除建议 ──
    print(f"\n{'=' * 85}")
    print(f"  因子筛选建议")
    print(f"{'=' * 85}")
    print(f"  ✅ 保留 ({len(kept_factors)}个, ICIR > 0.10):")
    print(f"     {', '.join(kept_factors)}")
    print(f"  ❌ 剔除 ({len(dropped_factors)}个, ICIR ≤ 0.10):")
    print(f"     {', '.join(dropped_factors) if dropped_factors else '(无)'}")

    # ── 保存结果 ──
    out_path = os.path.join(os.path.dirname(__file__), "factor_ic_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n  结果已保存: {out_path}")

    # ── 生成建议的 ic_optimized_v2 预设 ──
    print(f"\n{'=' * 85}")
    print(f"  建议的因子预设 (ic_optimized_v2)")
    print(f"{'=' * 85}")
    print(f"  # 基于 IC 分析的因子权重")
    print(f"  'ic_optimized_v2': {{")
    print(f"      'name': 'IC优化v2',")
    print(f"      'factors': {{")

    # 分配权重: 按 abs_ic_mean 占比
    kept_best = best_per_factor[best_per_factor["factor"].isin(kept_factors)]
    total_abs_ic = kept_best["abs_ic_mean"].sum()
    for _, r in kept_best.iterrows():
        w = r["abs_ic_mean"] / total_abs_ic
        print(f"          '{r['factor']}': {w:.3f},  # IC={r['ic_mean']:+.4f} ICIR={r['icir']:+.3f}")

    print(f"      }},")
    print(f"      'buy_threshold': 0.15,")
    print(f"      'sell_threshold': -0.10,")
    print(f"  }}")

    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()
