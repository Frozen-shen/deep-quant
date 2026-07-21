"""
因子 IC/ICIR 分析 — 参考 Qlib SigAnaRecord

Spearman Rank IC: 因子值与未来N日收益的截面相关性
ICIR = IC均值 / IC标准差

告诉哪些因子有预测力，替代"手调权重"

用法:
  python factor_analysis.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy import stats
from data_fetcher import DataFetcher
from factor_scorer import FactorScorer

# A股10只测试集
SYMBOLS = ["688981","002371","603986","002049","300033","002230",
           "300750","002594","600519","600036"]
START, END = "2024-01-01", "2026-07-10"
FORWARD_RETURNS = [1, 3, 5, 10, 20]  # 未来N日收益


def compute_factor_panel(symbols: list, start: str, end: str) -> dict:
    """
    计算所有股票的因子面板数据。

    返回: {factor_name: DataFrame(date × symbol)}
    """
    fetcher = DataFetcher()
    scorer = FactorScorer.from_preset("trend_momentum")

    all_data = {}
    for sym in symbols:
        try:
            df = fetcher.fetch(sym, start, end, "qfq", market="a")
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except Exception as e:
            print(f"  {sym} ❌ {e}")

    # 收集所有交易日的所有股票因子
    factor_panels = {}  # factor_name → list of (date, symbol, value)
    returns_panel = []  # (date, symbol, return)

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    sample = list(all_data.values())[0]
    trading_days = sample[(sample["date"]>=start_dt)&(sample["date"]<=end_dt)]["date"].tolist()

    for today in trading_days:
        stock_data = {}
        for sym in symbols:
            if sym not in all_data: continue
            df_today = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today

        if len(stock_data) < 5: continue

        try:
            factors_raw = {}
            for sym, df in stock_data.items():
                factors = scorer.compute_factors(df)
                if len(factors) > 0:
                    factors_raw[sym] = factors.iloc[-1:]

            for col in factors_raw[list(factors_raw.keys())[0]].columns:
                if col in ("date",): continue
                if col not in factor_panels:
                    factor_panels[col] = []
                for sym in factors_raw:
                    val = factors_raw[sym][col].iloc[0]
                    if not np.isnan(val):
                        factor_panels[col].append({"date": today, "symbol": sym, "value": val})
        except: pass

    # 转 DataFrame
    result = {}
    for name in factor_panels:
        if len(factor_panels[name]) < 100: continue
        df_panel = pd.DataFrame(factor_panels[name])
        df_panel["date"] = pd.to_datetime(df_panel["date"])
        result[name] = df_panel.pivot(index="date", columns="symbol", values="value")
    return result


def compute_forward_returns(symbols: list, start: str, end: str,
                            horizons: list) -> dict:
    """计算未来N日收益面板。"""
    fetcher = DataFetcher()
    all_returns = {h: [] for h in horizons}

    for sym in symbols:
        try:
            df = fetcher.fetch(sym, start, end, "qfq", market="a")
            df["date"] = pd.to_datetime(df["date"])
            for h in horizons:
                df[f"fwd_{h}d"] = df["close"].shift(-h) / df["close"] - 1
            for _, row in df.iterrows():
                for h in horizons:
                    val = row.get(f"fwd_{h}d")
                    if not pd.isna(val):
                        all_returns[h].append({"date": row["date"], "symbol": sym, "value": val})
        except: pass

    result = {}
    for h, data in all_returns.items():
        df_r = pd.DataFrame(data)
        df_r["date"] = pd.to_datetime(df_r["date"])
        result[h] = df_r.pivot(index="date", columns="symbol", values="value")
    return result


def compute_ic(factor_panel: pd.DataFrame, return_panel: pd.DataFrame) -> dict:
    """
    计算 Spearman Rank IC。

    factor_panel: date × symbol
    return_panel: date × symbol
    """
    ic_values = []
    common_dates = factor_panel.index.intersection(return_panel.index)

    for d in common_dates:
        f_row = factor_panel.loc[d].dropna()
        r_row = return_panel.loc[d].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 5: continue

        ic, _ = stats.spearmanr(f_row[common], r_row[common])
        if not np.isnan(ic):
            ic_values.append(ic)

    if not ic_values:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "n": 0}

    ic_arr = np.array(ic_values)
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_pos_ratio": (ic_arr > 0).mean(),
        "n": len(ic_arr),
    }


def main():
    print("=" * 60)
    print("  因子 IC/ICIR 分析")
    print("  股票: 10只A股 | 区间: 2024-2026")
    print("=" * 60)

    print("\n[1/3] 计算因子面板...")
    factor_panels = compute_factor_panel(SYMBOLS, START, END)
    print(f"  有效因子: {len(factor_panels)} 个")

    print("\n[2/3] 计算未来收益...")
    fwd_returns = compute_forward_returns(SYMBOLS, START, END, FORWARD_RETURNS)
    print(f"  收益率面板: {len(fwd_returns)} 个 horizon")

    print("\n[3/3] 计算 IC...")
    results = []
    for fname, fpanel in factor_panels.items():
        for h, rpanel in fwd_returns.items():
            ic = compute_ic(fpanel, rpanel)
            if ic["n"] > 50:
                results.append({
                    "factor": fname,
                    "horizon": f"{h}d",
                    "ic_mean": ic["ic_mean"],
                    "ic_std": ic["ic_std"],
                    "icir": ic["icir"],
                    "ic_pos_ratio": ic["ic_pos_ratio"],
                    "abs_ic_mean": abs(ic["ic_mean"]),
                })

    df = pd.DataFrame(results)
    # 按 abs_ic_mean 排序
    df = df.sort_values("abs_ic_mean", ascending=False)

    print(f"\n{'='*80}")
    print(f"  IC 分析结果 (按 |IC| 降序)")
    print(f"{'='*80}")
    print(f"{'因子':<25} {'horizon':<8} {'IC均值':>8} {'ICIR':>8} {'IC>0%':>8} {'N':>6}")
    print("-" * 60)
    for _, r in df.head(30).iterrows():
        mark = "✅" if abs(r["icir"]) > 0.5 else ("⚠️" if abs(r["icir"]) > 0.3 else "❌")
        print(f"{r['factor']:<25} {r['horizon']:<8} {r['ic_mean']:>+8.4f} {r['icir']:>+8.3f} {r['ic_pos_ratio']*100:>7.0f}%")

    # 分组汇总: 每个因子的最佳 horizon
    print(f"\n{'='*80}")
    print(f"  因子最佳 horizon (按 ICIR)")
    print(f"{'='*80}")
    best = df.loc[df.groupby("factor")["abs_ic_mean"].idxmax()].sort_values("abs_ic_mean", ascending=False)
    for _, r in best.iterrows():
        mark = "✅" if abs(r["icir"]) > 0.5 else ("⚠️" if abs(r["icir"]) > 0.3 else "❌")
        print(f"  {mark} {r['factor']:<25} best_h={r['horizon']} IC={r['ic_mean']:+.4f} ICIR={r['icir']:+.3f}")


if __name__ == "__main__":
    main()
