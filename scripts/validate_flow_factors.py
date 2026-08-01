"""
资金流因子 IC 验证 — Spearman Rank IC + 独立性检验

重要说明:
  资金流数据仅有 1-2 年历史, 无法做严格的 walk-forward 验证。
  本脚本的结果仅供参考, 因子最佳使用方式是作为"实时叠加层",
  而非独立 alpha 来源。

验证内容:
  1. 各因子在 5/10/20 日预测窗口的 Spearman Rank IC
  2. 与价量因子 (turnover_vol, return_30d) 的秩相关 — 独立性
  3. 数据覆盖率和有效验证天数

输出: data/ic_validation/p2_flow_ic.json

用法:
  python scripts/validate_flow_factors.py
  python scripts/validate_flow_factors.py --horizon 5 10 20
  python scripts/validate_flow_factors.py --sample-interval 3  # 每3天取一个截面
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from factors.money_flow import MoneyFlowFactor, FACTOR_NAMES, FUND_FLOW_CACHE, NORTH_FLOW_CACHE

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_price_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """加载价量数据 (用于计算前瞻收益和价量因子)。"""
    data = {}
    data_cache_dir = os.path.join(BASE_DIR, "data_cache")
    data_store_dir = os.path.join(BASE_DIR, "data_store")

    for sym in symbols:
        df = None
        # 优先 data_cache
        path = os.path.join(data_cache_dir, f"{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            # 回退 data_store
            path = os.path.join(data_store_dir, f"{sym}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)

        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            data[sym] = df

    return data


def compute_forward_returns(price_data: Dict[str, pd.DataFrame],
                            horizons: List[int]) -> Dict[str, Dict[int, pd.Series]]:
    """
    计算前瞻收益: {symbol: {horizon: Series(index=date, value=fwd_return)}}
    """
    fwd_rets = {}
    for sym, df in price_data.items():
        fwd_rets[sym] = {}
        for h in horizons:
            # fwd_return[t] = close[t+h] / close[t] - 1
            fwd = df["close"].shift(-h) / df["close"] - 1
            fwd.index = df["date"]
            fwd_rets[sym][h] = fwd
    return fwd_rets


def compute_price_volume_factors(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    计算价量因子 (用于独立性检验):
      - turnover_vol: Std(turnover, 20) / Mean(turnover, 20)
      - return_30d: close[t] / close[t-30] - 1
    """
    pv_factors = {}
    for sym, df in price_data.items():
        if "turnover" not in df.columns or len(df) < 30:
            continue

        turnover = df["turnover"].astype(float)
        close = df["close"].astype(float)

        tv = turnover.rolling(20).std() / (turnover.rolling(20).mean() + 0.01)
        r30 = close / close.shift(30) - 1

        pv_df = pd.DataFrame({
            "date": df["date"],
            "turnover_vol": tv.values,
            "return_30d": r30.values,
        })
        pv_df = pv_df.set_index("date")
        pv_factors[sym] = pv_df

    return pv_factors


def run_ic_analysis(sample_interval: int = 5,
                    horizons: List[int] = None) -> dict:
    """
    主验证流程。

    Returns:
      验证结果字典
    """
    if horizons is None:
        horizons = [5, 10, 20]

    print("=" * 60, flush=True)
    print("  资金流因子 IC 验证", flush=True)
    print("=" * 60, flush=True)

    # ── 1. 检查数据可用性 ──
    mf = MoneyFlowFactor()
    stats = mf.get_data_stats()
    print(f"\n[数据覆盖]", flush=True)
    print(f"  资金流股票: {stats['fund_flow_stocks']}", flush=True)
    print(f"  北向持仓股票: {stats['north_flow_stocks']}", flush=True)
    print(f"  数据天数: {stats['fund_flow_sample_days']}", flush=True)

    if stats["fund_flow_stocks"] == 0:
        print("\n[ERROR] 无资金流数据, 请先运行 scripts/fetch_fund_flow.py", flush=True)
        return {"error": "no_data"}

    # ── 2. 获取有资金流数据的股票 ──
    flow_symbols = sorted([
        f.replace(".parquet", "") for f in os.listdir(FUND_FLOW_CACHE)
        if f.endswith(".parquet")
    ])
    print(f"\n[加载] 资金流股票: {len(flow_symbols)}", flush=True)

    # ── 3. 加载价量数据 ──
    price_data = load_price_data(flow_symbols)
    print(f"[加载] 有价量数据: {len(price_data)}", flush=True)

    if len(price_data) < 10:
        print("[ERROR] 价量数据不足, 无法验证", flush=True)
        return {"error": "insufficient_price_data"}

    # ── 4. 计算前瞻收益 ──
    fwd_rets = compute_forward_returns(price_data, horizons)

    # ── 5. 收集验证日期 ──
    # 取所有股票共同覆盖的日期范围
    all_dates = set()
    for sym, df in price_data.items():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    # 资金流数据日期范围
    flow_dates = set()
    for sym in flow_symbols[:50]:  # 抽样
        path = os.path.join(FUND_FLOW_CACHE, f"{sym}.parquet")
        try:
            fdf = pd.read_parquet(path, columns=["date"])
            flow_dates.update(pd.to_datetime(fdf["date"]).tolist())
        except Exception:
            pass

    # 取交集: 既有价量又有资金流的日期
    flow_dates_set = set(flow_dates)
    valid_dates = sorted([d for d in all_dates if d in flow_dates_set])

    # 按 sample_interval 采样
    sample_dates = valid_dates[::sample_interval]

    # 去掉最后 max(horizons) 天 (没有前瞻收益)
    max_h = max(horizons)
    if len(all_dates) > max_h:
        cutoff = all_dates[-max_h]
        sample_dates = [d for d in sample_dates if d < cutoff]

    print(f"[验证] 有效日期: {len(valid_dates)}, 采样: {len(sample_dates)} 天", flush=True)
    print(f"[验证] 预测窗口: {horizons}", flush=True)

    if len(sample_dates) < 10:
        print(f"\n[WARNING] 有效验证天数过少 ({len(sample_dates)}), 结果不可靠!", flush=True)
        print(f"  资金流数据历史短, 这是预期限制。", flush=True)

    # ── 6. 逐日计算 IC ──
    # {factor: {horizon: [ic_values]}}
    ic_results = {f: {h: [] for h in horizons} for f in FACTOR_NAMES}
    daily_counts = []

    for di, today in enumerate(sample_dates):
        # 计算当日资金流因子
        day_factors = mf.compute_factors(as_of_date=str(today.date()), symbols=flow_symbols)

        if len(day_factors) < 10:
            continue

        # 收集当日有效股票的前瞻收益
        for h in horizons:
            rets = {}
            factor_vals = {f: {} for f in FACTOR_NAMES}

            for sym, fvals in day_factors.items():
                if sym not in fwd_rets:
                    continue
                fwd_series = fwd_rets[sym].get(h)
                if fwd_series is None:
                    continue
                if today not in fwd_series.index:
                    continue
                ret_val = fwd_series.loc[today]
                if pd.isna(ret_val):
                    continue

                rets[sym] = ret_val
                for f in FACTOR_NAMES:
                    factor_vals[f][sym] = fvals.get(f, np.nan)

            if len(rets) < 10:
                continue

            daily_counts.append(len(rets))

            # 对每个因子计算 Spearman IC
            ret_arr = np.array([rets[s] for s in rets])
            for f in FACTOR_NAMES:
                fv = np.array([factor_vals[f].get(s, np.nan) for s in rets])
                valid_mask = ~np.isnan(fv)
                if valid_mask.sum() < 10:
                    continue
                if np.std(fv[valid_mask]) < 1e-9:
                    continue
                try:
                    ic, _ = spearmanr(fv[valid_mask], ret_arr[valid_mask])
                    if not np.isnan(ic):
                        ic_results[f][h].append(ic)
                except Exception:
                    pass

        if (di + 1) % 20 == 0:
            print(f"  进度: {di+1}/{len(sample_dates)}", flush=True)

    # ── 7. 独立性检验: 与价量因子的秩相关 ──
    print(f"\n[独立性] 计算与价量因子的秩相关...", flush=True)
    pv_factors = compute_price_volume_factors(price_data)

    # 取最新一天的截面
    independence = {}
    if sample_dates and pv_factors:
        last_date = sample_dates[-1]
        day_factors = mf.compute_factors(as_of_date=str(last_date.date()), symbols=flow_symbols)

        for flow_f in FACTOR_NAMES:
            for pv_f in ["turnover_vol", "return_30d"]:
                flow_vals = []
                pv_vals = []
                for sym, fvals in day_factors.items():
                    fv = fvals.get(flow_f, np.nan)
                    if np.isnan(fv):
                        continue
                    if sym not in pv_factors:
                        continue
                    pv_df = pv_factors[sym]
                    if last_date not in pv_df.index:
                        continue
                    pv_val = pv_df.loc[last_date, pv_f]
                    if pd.isna(pv_val):
                        continue
                    flow_vals.append(fv)
                    pv_vals.append(pv_val)

                if len(flow_vals) >= 20:
                    corr, pval = spearmanr(flow_vals, pv_vals)
                    independence[f"{flow_f}_vs_{pv_f}"] = {
                        "spearman_corr": round(float(corr), 4),
                        "p_value": round(float(pval), 6),
                        "n_samples": len(flow_vals),
                    }

    # ── 8. 汇总输出 ──
    summary = {
        "metadata": {
            "description": "资金流因子 IC 验证 (P2 行为信号)",
            "limitation": "资金流数据仅1-2年历史, 验证天数有限, 结果仅供参考",
            "recommended_use": "作为实时叠加层 (overlay), 而非独立alpha",
            "sample_interval": sample_interval,
            "horizons": horizons,
            "n_stocks_with_flow": stats["fund_flow_stocks"],
            "n_stocks_with_north": stats["north_flow_stocks"],
            "n_validation_days": len(sample_dates),
            "avg_stocks_per_day": int(np.mean(daily_counts)) if daily_counts else 0,
        },
        "ic_by_factor": {},
        "independence": independence,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"  结果汇总", flush=True)
    print(f"{'='*60}", flush=True)

    for f in FACTOR_NAMES:
        factor_summary = {}
        print(f"\n  {f}:", flush=True)

        for h in horizons:
            ics = ic_results[f][h]
            if len(ics) < 5:
                factor_summary[f"h{h}"] = {"status": "insufficient_data", "n_days": len(ics)}
                print(f"    h={h:2d}: 数据不足 ({len(ics)}天)", flush=True)
                continue

            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            icir = mean_ic / std_ic if std_ic > 0 else 0
            pos_ratio = sum(1 for x in ics if x > 0) / len(ics)

            factor_summary[f"h{h}"] = {
                "ic_mean": round(mean_ic, 6),
                "ic_std": round(std_ic, 6),
                "icir": round(icir, 4),
                "pos_ratio": round(pos_ratio, 4),
                "n_days": len(ics),
            }

            sign = "+" if mean_ic > 0 else ""
            print(f"    h={h:2d}: IC={sign}{mean_ic:.4f}  ICIR={sign}{icir:.2f}  "
                  f"pos={pos_ratio:.0%}  n={len(ics)}", flush=True)

        summary["ic_by_factor"][f] = factor_summary

    # 独立性
    if independence:
        print(f"\n  [独立性] 与价量因子秩相关 (|corr|<0.3 视为独立):", flush=True)
        for key, val in independence.items():
            flag = "OK" if abs(val["spearman_corr"]) < 0.3 else "HIGH"
            print(f"    {key}: corr={val['spearman_corr']:+.3f}  [{flag}]", flush=True)

    # ── 9. 保存 ──
    out_path = os.path.join(OUTPUT_DIR, "p2_flow_ic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  输出: {out_path}", flush=True)
    print(f"\n  [NOTE] 资金流因子历史短, 建议:", flush=True)
    print(f"    - 权重设低 (5-10%), 作为叠加层", flush=True)
    print(f"    - 与长历史价量因子组合使用", flush=True)
    print(f"    - 持续积累数据, 3个月后再重新验证", flush=True)

    return summary


# ════════════════════════════════════════
#  主入口
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="资金流因子IC验证")
    parser.add_argument("--horizon", nargs="*", type=int, default=[5, 10, 20],
                        help="预测窗口 (交易日)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="截面采样间隔 (天)")
    args = parser.parse_args()

    run_ic_analysis(
        sample_interval=args.sample_interval,
        horizons=args.horizon,
    )
