"""
分析师修正 + 事件信号因子 IC 验证 — Spearman Rank IC + 独立性检验

重要说明 (务必阅读):
  ★ 分析师数据历史极短 (盈利预测接口仅返回当前快照, 需逐日积累),
    revision_30d / coverage_change 需 >=2 个快照才可计算。
  ★ 覆盖窄: 仅被充分覆盖的个股 (研报数>=3) 有分析师因子。
  ★ lockup / lhb 为稀有事件, 多数股票因子值为0。
  因此本验证的"有效天数"通常很少, 统计功效 (statistical power) 弱,
  结果仅供参考。因子最佳使用方式是作为"实时叠加层", 而非独立 alpha。

验证内容:
  1. 7个因子 (4分析师 + 3事件) 在 5/10/20 日窗口的 Spearman Rank IC
  2. 覆盖率统计 (研报数>=3 的股票数 / 各因子有效覆盖)
  3. 与价量因子 (turnover_vol, return_30d) 的秩相关 — 独立性

输出: data/ic_validation/p4_analyst_event_ic.json

用法:
  python scripts/validate_analyst_factors.py
  python scripts/validate_analyst_factors.py --horizon 5 10 20
  python scripts/validate_analyst_factors.py --sample-interval 5
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

from factors.analyst_revision import AnalystRevision
from factors.analyst_revision import FACTOR_NAMES as ANALYST_FACTORS
from factors.event_signals import EventSignals
from factors.event_signals import FACTOR_NAMES as EVENT_FACTORS

ALL_FACTORS = ANALYST_FACTORS + EVENT_FACTORS

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_price_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """加载价量数据 (优先 data_store, 回退 data_cache)。"""
    data = {}
    ds = os.path.join(BASE_DIR, "data_store")
    dc = os.path.join(BASE_DIR, "data_cache")
    for sym in symbols:
        df = None
        for base in (ds, dc):
            path = os.path.join(base, f"{sym}.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    break
                except Exception:
                    df = None
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            data[sym] = df
    return data


def compute_forward_returns(price_data: Dict[str, pd.DataFrame],
                            horizons: List[int]) -> Dict[str, Dict[int, pd.Series]]:
    """{symbol: {horizon: Series(index=date, value=fwd_return)}}"""
    fwd = {}
    for sym, df in price_data.items():
        fwd[sym] = {}
        for h in horizons:
            r = df["close"].shift(-h) / df["close"] - 1
            r.index = df["date"]
            fwd[sym][h] = r
    return fwd


def compute_price_volume_factors(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """价量因子 (独立性检验): turnover_vol, return_30d。"""
    pv = {}
    for sym, df in price_data.items():
        if "turnover" not in df.columns or len(df) < 30:
            continue
        turnover = df["turnover"].astype(float)
        close = df["close"].astype(float)
        tv = turnover.rolling(20).std() / (turnover.rolling(20).mean() + 0.01)
        r30 = close / close.shift(30) - 1
        pv_df = pd.DataFrame({"date": df["date"], "turnover_vol": tv.values, "return_30d": r30.values})
        pv[sym] = pv_df.set_index("date")
    return pv


def get_universe_symbols() -> List[str]:
    """股票池: data_store 全量。"""
    ds = os.path.join(BASE_DIR, "data_store")
    if not os.path.exists(ds):
        return []
    return sorted([f.replace(".parquet", "") for f in os.listdir(ds)
                   if f.endswith(".parquet") and f[0].isdigit()])


def run_validation(sample_interval: int = 5, horizons: List[int] = None) -> dict:
    if horizons is None:
        horizons = [5, 10, 20]

    print("=" * 60, flush=True)
    print("  分析师 + 事件因子 IC 验证", flush=True)
    print("=" * 60, flush=True)

    ar = AnalystRevision()
    es = EventSignals()

    # ── 1. 数据可用性 ──
    a_stats = ar.get_data_stats()
    e_stats = es.get_data_stats()
    print("\n[数据覆盖]", flush=True)
    print(f"  分析师快照数: {a_stats.get('n_snapshots', 0)} "
          f"({a_stats.get('snapshot_dates', 'N/A')})", flush=True)
    print(f"  覆盖>=3研报: {a_stats.get('stocks_covered_ge3', 0)} 只 "
          f"(占最新快照 {a_stats.get('coverage_pct', 0)*100:.1f}%)", flush=True)
    print(f"  解禁事件: {e_stats.get('lockup_events', 0)} 条 "
          f"({e_stats.get('lockup_stocks', 0)} 只)", flush=True)
    print(f"  龙虎榜事件: {e_stats.get('lhb_events', 0)} 条 "
          f"({e_stats.get('lhb_stocks', 0)} 只)", flush=True)
    print(f"  业绩预告: {e_stats.get('preview_events', 0)} 条", flush=True)

    no_analyst = a_stats.get("n_snapshots", 0) == 0
    no_events = (e_stats.get("lockup_events", 0) == 0 and
                 e_stats.get("lhb_events", 0) == 0 and
                 e_stats.get("preview_events", 0) == 0)
    if no_analyst and no_events:
        print("\n[ERROR] 无任何分析师/事件数据。请先运行:", flush=True)
        print("  python scripts/fetch_analyst_data.py", flush=True)
        print("  python scripts/fetch_events.py --source all", flush=True)
        return {"error": "no_data"}

    # ── 2. 股票池 + 价量 ──
    symbols = get_universe_symbols()
    print(f"\n[加载] 股票池: {len(symbols)}", flush=True)
    price_data = load_price_data(symbols)
    print(f"[加载] 有价量数据: {len(price_data)}", flush=True)
    if len(price_data) < 30:
        print("[ERROR] 价量数据不足30只", flush=True)
        return {"error": "insufficient_price_data"}

    fwd_rets = compute_forward_returns(price_data, horizons)

    # ── 3. 采样日期 ──
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    max_h = max(horizons)
    cutoff = all_dates[-max_h] if len(all_dates) > max_h else all_dates[-1]
    sample_dates = all_dates[::sample_interval]
    sample_dates = [d for d in sample_dates if d < cutoff]
    print(f"[验证] 采样日: {len(sample_dates)} 天, 窗口 {horizons}", flush=True)

    # ── 4. 逐日 IC ──
    ic_results = {f: {h: [] for h in horizons} for f in ALL_FACTORS}
    daily_counts = {f: [] for f in ALL_FACTORS}

    for di, today in enumerate(sample_dates):
        today_str = str(today.date())
        # 合并分析师 + 事件因子
        a_fac = ar.compute_factors(as_of_date=today_str, symbols=list(price_data.keys()))
        e_fac = es.compute_factors(as_of_date=today_str, symbols=list(price_data.keys()))

        # 构建当日截面 {symbol: {factor: val}}
        cross = {}
        for sym in price_data:
            row = {}
            if sym in a_fac:
                row.update(a_fac[sym])
            if sym in e_fac:
                row.update(e_fac[sym])
            if row:
                cross[sym] = row

        if not cross:
            continue

        for h in horizons:
            rets = {}
            fvals = {f: {} for f in ALL_FACTORS}
            for sym, row in cross.items():
                if sym not in fwd_rets:
                    continue
                s = fwd_rets[sym].get(h)
                if s is None or today not in s.index:
                    continue
                rv = s.loc[today]
                if pd.isna(rv):
                    continue
                rets[sym] = rv
                for f in ALL_FACTORS:
                    fvals[f][sym] = row.get(f, np.nan)

            if len(rets) < 10:
                continue

            ret_arr = np.array([rets[s] for s in rets], dtype=float)
            for f in ALL_FACTORS:
                fv = np.array([fvals[f].get(s, np.nan) for s in rets], dtype=float)
                mask = ~np.isnan(fv)
                if mask.sum() < 10 or np.std(fv[mask]) < 1e-9:
                    continue
                try:
                    ic, _ = spearmanr(fv[mask], ret_arr[mask])
                    if not np.isnan(ic):
                        ic_results[f][h].append(float(ic))
                        daily_counts[f].append(int(mask.sum()))
                except Exception:
                    pass

        if (di + 1) % 20 == 0:
            print(f"  进度: {di+1}/{len(sample_dates)}", flush=True)

    # ── 5. 独立性检验 (最新截面) ──
    print("\n[独立性] 与价量因子秩相关...", flush=True)
    pv = compute_price_volume_factors(price_data)
    independence = {}
    if sample_dates and pv:
        last_date = sample_dates[-1]
        last_str = str(last_date.date())
        a_fac = ar.compute_factors(as_of_date=last_str, symbols=list(price_data.keys()))
        e_fac = es.compute_factors(as_of_date=last_str, symbols=list(price_data.keys()))
        cross = {}
        for sym in price_data:
            row = {}
            if sym in a_fac:
                row.update(a_fac[sym])
            if sym in e_fac:
                row.update(e_fac[sym])
            if row:
                cross[sym] = row

        for f in ALL_FACTORS:
            for pv_f in ["turnover_vol", "return_30d"]:
                xs, ys = [], []
                for sym, row in cross.items():
                    fv = row.get(f, np.nan)
                    if fv is None or (isinstance(fv, float) and np.isnan(fv)):
                        continue
                    if sym not in pv or last_date not in pv[sym].index:
                        continue
                    pvv = pv[sym].loc[last_date, pv_f]
                    if pd.isna(pvv):
                        continue
                    xs.append(float(fv))
                    ys.append(float(pvv))
                if len(xs) >= 20:
                    corr, pval = spearmanr(xs, ys)
                    independence[f"{f}_vs_{pv_f}"] = {
                        "spearman_corr": round(float(corr), 4),
                        "p_value": round(float(pval), 6),
                        "n_samples": len(xs),
                    }

    # ── 6. 汇总 ──
    summary = {
        "metadata": {
            "description": "分析师修正 + 事件信号因子 IC 验证 (P4)",
            "limitation": (
                "分析师数据历史极短(快照需逐日积累)、覆盖窄(研报数>=3), "
                "lockup/lhb为稀有事件。有效天数少→统计功效弱, 结果仅供参考。"
            ),
            "recommended_use": "作为实时叠加层 (overlay), 权重设低 (5-10%)",
            "sample_interval": sample_interval,
            "horizons": horizons,
            "n_analyst_snapshots": a_stats.get("n_snapshots", 0),
            "n_stocks_covered_ge3": a_stats.get("stocks_covered_ge3", 0),
            "coverage_pct": a_stats.get("coverage_pct", 0),
            "n_lockup_events": e_stats.get("lockup_events", 0),
            "n_lhb_events": e_stats.get("lhb_events", 0),
            "n_preview_events": e_stats.get("preview_events", 0),
            "n_sample_days": len(sample_dates),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ic_by_factor": {},
        "independence": independence,
    }

    print(f"\n{'='*60}", flush=True)
    print("  结果汇总", flush=True)
    print(f"{'='*60}", flush=True)

    for f in ALL_FACTORS:
        fsum = {}
        print(f"\n  {f}:", flush=True)
        for h in horizons:
            ics = ic_results[f][h]
            if len(ics) < 3:
                fsum[f"h{h}"] = {"status": "insufficient_data", "n_days": len(ics)}
                print(f"    h={h:2d}: 数据不足 ({len(ics)}天)", flush=True)
                continue
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            icir = mean_ic / std_ic if std_ic > 0 else 0
            pos = sum(1 for x in ics if x > 0) / len(ics)
            fsum[f"h{h}"] = {
                "ic_mean": round(mean_ic, 6), "ic_std": round(std_ic, 6),
                "icir": round(icir, 4), "pos_ratio": round(pos, 4), "n_days": len(ics),
            }
            sgn = "+" if mean_ic > 0 else ""
            print(f"    h={h:2d}: IC={sgn}{mean_ic:.4f} ICIR={sgn}{icir:.2f} "
                  f"pos={pos:.0%} n={len(ics)}", flush=True)
        summary["ic_by_factor"][f] = fsum

    if independence:
        print(f"\n  [独立性] 与价量因子秩相关 (|corr|<0.3 视为独立):", flush=True)
        for key, val in independence.items():
            flag = "OK" if abs(val["spearman_corr"]) < 0.3 else "HIGH"
            print(f"    {key}: corr={val['spearman_corr']:+.3f} [{flag}]", flush=True)

    out_path = os.path.join(OUTPUT_DIR, "p4_analyst_event_ic.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(f"\n  输出: {out_path}", flush=True)
    print(f"\n  [NOTE] 分析师/事件因子历史短、覆盖窄, 建议:", flush=True)
    print(f"    - 权重设低 (5-10%), 作为叠加层", flush=True)
    print(f"    - 持续逐日积累快照, 3个月后重验 revision 类因子", flush=True)
    print(f"    - 与长历史价量/基本面因子组合使用", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析师+事件因子IC验证")
    parser.add_argument("--horizon", nargs="*", type=int, default=[5, 10, 20],
                        help="预测窗口 (交易日)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="截面采样间隔 (天)")
    args = parser.parse_args()

    run_validation(sample_interval=args.sample_interval, horizons=args.horizon)
