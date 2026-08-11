"""
scripts/active/run_minute_ic_decay.py — 分钟因子 IC 衰减诊断 (路线B v21)

对 15 个分钟因子 (现有10 + 分布特征5) 计算 T+1/5/10/20 截面秩 IC,
输出 IC 衰减曲线 → 判定哪些因子"配得上月频调仓" (T+20 IC 显著)
vs 只有短周期 IC (T+1/5, 留给周频路线C)。

频率: 5m / 15m 双频率独立诊断 (--freq 5 或 15, 默认 15 对齐生产)。
数据: data_store/minute_<freq>m/, 2022-01 起。
样本: 每 5 个交易日取截面 (降自相关), lookback=20 均值聚合 (与生产一致)。

用法:
  py scripts/active/run_minute_ic_decay.py              # 15m
  py scripts/active/run_minute_ic_decay.py --freq 5     # 5m
  py scripts/active/run_minute_ic_decay.py --max 500    # 抽样 (快速)
输出: data/ic_validation/p10_minute_ic_decay_<freq>m.json
"""

import os
import sys
import time
import argparse
import json
import warnings

import numpy as np
import pandas as pd

# spearmanr 对含 NaN 的截面会产生 invalid value 警告 (数据已在 keep 掩码过滤,
# 但 scipy 内部仍警告); 抑制以减少日志刷屏
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from minute_factors import MINUTE_FACTOR_NAMES, compute_minute_factors

HORIZONS = [1, 5, 10, 20]      # 持有期 (交易日): 周频→1/5, 月频→20
LOOKBACK = 20                  # 因子聚合回看 (与生产一致)
SAMPLE_STEP = 5                # 截面采样步长 (降自相关)
MIN_CROSS_SECTION = 100        # 每日最少股票数
MIN_DAYS = 20                  # 因子最少有效截面数
RESEARCH_START = "2022-06-01"  # 留出 lookback+标签余量


def load_minute_data(cache_dir: str, max_stocks: int = 0) -> dict:
    """加载分钟数据 {symbol: DataFrame} (仅列 day/close/volume/amount 足够算因子)."""
    data = {}
    if not os.path.isdir(cache_dir):
        return data
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".parquet"))
    if max_stocks > 0:
        files = files[:max_stocks]
    for f in files:
        sym = f.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(cache_dir, f))
            if len(df) < 100 or "day" not in df.columns:
                continue
            df["day"] = pd.to_datetime(df["day"])
            data[sym] = df
        except Exception:
            pass
    return data


def build_factor_series(all_minute: dict, as_of_dates: list) -> dict:
    """逐截面日计算全部因子的截面值: {factor: DataFrame(日期×股票)}"""
    # 先算每只股票的逐日因子 (按 trade_date), 再按截面日取
    per_sym = {}   # {sym: DataFrame(索引=trade_date, 列=因子)}
    for sym, df in all_minute.items():
        df = df[df["day"] <= as_of_dates[-1]]
        if len(df) == 0:
            continue
        df["trade_date"] = df["day"].dt.date
        dates = sorted(df["trade_date"].unique())
        if len(dates) < LOOKBACK:
            continue
        rows = {}
        prev_close = None
        for td in dates:
            day_bars = df[df["trade_date"] == td]
            from minute_factors import _compute_single_day
            f = _compute_single_day(day_bars, prev_close)
            prev_close = float(day_bars["close"].iloc[-1]) if len(day_bars) else prev_close
            if f:
                rows[td] = f
        if len(rows) >= 5:
            per_sym[sym] = pd.DataFrame.from_dict(rows, orient="index")
    # 滚动均值聚合 (与生产 compute_minute_factors 的均值语义一致)
    series = {fn: {} for fn in MINUTE_FACTOR_NAMES}
    for sym, fdf in per_sym.items():
        roll = fdf.rolling(LOOKBACK, min_periods=5).mean()
        for td in as_of_dates:
            d = td.date()
            if d in roll.index:
                row = roll.loc[d]
                for fn in MINUTE_FACTOR_NAMES:
                    v = row.get(fn, np.nan)
                    if np.isfinite(v):
                        series[fn].setdefault(td, {})[sym] = float(v)
    return series


def main():
    parser = argparse.ArgumentParser(description="分钟因子 IC 衰减诊断")
    parser.add_argument("--freq", type=str, default="15", choices=["5", "15"])
    parser.add_argument("--max", type=int, default=0, help="抽样股票数 (0=全部)")
    args = parser.parse_args()

    cache_dir = os.path.join(BASE_DIR, "data_store", f"minute_{args.freq}m")
    print("=" * 60)
    print(f"  分钟因子 IC 衰减诊断 ({args.freq}m)")
    print(f"  因子: {len(MINUTE_FACTOR_NAMES)} 个 | 持有期: {HORIZONS}")
    print(f"  数据: {cache_dir}")
    print("=" * 60)

    t0 = time.time()
    all_minute = load_minute_data(cache_dir, args.max)
    print(f"加载分钟数据: {len(all_minute)} 只 ({time.time()-t0:.0f}s)")

    # 截面日: 全部日期中每 SAMPLE_STEP 取一个
    all_dates = []
    for df in all_minute.values():
        all_dates.extend(pd.to_datetime(df["day"]).dt.date.tolist())
    all_dates = sorted(set(all_dates))
    as_of_dates = [pd.Timestamp(d) for d in all_dates
                   if pd.Timestamp(d) >= pd.Timestamp(RESEARCH_START)][::SAMPLE_STEP]
    print(f"截面日: {len(as_of_dates)} 个 ({as_of_dates[0].date()} ~ {as_of_dates[-1].date()})")

    # 前瞻收益 (每股票取日末收盘: 同日多bar时取最后)
    close_wide = {}
    for sym, df in all_minute.items():
        s = df.set_index(pd.to_datetime(df["day"]).dt.normalize())["close"]
        s = s[~s.index.duplicated(keep="last")]
        close_wide[sym] = s
    close_wide = pd.DataFrame(close_wide).sort_index()
    ret_wide = {h: close_wide.shift(-h) / close_wide - 1 for h in HORIZONS}

    t1 = time.time()
    series = build_factor_series(all_minute, as_of_dates)
    print(f"因子序列构建: {time.time()-t1:.0f}s")

    # 逐因子逐 horizon 计算截面 IC
    from scipy.stats import spearmanr
    out = {}
    for fn in MINUTE_FACTOR_NAMES:
        fdict = series.get(fn, {})
        if len(fdict) < MIN_DAYS:
            out[fn] = {"error": f"截面不足 ({len(fdict)}天)"}
            continue
        row_ic = {}
        for h in HORIZONS:
            ics = []
            for td in as_of_dates:
                if td not in fdict:
                    continue
                fv = fdict[td]
                rv = ret_wide[h].loc[td] if td in ret_wide[h].index else None
                if rv is None:
                    continue
                syms = [s for s in fv if s in rv.index]
                if len(syms) < MIN_CROSS_SECTION:
                    continue
                fvk = np.array([fv[s] for s in syms])
                rvk = rv[syms].to_numpy()
                keep = ~(np.isnan(fvk) | np.isnan(rvk))
                if keep.sum() < MIN_CROSS_SECTION or np.std(fvk[keep]) < 1e-9:
                    continue
                ic, _ = spearmanr(fvk[keep], rvk[keep])
                if np.isfinite(ic):
                    ics.append(ic)
            if len(ics) >= MIN_DAYS:
                ics = np.array(ics)
                row_ic[h] = {
                    "ic_mean": float(ics.mean()),
                    "icir": float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0,
                    "n": int(len(ics)),
                    "pos_ratio": float((ics > 0).mean()),
                }
            else:
                row_ic[h] = {"error": f"截面不足 ({len(ics)}天)"}
        out[fn] = row_ic

    # 汇总表
    print("\n因子                 | T+1       T+5       T+10      T+20      | 判定")
    print("-" * 85)
    summary = {}
    for fn in MINUTE_FACTOR_NAMES:
        r = out[fn]
        cells = []
        for h in HORIZONS:
            v = r.get(h, {})
            if "error" in v:
                cells.append("  N/A     ")
            else:
                cells.append(f"{v['ic_mean']:+.3f}({v['n']}) ")
        # 判定: T+20 |IC|>=0.02 → 月频可用; T+5 强但 T+20 弱 → 仅周频
        t20 = r.get(20, {})
        t5 = r.get(5, {})
        t1 = r.get(1, {})
        if "error" not in t20 and abs(t20.get("ic_mean", 0)) >= 0.02:
            verdict = "月频✓"
        elif "error" not in t5 and abs(t5.get("ic_mean", 0)) >= 0.02:
            verdict = "仅短周期"
        elif "error" not in t1 and abs(t1.get("ic_mean", 0)) >= 0.02:
            verdict = "仅超短"
        else:
            verdict = "无效"
        summary[fn] = verdict
        print(f"{fn:<22} | {''.join(cells)} | {verdict}")

    # 保存
    os.makedirs(os.path.join(BASE_DIR, "data", "ic_validation"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "data", "ic_validation",
                            f"p10_minute_ic_decay_{args.freq}m.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"freq": args.freq, "horizons": HORIZONS,
                   "n_stocks": len(all_minute), "summary": summary,
                   "factors": out}, f, ensure_ascii=False, indent=1)
    print(f"\n输出: {out_path} | 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
