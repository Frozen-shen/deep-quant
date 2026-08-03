"""
scripts/active/run_minute_ic_validation_v2.py — 分钟频因子 IC 验证 (优化版)

优化: 预计算所有日期的因子值 (单次遍历), 然后向量化计算 IC。
避免对每个采样日重复过滤 17K 行 DataFrame。

用法:
  py scripts/active/run_minute_ic_validation_v2.py
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from minute_factors import (
    MINUTE_CACHE_DIR, MINUTE_FACTOR_NAMES, _compute_single_day, _gini_coefficient
)

log = get_logger("minute_ic_v2")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p9_minute_ic.json")
FORWARD_DAYS = 20
MIN_CROSS_SECTION = 100
LOOKBACK = 20  # 因子聚合回看天数


def load_daily_data() -> dict:
    """加载日线数据 (data_store/)。"""
    store_dir = os.path.join(BASE_DIR, "data_store")
    all_data = {}
    files = [f for f in os.listdir(store_dir)
             if f.endswith(".parquet") and not f.startswith("index_")
             and "minute" not in f]
    for fname in files:
        sym = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(store_dir, fname))
            if len(df) > 0 and "date" in df.columns and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                all_data[sym] = df.set_index("date")["close"]
        except Exception:
            pass
    return all_data


def precompute_daily_factors(minute_dir: str) -> pd.DataFrame:
    """
    单次遍历所有分钟数据, 预计算每只股票每天的因子值。
    
    优化: groupby 替代逐日过滤, rolling 替代手动回看聚合。
    返回: DataFrame, index=(symbol, trade_date), columns=factor_names
    """
    files = [f for f in os.listdir(minute_dir) if f.endswith(".parquet")]
    log.info("  预计算日频因子: %d 只股票...", len(files))

    all_frames = []
    for i, fname in enumerate(files):
        sym = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(minute_dir, fname))
            if len(df) < 16:
                continue

            df["day"] = pd.to_datetime(df["day"])
            df["trade_date"] = df["day"].dt.date

            # groupby 替代逐日过滤 — O(1) 访问每日数据
            prev_close = None
            daily_records = []
            for td, group in df.groupby("trade_date", sort=True):
                day_bars = group.reset_index(drop=True)
                factors = _compute_single_day(day_bars, prev_close)
                if factors:
                    daily_records.append({"date": pd.Timestamp(td), **factors})
                if len(day_bars) > 0:
                    prev_close = float(day_bars["close"].iloc[-1])

            if len(daily_records) < LOOKBACK:
                continue

            # rolling 均值替代手动回看循环
            stock_df = pd.DataFrame(daily_records).set_index("date").sort_index()
            agg_df = stock_df.rolling(LOOKBACK, min_periods=5).mean()
            agg_df = agg_df.dropna(how="all")
            agg_df["symbol"] = sym
            all_frames.append(agg_df)

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            log.info("    %d/%d (%d frames)", i + 1, len(files), len(all_frames))

    log.info("  合并 %d 个 DataFrame...", len(all_frames))
    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames)
    result = result.reset_index().set_index(["symbol", "date"]).sort_index()
    log.info("  完成: %d 条记录, %d 只股票",
             len(result), result.index.get_level_values("symbol").nunique())
    return result


def compute_forward_returns(daily_data: dict, dates: list) -> pd.DataFrame:
    """
    计算前向收益矩阵。
    
    返回: DataFrame, index=date, columns=symbol, values=forward_return
    """
    # 收集所有交易日
    all_trade_dates = set()
    for sym, close_series in daily_data.items():
        all_trade_dates.update(close_series.index.tolist())
    all_trade_dates = sorted(all_trade_dates)
    date_to_idx = {d: i for i, d in enumerate(all_trade_dates)}

    fwd_records = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        if d_ts not in date_to_idx:
            continue
        idx = date_to_idx[d_ts]
        if idx + FORWARD_DAYS >= len(all_trade_dates):
            continue

        future_date = all_trade_dates[idx + FORWARD_DAYS]
        row = {"date": d_ts}
        count = 0
        for sym, close_series in daily_data.items():
            if d_ts in close_series.index and future_date in close_series.index:
                curr = close_series[d_ts]
                fut = close_series[future_date]
                if curr > 0:
                    row[sym] = fut / curr - 1
                    count += 1
        if count >= MIN_CROSS_SECTION:
            fwd_records.append(row)

    if not fwd_records:
        return pd.DataFrame()

    fwd_df = pd.DataFrame(fwd_records).set_index("date")
    return fwd_df


def main():
    log.info("=" * 60)
    log.info("  分钟频因子 IC 验证 v2 (Baostock 15min, 2022-2026)")
    log.info("  前向收益窗口: %d 天, 因子回看: %d 天", FORWARD_DAYS, LOOKBACK)
    log.info("=" * 60)

    # 1. 预计算所有日频因子 (单次遍历)
    log.info("Step 1: 预计算日频因子...")
    factors_df = precompute_daily_factors(MINUTE_CACHE_DIR)
    if factors_df.empty:
        log.error("无因子数据!")
        return

    all_dates = sorted(factors_df.index.get_level_values("date").unique())
    log.info("  因子日期范围: %s ~ %s (%d 天)",
             all_dates[0].date(), all_dates[-1].date(), len(all_dates))

    # 2. 加载日线数据
    log.info("Step 2: 加载日线数据...")
    daily_data = load_daily_data()
    log.info("  日线: %d 只", len(daily_data))

    # 3. 采样日期 (每5天)
    sample_dates = all_dates[::5]
    # 排除最后 FORWARD_DAYS 天
    if len(all_dates) > FORWARD_DAYS:
        cutoff = all_dates[-FORWARD_DAYS]
        sample_dates = [d for d in sample_dates if d < cutoff]
    log.info("  IC 采样日: %d 天", len(sample_dates))

    # 4. 计算前向收益
    log.info("Step 3: 计算前向收益...")
    fwd_df = compute_forward_returns(daily_data, sample_dates)
    if fwd_df.empty:
        log.error("无前向收益数据!")
        return
    log.info("  有效截面: %d 天, 平均 %d 只/天",
             len(fwd_df), int(fwd_df.notna().sum(axis=1).mean()))

    # 5. 逐截面计算 Rank IC
    log.info("Step 4: 计算 Rank IC...")
    factor_names = MINUTE_FACTOR_NAMES
    ic_records = {name: [] for name in factor_names}

    # 预计算可用日期集合 (避免每次迭代重建 Index)
    available_dates = set(factors_df.index.get_level_values("date"))
    n_computed = 0

    for date in fwd_df.index:
        if date not in available_dates:
            continue

        # 取当日因子截面
        try:
            day_factors = factors_df.xs(date, level="date")
        except KeyError:
            continue

        # 取当日收益
        day_returns = fwd_df.loc[date].dropna()

        # 交集
        common_syms = day_factors.index.intersection(day_returns.index)
        if len(common_syms) < MIN_CROSS_SECTION:
            continue

        ret_arr = day_returns[common_syms].values
        for name in factor_names:
            vals = day_factors.loc[common_syms, name].values
            valid = ~np.isnan(vals)
            if valid.sum() < MIN_CROSS_SECTION:
                ic_records[name].append(np.nan)
                continue
            corr, _ = spearmanr(vals[valid], ret_arr[valid])
            ic_records[name].append(corr)

        n_computed += 1
        if n_computed % 50 == 0:
            log.info("    IC 进度: %d/%d 截面", n_computed, len(fwd_df))

    # 6. 汇总
    results = []
    log.info("")
    log.info("  因子 IC 汇总 (218天采样, 2022-2026):")
    log.info("  %-25s %8s %8s %8s %8s %6s", "Factor", "ICIR", "IC_mean", "IC_std", "Pos%", "N")
    log.info("  " + "-" * 70)

    for name in factor_names:
        ics = [x for x in ic_records[name] if not np.isnan(x)]
        if len(ics) < 10:
            log.info("  %-25s %8s", name, "N/A (样本不足)")
            continue

        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        pos_ratio = float(np.mean(ic_arr > 0))

        results.append({
            "factor": name,
            "icir": round(icir, 4),
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "n_days": len(ics),
            "pos_ratio": round(pos_ratio, 4),
        })

        log.info("  %-25s %+8.4f %+8.5f %8.5f %7.1f%% %5d",
                 name, icir, ic_mean, ic_std, pos_ratio * 100, len(ics))

    # 7. 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "分钟频因子 IC 验证 (Baostock 15min, 全量历史)",
            "n_stocks": int(factors_df.index.get_level_values("symbol").nunique()),
            "n_sample_dates": len(sample_dates),
            "forward_days": FORWARD_DAYS,
            "lookback_days": LOOKBACK,
            "data_range": f"{all_dates[0].date()} ~ {all_dates[-1].date()}",
            "source": "baostock_15min",
        },
        "results": sorted(results, key=lambda x: -abs(x["icir"])),
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("  结果已保存: %s", OUTPUT_PATH)
    n_pass = sum(1 for r in results if abs(r["icir"]) >= 0.3)
    n_marginal = sum(1 for r in results if 0.2 <= abs(r["icir"]) < 0.3)
    log.info("  强信号 (|ICIR| >= 0.3): %d / %d", n_pass, len(results))
    log.info("  边际信号 (0.2 <= |ICIR| < 0.3): %d / %d", n_marginal, len(results))


if __name__ == "__main__":
    main()
