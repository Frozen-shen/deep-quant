"""
scripts/run_minute_ic_validation.py — 分钟频因子 IC 验证

由于分钟数据历史较短 (~40天), 本脚本做 preliminary IC check:
  - 用可用数据计算截面 IC
  - 输出 ICIR / IC均值 / 正IC比率
  - 结果保存为 p9_minute_ic.json

注意: 样本量有限 (~20个截面日), 结果仅供参考。
      随着每日数据积累, IC 估计会越来越稳定。

用法:
  py scripts/run_minute_ic_validation.py
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from minute_factors import (
    load_minute_data, compute_minute_factors_batch, get_minute_factor_names
)

log = get_logger("minute_ic")

# ── 配置 ──
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p9_minute_ic.json")
FORWARD_DAYS = 20  # 前向收益窗口 (与 P5 调仓周期一致)
MIN_CROSS_SECTION = 100  # 截面最少股票数


def load_daily_returns() -> dict:
    """加载日线数据用于计算前向收益 (使用 data_store/ 完整宇宙)。"""
    store_dir = os.path.join(BASE_DIR, "data_store")
    if not os.path.exists(store_dir):
        return {}

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
                all_data[sym] = df
        except Exception:
            pass
    return all_data


def compute_forward_returns(all_data: dict, as_of_date, horizon: int = 20) -> dict:
    """计算从 as_of_date 起 horizon 天的前向收益。"""
    as_of = pd.Timestamp(as_of_date)
    fwd_ret = {}
    for sym, df in all_data.items():
        mask = df["date"] > as_of
        future = df.loc[mask, "close"]
        if len(future) < horizon:
            continue
        # 当前价格
        curr_mask = df["date"] <= as_of
        if not curr_mask.any():
            continue
        curr_price = df.loc[curr_mask, "close"].iloc[-1]
        if curr_price <= 0:
            continue
        fwd_price = future.iloc[horizon - 1]
        fwd_ret[sym] = fwd_price / curr_price - 1
    return fwd_ret


def main():
    log.info("=" * 60)
    log.info("  分钟频因子 IC 验证 (Preliminary)")
    log.info("  前向收益窗口: %d 天", FORWARD_DAYS)
    log.info("=" * 60)

    # 1. 加载分钟数据
    minute_data = load_minute_data(use_cache=False)
    if not minute_data:
        log.error("无分钟数据! 请先运行: py scripts/fetch_minute_data.py")
        return

    log.info("分钟数据: %d 只股票", len(minute_data))

    # 确定可用日期范围
    all_dates = set()
    for sym, df in minute_data.items():
        if "day" in df.columns:
            dates = df["day"].dt.date.unique()
            all_dates.update(dates)
    trade_dates = sorted(all_dates)
    log.info("交易日范围: %s ~ %s (%d 天)", trade_dates[0], trade_dates[-1], len(trade_dates))

    # 2. 加载日线数据 (计算前向收益)
    log.info("加载日线数据...")
    daily_data = load_daily_returns()
    log.info("日线数据: %d 只", len(daily_data))

    # 3. 逐日计算 IC
    factor_names = get_minute_factor_names()
    ic_records = {name: [] for name in factor_names}

    # 采样: 每5天取一个截面 (避免重叠)
    sample_dates = trade_dates[::5]
    # 排除最后 FORWARD_DAYS 天 (没有前向收益)
    if len(trade_dates) > FORWARD_DAYS:
        cutoff_date = trade_dates[-FORWARD_DAYS]
        sample_dates = [d for d in sample_dates if d < cutoff_date]

    log.info("IC 采样日: %d 天", len(sample_dates))

    for today in sample_dates:
        # 计算分钟因子
        factors = compute_minute_factors_batch(minute_data, today, lookback=10)
        if len(factors) < MIN_CROSS_SECTION:
            continue

        # 计算前向收益
        fwd_ret = compute_forward_returns(daily_data, today, FORWARD_DAYS)
        if len(fwd_ret) < MIN_CROSS_SECTION:
            continue

        # 取交集
        common_syms = set(factors.keys()) & set(fwd_ret.keys())
        if len(common_syms) < MIN_CROSS_SECTION:
            continue

        syms = sorted(common_syms)
        ret_arr = np.array([fwd_ret[s] for s in syms])

        for name in factor_names:
            vals = np.array([factors[s].get(name, np.nan) for s in syms])
            valid = ~np.isnan(vals)
            if valid.sum() < MIN_CROSS_SECTION:
                ic_records[name].append(np.nan)
                continue
            corr, _ = spearmanr(vals[valid], ret_arr[valid])
            ic_records[name].append(corr)

    # 4. 汇总
    results = []
    log.info("")
    log.info("  因子 IC 汇总:")
    log.info("  %-25s %8s %8s %8s %8s", "Factor", "ICIR", "IC_mean", "IC_std", "Pos%")
    log.info("  " + "-" * 60)

    for name in factor_names:
        ics = [x for x in ic_records[name] if not np.isnan(x)]
        if len(ics) < 3:
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

        log.info("  %-25s %+8.4f %+8.5f %8.5f %7.1f%%",
                 name, icir, ic_mean, ic_std, pos_ratio * 100)

    # 5. 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "分钟频因子 IC 验证 (Preliminary — 样本量有限)",
            "n_stocks": len(minute_data),
            "n_sample_dates": len(sample_dates),
            "forward_days": FORWARD_DAYS,
            "data_range": f"{trade_dates[0]} ~ {trade_dates[-1]}",
        },
        "results": sorted(results, key=lambda x: -abs(x["icir"])),
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("  结果已保存: %s", OUTPUT_PATH)
    n_pass = sum(1 for r in results if abs(r["icir"]) >= 0.2)
    log.info("  通过 |ICIR| >= 0.2: %d / %d 个因子", n_pass, len(results))


if __name__ == "__main__":
    main()
