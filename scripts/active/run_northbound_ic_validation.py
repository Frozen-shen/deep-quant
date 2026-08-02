"""
run_northbound_ic_validation.py — 北向资金因子 IC 验证

对北向持仓因子计算 rank IC (与未来20日收益的 Spearman 相关),
输出格式与 p3_full_ic.json 一致, 可直接合并到 P5 因子选择。

因子列表:
  nb_holding_pct  - 北向持股占A股百分比
  nb_change_5d    - 5日持股变化
  nb_change_20d   - 20日持股变化
  nb_momentum     - 北向加速度 (5d变化 / 20d变化)
  nb_new_high     - 持股是否为60日新高

用法:
  py scripts/run_northbound_ic_validation.py

输出:
  data/ic_validation/p8_northbound_ic.json
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from gate import load_config
from logger import get_logger

log = get_logger("northbound_ic")

config = load_config(os.path.join(BASE_DIR, "config.yaml"))
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p8_northbound_ic.json")

# 研究期 (与 P3 一致)
RESEARCH_START = config["data_partition"]["research"]["start"]
RESEARCH_END = config["data_partition"]["research"]["end"]

# IC 计算参数
HORIZON = 20
SAMPLE_INTERVAL = 20

NORTHBOUND_FACTORS = [
    "nb_holding_pct",
    "nb_change_5d",
    "nb_change_20d",
    "nb_momentum",
    "nb_new_high",
]


def load_price_data() -> Dict[str, pd.DataFrame]:
    """加载行情数据。"""
    from data_cache import get_cached_symbols, load

    syms = get_cached_symbols()
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df

    log.info("行情数据: %d 只", len(all_data))
    return all_data


def load_northbound_data() -> Dict[str, pd.DataFrame]:
    """加载北向持仓数据。"""
    from smart_money_fetcher import load_smart_money_data
    data = load_smart_money_data()
    log.info("北向数据: %d 只", len(data))
    return data


def compute_forward_returns(all_data: Dict[str, pd.DataFrame],
                            horizon: int = HORIZON) -> Dict[str, pd.Series]:
    """预计算每只股票的未来N日收益率。"""
    fwd_ret = {}
    for sym, df in all_data.items():
        close = df.set_index("date")["close"]
        ret = close.shift(-horizon) / close - 1
        fwd_ret[sym] = ret
    return fwd_ret


def compute_nb_factors_for_date(nb_data: Dict[str, pd.DataFrame],
                                as_of_date: pd.Timestamp) -> Dict[str, dict]:
    """对指定日期计算所有股票的北向因子 (point-in-time)。"""
    from smart_money_fetcher import compute_northbound_factors
    return compute_northbound_factors(nb_data, as_of_date)


def compute_rank_ic(factor_values: Dict[str, float],
                    forward_returns: Dict[str, float]) -> float:
    """计算截面 rank IC (Spearman 相关)。"""
    common_syms = set(factor_values.keys()) & set(forward_returns.keys())
    if len(common_syms) < 30:
        return np.nan

    syms = sorted(common_syms)
    fvals = np.array([factor_values[s] for s in syms])
    frets = np.array([forward_returns[s] for s in syms])

    valid = ~(np.isnan(fvals) | np.isnan(frets))
    if valid.sum() < 30:
        return np.nan

    corr, _ = spearmanr(fvals[valid], frets[valid])
    return corr


def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("  北向资金因子 IC 验证 (P8)")
    log.info("  研究期: %s ~ %s", RESEARCH_START, RESEARCH_END)
    log.info("  Horizon: %d 日, 采样间隔: %d 日", HORIZON, SAMPLE_INTERVAL)
    log.info("  因子: %s", ", ".join(NORTHBOUND_FACTORS))
    log.info("=" * 60)

    # 加载数据
    all_data = load_price_data()
    nb_data = load_northbound_data()

    if not nb_data:
        log.error("无北向数据! 请先运行: py scripts/fetch_smart_money.py")
        sys.exit(1)

    # 预计算未来收益
    log.info("预计算未来 %d 日收益...", HORIZON)
    fwd_ret = compute_forward_returns(all_data, HORIZON)

    # 收集采样日期
    rs = pd.Timestamp(RESEARCH_START)
    re_ = pd.Timestamp(RESEARCH_END)
    all_dates = set()
    for sym in list(all_data.keys())[:200]:
        df = all_data[sym]
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    sample_dates = sorted(all_dates)[::SAMPLE_INTERVAL]
    log.info("采样日: %d 天", len(sample_dates))

    # 逐日计算 IC
    ic_records = {name: [] for name in NORTHBOUND_FACTORS}

    for di, today in enumerate(sample_dates):
        # Point-in-time 北向因子
        nb_factors = compute_nb_factors_for_date(nb_data, today)

        if len(nb_factors) < 30:
            continue

        # 当日未来收益
        today_fwd = {}
        for sym, ret_series in fwd_ret.items():
            if today in ret_series.index:
                val = ret_series.loc[today]
                if pd.notna(val):
                    today_fwd[sym] = float(val)

        if len(today_fwd) < 30:
            continue

        # 对每个因子计算截面 IC
        for fname in NORTHBOUND_FACTORS:
            factor_vals = {}
            for sym, factors in nb_factors.items():
                if fname in factors:
                    val = factors[fname]
                    if val is not None and not np.isnan(val):
                        factor_vals[sym] = val

            if len(factor_vals) >= 30:
                ic = compute_rank_ic(factor_vals, today_fwd)
                if not np.isnan(ic):
                    ic_records[fname].append(ic)

        if (di + 1) % 20 == 0:
            log.info("  进度: %d/%d", di + 1, len(sample_dates))

    # 汇总统计
    results = []
    log.info("\n  北向资金因子 IC 统计:")
    log.info("  %-25s %8s %10s %6s %8s", "因子", "ICIR", "IC均值", "天数", "正比例")
    log.info("  " + "-" * 65)

    for fname in NORTHBOUND_FACTORS:
        ics = ic_records[fname]
        if len(ics) < 10:
            log.info("  %-25s 数据不足 (%d天)", fname, len(ics))
            continue

        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        pos_ratio = float(np.mean(ic_arr > 0))

        results.append({
            "factor": fname,
            "icir": round(icir, 4),
            "ic_mean": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "n_days": len(ics),
            "pos_ratio": round(pos_ratio, 3),
            "abs_ic_mean": round(abs(ic_mean), 5),
            "horizon": HORIZON,
            "category": "northbound",
        })

        log.info("  %-25s %+8.4f %+10.5f %6d %8.3f",
                 fname, icir, ic_mean, len(ics), pos_ratio)

    # 输出
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "北向资金因子 IC 验证 (P8)",
            "research_period": f"{RESEARCH_START} ~ {RESEARCH_END}",
            "horizon": HORIZON,
            "sample_interval": SAMPLE_INTERVAL,
            "n_stocks": len(all_data),
            "n_northbound": len(nb_data),
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "results": results,
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("\n  输出: %s", OUTPUT_PATH)
    log.info("  耗时: %.0fs", time.time() - t_start)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
