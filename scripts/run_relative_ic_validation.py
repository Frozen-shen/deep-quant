"""
run_relative_ic_validation.py — 市场相对因子 IC 验证

对市场相对因子计算 rank IC (与未来20日收益的 Spearman 相关),
输出格式与 p3_full_ic.json 一致, 可直接合并到 P5 因子选择。

因子列表:
  rel_mom_20d   - 20日相对动量 (个股收益 - 指数收益)
  rel_mom_60d   - 60日相对动量
  true_beta     - CAPM Beta (OLS回归斜率)
  idio_vol      - 特质波动率 (残差标准差)
  rel_strength  - 相对强度比
  max_dd_60d    - 60日最大回撤
  downside_vol  - 下行波动率
  sortino_20d   - Sortino比率

用法:
  py scripts/run_relative_ic_validation.py

输出:
  data/ic_validation/p7_relative_ic.json
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gate import load_config
from logger import get_logger

log = get_logger("relative_ic")

config = load_config(os.path.join(BASE_DIR, "config.yaml"))
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p7_relative_ic.json")

# 研究期 (与 P3 一致)
RESEARCH_START = config["data_partition"]["research"]["start"]
RESEARCH_END = config["data_partition"]["research"]["end"]

# IC 计算参数
HORIZON = 20          # 未来20日收益
SAMPLE_INTERVAL = 20  # 每20个交易日采样一次 (月度)

RELATIVE_FACTORS = [
    "rel_mom_20d",
    "rel_mom_60d",
    "true_beta",
    "idio_vol",
    "rel_strength",
    "max_dd_60d",
    "downside_vol",
    "sortino_20d",
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


def compute_forward_returns(all_data: Dict[str, pd.DataFrame],
                            horizon: int = HORIZON) -> Dict[str, pd.Series]:
    """预计算每只股票的未来N日收益率。"""
    fwd_ret = {}
    for sym, df in all_data.items():
        close = df.set_index("date")["close"]
        ret = close.shift(-horizon) / close - 1
        fwd_ret[sym] = ret
    return fwd_ret


def compute_rank_ic(factor_values: Dict[str, float],
                    forward_returns: Dict[str, float]) -> float:
    """计算截面 rank IC (Spearman 相关)。"""
    common_syms = set(factor_values.keys()) & set(forward_returns.keys())
    if len(common_syms) < 30:
        return np.nan

    syms = sorted(common_syms)
    fvals = np.array([factor_values[s] for s in syms])
    frets = np.array([forward_returns[s] for s in syms])

    # 过滤 NaN
    valid = ~(np.isnan(fvals) | np.isnan(frets))
    if valid.sum() < 30:
        return np.nan

    corr, _ = spearmanr(fvals[valid], frets[valid])
    return corr


def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("  市场相对因子 IC 验证 (P7)")
    log.info("  研究期: %s ~ %s", RESEARCH_START, RESEARCH_END)
    log.info("  Horizon: %d 日, 采样间隔: %d 日", HORIZON, SAMPLE_INTERVAL)
    log.info("  因子: %s", ", ".join(RELATIVE_FACTORS))
    log.info("=" * 60)

    # 加载数据
    all_data = load_price_data()

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
    from relative_factors import compute_relative_factors_batch

    ic_records = {name: [] for name in RELATIVE_FACTORS}

    for di, today in enumerate(sample_dates):
        # 批量计算相对因子
        batch_factors = compute_relative_factors_batch(all_data, today)

        if len(batch_factors) < 30:
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
        for fname in RELATIVE_FACTORS:
            factor_vals = {}
            for sym, factors in batch_factors.items():
                if fname in factors:
                    val = factors[fname]
                    if not np.isnan(val):
                        factor_vals[sym] = val

            if len(factor_vals) >= 30:
                ic = compute_rank_ic(factor_vals, today_fwd)
                if not np.isnan(ic):
                    ic_records[fname].append(ic)

        if (di + 1) % 20 == 0:
            log.info("  进度: %d/%d", di + 1, len(sample_dates))

    # 汇总统计
    results = []
    log.info("\n  市场相对因子 IC 统计:")
    log.info("  %-25s %8s %10s %6s %8s", "因子", "ICIR", "IC均值", "天数", "正比例")
    log.info("  " + "-" * 65)

    for fname in RELATIVE_FACTORS:
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
            "category": "relative",
        })

        log.info("  %-25s %+8.4f %+10.5f %6d %8.3f",
                 fname, icir, ic_mean, len(ics), pos_ratio)

    # 输出
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "市场相对因子 IC 验证 (P7)",
            "research_period": f"{RESEARCH_START} ~ {RESEARCH_END}",
            "horizon": HORIZON,
            "sample_interval": SAMPLE_INTERVAL,
            "n_stocks": len(all_data),
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
