"""
run_fundamental_ic_validation.py — 基本面因子 IC 验证

对基本面因子计算 rank IC (与未来20日收益的 Spearman 相关),
输出格式与 p3_full_ic.json 一致, 可直接合并到 P5 因子选择。

基本面因子 (point-in-time, 避免前视偏差):
  fund_roe: ROE年化
  fund_profit_growth: 净利润同比增速
  fund_revenue_growth: 营收同比增速
  fund_debt_ratio: 资产负债率
  fund_net_margin: 销售净利率
  fund_ocf_ps: 每股经营现金流
  fund_pb: 市净率 (price/bvps)
  fund_profit_growth_ded: 扣非利润增速

用法:
  py scripts/run_fundamental_ic_validation.py

输出:
  data/ic_validation/p6_fundamental_ic.json
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

log = get_logger("fund_ic")

config = load_config(os.path.join(BASE_DIR, "config.yaml"))
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "p6_fundamental_ic.json")

# 研究期 (与 P3 一致)
RESEARCH_START = config["data_partition"]["research"]["start"]
RESEARCH_END = config["data_partition"]["research"]["end"]

# IC 计算参数
HORIZON = 20          # 未来20日收益
SAMPLE_INTERVAL = 20  # 每20个交易日采样一次 (月度)

FUNDAMENTAL_FACTORS = [
    "fund_roe",
    "fund_profit_growth",
    "fund_revenue_growth",
    "fund_debt_ratio",
    "fund_net_margin",
    "fund_ocf_ps",
    "fund_pb",
    "fund_profit_growth_ded",
    "fund_ep",
    "fund_bp",
    "fund_sp",
    "fund_ocf_yield",
    "fund_accruals",
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


def load_fundamental_panel() -> Dict[str, pd.DataFrame]:
    """加载基本面数据。"""
    from fundamental_fetcher import load_fundamental_panel as load_panel

    panel = load_panel()
    log.info("基本面数据: %d 只", len(panel))
    return panel


def compute_forward_returns(all_data: Dict[str, pd.DataFrame],
                            horizon: int = HORIZON) -> Dict[str, pd.Series]:
    """预计算每只股票的未来N日收益率。"""
    fwd_ret = {}
    for sym, df in all_data.items():
        close = df.set_index("date")["close"]
        ret = close.shift(-horizon) / close - 1
        fwd_ret[sym] = ret
    return fwd_ret


def get_pit_fundamentals(panel: Dict[str, pd.DataFrame],
                         as_of_date: pd.Timestamp) -> Dict[str, dict]:
    """
    Point-in-time 基本面因子: 取 as_of_date 之前最新已发布的财报。

    财报发布延迟: 报告期 + 2个月 (保守估计)
    """
    results = {}

    for sym, fund_df in panel.items():
        if "report_date" not in fund_df.columns:
            continue

        available = fund_df.copy()
        available["available_date"] = available["report_date"] + pd.DateOffset(months=2)
        mask = available["available_date"] <= as_of_date
        available = available[mask]

        if len(available) == 0:
            continue

        latest = available.iloc[-1]
        factors = {}

        # ROE 年化
        roe = latest.get("roe", np.nan)
        if pd.notna(roe):
            month = latest["report_date"].month
            if month == 3:
                roe = roe * 4
            elif month == 6:
                roe = roe * 2
            elif month == 9:
                roe = roe * 4 / 3
            factors["fund_roe"] = float(roe)

        # 利润增速
        pg = latest.get("profit_growth", np.nan)
        if pd.notna(pg):
            factors["fund_profit_growth"] = float(pg)

        # 营收增速
        rg = latest.get("revenue_growth", np.nan)
        if pd.notna(rg):
            factors["fund_revenue_growth"] = float(rg)

        # 负债率
        dr = latest.get("debt_ratio", np.nan)
        if pd.notna(dr):
            factors["fund_debt_ratio"] = float(dr)

        # 净利率
        nm = latest.get("net_margin", np.nan)
        if pd.notna(nm):
            factors["fund_net_margin"] = float(nm)

        # 每股经营现金流
        ocf = latest.get("ocf_ps", np.nan)
        if pd.notna(ocf):
            factors["fund_ocf_ps"] = float(ocf)

        # PB
        bvps = latest.get("bvps", np.nan)
        if pd.notna(bvps) and bvps > 0:
            # 需要当日收盘价
            factors["_bvps"] = float(bvps)

        # 扣非利润增速
        pgd = latest.get("profit_growth_deducted", np.nan)
        if pd.notna(pgd):
            factors["fund_profit_growth_ded"] = float(pgd)

        if factors:
            results[sym] = factors

    return results


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
    log.info("  基本面因子 IC 验证")
    log.info("  研究期: %s ~ %s", RESEARCH_START, RESEARCH_END)
    log.info("  Horizon: %d 日, 采样间隔: %d 日", HORIZON, SAMPLE_INTERVAL)
    log.info("=" * 60)

    # 加载数据
    all_data = load_price_data()
    panel = load_fundamental_panel()

    if not panel:
        log.error("无基本面数据! 请先运行: py fundamental_fetcher.py")
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
    ic_records = {name: [] for name in FUNDAMENTAL_FACTORS}

    from fundamental_fetcher import compute_fundamental_factors

    for di, today in enumerate(sample_dates):
        # Point-in-time 基本面 (含 value 因子: EP/BP/SP/OCF_Yield/Accruals)
        pit_factors = compute_fundamental_factors(panel, all_data, today)

        if len(pit_factors) < 30:
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
        for fname in FUNDAMENTAL_FACTORS:
            factor_vals = {}
            for sym, factors in pit_factors.items():
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
    log.info("\n  基本面因子 IC 统计:")
    log.info("  {'因子':<30} {'ICIR':>8} {'IC均值':>10} {'天数':>6} {'正比例':>8}")
    log.info("  " + "-" * 70)

    for fname in FUNDAMENTAL_FACTORS:
        ics = ic_records[fname]
        if len(ics) < 10:
            log.info("  %-30s 数据不足 (%d天)", fname, len(ics))
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
            "category": "fundamental",
        })

        log.info("  %-30s %+8.4f %+10.5f %6d %8.3f",
                 fname, icir, ic_mean, len(ics), pos_ratio)

    # 输出
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "基本面因子 IC 验证 (P6)",
            "research_period": f"{RESEARCH_START} ~ {RESEARCH_END}",
            "horizon": HORIZON,
            "sample_interval": SAMPLE_INTERVAL,
            "n_stocks": len(all_data),
            "n_fundamental": len(panel),
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
