"""
run_p5_portfolio_validation.py — P5 全因子组合验证 (v3 — 四项优化)

优化内容:
  A. 降换手: 使用 regime-adaptive hold_thresh/n_drop/cost_threshold (月换手<30%)
  B. 状态自适应: 牛市加动量(正IC)权重, 熊市加防御(负IC)权重
  C. 放宽阈值: |ICIR| > 0.2 (原0.3), 纳入动量因子
  D. 基本面因子: 预留接口 (需网络, 当前graceful fallback)

流程:
  1. 因子筛选 (|ICIR| > 0.2)
  2. 独立性剪枝 (|corr| > 0.6)
  3. Regime-adaptive IC加权线性组合
  4. Walk-Forward 回测 (development 分区: 2023-01 ~ 2024-06)
  5. Bootstrap 统计检验
  6. Go/No-Go 判定

基准: CSI1000 指数 (data/cache/index_csi1000.parquet)

用法:
  py scripts/run_p5_portfolio_validation.py
  py scripts/run_p5_portfolio_validation.py --skip-bt   # 仅因子选择+独立性
  py scripts/run_p5_portfolio_validation.py --no-regime  # 禁用regime自适应 (对比用)

输出:
  data/ic_validation/p5_portfolio_report.json
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gate import load_config, check, GateViolation
from logger import get_logger
from experiment_tracker import log_experiment

log = get_logger("p5_validation")

# ── 配置 ──
config = load_config(os.path.join(BASE_DIR, "config.yaml"))

# 门禁: P5 验证使用 development 分区
try:
    check(partition="development", script_name="run_p5_portfolio_validation", config=config)
except GateViolation as e:
    log.error("门禁拦截: %s", e)
    sys.exit(2)

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
IC_INPUT = os.path.join(IC_DIR, "p3_full_ic.json")
REPORT_PATH = os.path.join(IC_DIR, "p5_portfolio_report.json")

# 从 config.yaml 读取参数
CORR_THRESHOLD = config["factors"]["corr_threshold"]  # 0.6
DEV_START = config["data_partition"]["development"]["start"]
DEV_END = config["data_partition"]["development"]["end"]

# 回测参数
BT_CONFIG = {
    "start": DEV_START,
    "end": DEV_END,
    "rebalance_days": 20,
    "top_k": config["execution"]["top_k"],  # 30
    "initial_capital": config["execution"]["initial_capital"],  # 100000
    "lot_size": config["execution"]["lot_size"],  # 100
    "slippage_bps": config["execution"]["slippage_bps"],  # 30
    "commission_buy": config["execution"]["commission_buy"],
    "commission_sell": config["execution"]["commission_sell"],
}

# ═══ 优化 C: 放宽因子阈值 ═══
MIN_ABS_ICIR = 0.2  # 原 0.3, 放宽以纳入动量因子 (return_30d ICIR=+0.56 等)

# Go/No-Go 标准
GO_CRITERIA = {
    "min_excess_annual": 0.05,   # 年化超额 > 5%
    "min_ir": 0.5,               # IR > 0.5
    "max_drawdown": -0.20,       # MaxDD < 20%
    "ci_excludes_zero": True,    # Bootstrap 95% CI 不含 0
}


# ═══════════════════════════════════════════════════════════
#  Part 1: 因子选择
# ═══════════════════════════════════════════════════════════

def load_and_select_factors() -> List[dict]:
    """加载 IC 结果并筛选 (优化 C: 阈值 0.2)。"""
    if not os.path.exists(IC_INPUT):
        log.error("IC 结果文件不存在: %s", IC_INPUT)
        sys.exit(1)

    with open(IC_INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    log.info("加载 IC 结果: %d 个因子", len(results))

    # 筛选 |ICIR| > 阈值
    selected = []
    for r in results:
        if abs(r["icir"]) >= MIN_ABS_ICIR:
            selected.append({
                "name": r["factor"],
                "icir": r["icir"],
                "abs_icir": abs(r["icir"]),
                "ic_mean": r["ic_mean"],
                "n_days": r["n_days"],
                "pos_ratio": r["pos_ratio"],
                "category": "price_volume",
                "weight_multiplier": 1.0,
            })

    selected.sort(key=lambda x: -x["abs_icir"])

    # 统计正/负IC因子
    n_pos = sum(1 for f in selected if f["icir"] > 0)
    n_neg = sum(1 for f in selected if f["icir"] < 0)
    log.info("筛选通过 (|ICIR| > %.1f): %d 个因子 (正IC: %d, 负IC: %d)",
             MIN_ABS_ICIR, len(selected), n_pos, n_neg)

    for f in selected[:10]:
        log.info("  %s: ICIR=%+.4f", f["name"], f["icir"])
    if len(selected) > 10:
        log.info("  ... 共 %d 个", len(selected))

    return selected


# ═══════════════════════════════════════════════════════════
#  Part 2: 独立性剪枝
# ═══════════════════════════════════════════════════════════

def prune_correlated_factors(selected: List[dict], all_data: dict,
                             factor_cache) -> Tuple[List[dict], dict]:
    """贪心剪枝: 按|ICIR|降序, 与已选因子相关>阈值的剔除。"""
    if len(selected) <= 1:
        return selected, {}

    log.info("独立性检验 (|corr| > %.1f 剪枝)...", CORR_THRESHOLD)

    factor_names = [f["name"] for f in selected]

    # 采样日期 (development 期内每 10 天)
    rs = pd.Timestamp(BT_CONFIG["start"])
    re_ = pd.Timestamp(BT_CONFIG["end"])
    all_dates = set()
    for sym in list(all_data.keys())[:200]:
        df = all_data[sym]
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    sample_dates = sorted(all_dates)[::10]
    log.info("  采样日: %d 天", len(sample_dates))

    # 收集截面 rank
    factor_ranks = {name: {} for name in factor_names}
    for today in sample_dates:
        day_values = {name: {} for name in factor_names}
        for sym in all_data:
            feats = factor_cache.get(sym, today)
            if feats is None:
                continue
            for name in factor_names:
                val = feats.get(name, np.nan)
                if not np.isnan(val):
                    day_values[name][sym] = val

        for name in factor_names:
            vals = day_values[name]
            if len(vals) < 30:
                continue
            syms = list(vals.keys())
            arr = np.array([vals[s] for s in syms])
            ranks = rankdata(arr)
            factor_ranks[name][today] = dict(zip(syms, ranks))

    # 计算 pairwise 相关
    n = len(factor_names)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            ni, nj = factor_names[i], factor_names[j]
            common_dates = set(factor_ranks[ni].keys()) & set(factor_ranks[nj].keys())
            if len(common_dates) < 5:
                continue
            corrs = []
            for d in common_dates:
                ri = factor_ranks[ni][d]
                rj = factor_ranks[nj][d]
                common_syms = set(ri.keys()) & set(rj.keys())
                if len(common_syms) < 30:
                    continue
                syms = list(common_syms)
                a = np.array([ri[s] for s in syms])
                b = np.array([rj[s] for s in syms])
                c, _ = spearmanr(a, b)
                if not np.isnan(c):
                    corrs.append(c)
            if corrs:
                corr_matrix[i, j] = np.mean(corrs)
                corr_matrix[j, i] = corr_matrix[i, j]

    # 贪心剪枝
    kept = []
    kept_indices = []
    pruned_info = {}

    for idx, f in enumerate(selected):
        should_prune = False
        for ki in kept_indices:
            if abs(corr_matrix[idx, ki]) > CORR_THRESHOLD:
                should_prune = True
                pruned_info[f["name"]] = {
                    "pruned_by": selected[ki]["name"],
                    "correlation": float(corr_matrix[idx, ki]),
                }
                break
        if not should_prune:
            kept.append(f)
            kept_indices.append(idx)

    n_pos_kept = sum(1 for f in kept if f["icir"] > 0)
    n_neg_kept = sum(1 for f in kept if f["icir"] < 0)
    log.info("  剪枝前: %d, 剪枝后: %d (正IC: %d, 负IC: %d), 剔除: %d",
             len(selected), len(kept), n_pos_kept, n_neg_kept, len(pruned_info))
    for name, info in list(pruned_info.items())[:5]:
        log.info("    ✂ %s (与 %s 相关 %.3f)", name, info["pruned_by"], info["correlation"])

    # 构建相关性 dict
    corr_dict = {}
    for i in range(min(n, 20)):
        for j in range(i + 1, min(n, 20)):
            if abs(corr_matrix[i, j]) > 0.3:
                key = f"{factor_names[i]}_vs_{factor_names[j]}"
                corr_dict[key] = round(float(corr_matrix[i, j]), 4)

    return kept, {"correlation_matrix": corr_dict, "pruned": pruned_info}


# ═══════════════════════════════════════════════════════════
#  Part 3: Walk-Forward 回测 (含 Regime 自适应)
# ═══════════════════════════════════════════════════════════

def compute_composite_scores(factors: List[dict], all_data: dict,
                             factor_cache, today) -> Dict[str, float]:
    """IC加权线性组合 → 截面 z-score → 复合得分。"""
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f["weight_multiplier"] for f in factors])
    abs_weight_sum = np.sum(np.abs(weights))
    if abs_weight_sum < 1e-9:
        return {}

    raw_values = {name: {} for name in factor_names}
    for sym in all_data:
        feats = factor_cache.get(sym, today)
        if feats is None:
            continue
        for name in factor_names:
            val = feats.get(name, np.nan)
            if not np.isnan(val):
                raw_values[name][sym] = val

    # 所有因子都有值的股票
    valid_syms = None
    for name in factor_names:
        s = set(raw_values[name].keys())
        valid_syms = s if valid_syms is None else (valid_syms & s)

    if valid_syms is None or len(valid_syms) < BT_CONFIG["top_k"]:
        return {}

    valid_syms = sorted(valid_syms)
    n = len(valid_syms)
    composite = np.zeros(n)

    for fi, name in enumerate(factor_names):
        vals = np.array([raw_values[name][s] for s in valid_syms])
        mean, std = vals.mean(), vals.std()
        if std < 1e-9:
            continue
        z = (vals - mean) / std
        composite += weights[fi] * z

    composite /= abs_weight_sum
    return dict(zip(valid_syms, composite.tolist()))


def load_benchmark() -> pd.DataFrame:
    """加载 CSI1000 基准。"""
    path = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
    if not os.path.exists(path):
        log.warning("CSI1000 基准不存在, 使用等权全池")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def run_walkforward_backtest(factors: List[dict], all_data: dict,
                             factor_cache, use_regime: bool = True) -> dict:
    """
    Walk-Forward 回测 (development 分区)。

    优化 A: regime-adaptive 换手控制
    优化 B: regime-adaptive 因子权重
    """
    log.info("=" * 60)
    log.info("  Walk-Forward 回测 (v3 — Regime自适应)")
    log.info("  期间: %s ~ %s", BT_CONFIG["start"], BT_CONFIG["end"])
    log.info("  调仓: 每%d日, Top-%d", BT_CONFIG["rebalance_days"], BT_CONFIG["top_k"])
    log.info("  因子: %d 个, Regime自适应: %s", len(factors), use_regime)
    log.info("=" * 60)

    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker
    from regime_detector import RegimeDetector, Regime

    # ═══ 优化 B: 初始化 Regime 检测器 ═══
    bench_path = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
    regime_detector = RegimeDetector.from_benchmark_parquet(bench_path)

    bt = SimpleBacktest(
        initial_capital=BT_CONFIG["initial_capital"],
        top_k=BT_CONFIG["top_k"],
        lot_size=BT_CONFIG["lot_size"],
        slippage_bps=BT_CONFIG["slippage_bps"],
        turnover_limit_pct=1.0,
    )
    rules = TradingRules()

    # ═══ 优化 A: 初始 ranker — 回测中适度控制换手 ═══
    # 注意: 回测中 n_drop 不能太小, 否则从0建仓到top_k需要太多周期
    # 生产环境可以用更严格的参数 (因为已有持仓)
    ranker = PortfolioRanker(
        top_k=BT_CONFIG["top_k"],
        n_drop=10,              # 回测中允许较快建仓 (生产用2~5)
        hold_thresh=10,         # 回测中10天 (生产用20~30)
        sell_rank_buffer=3,     # 缓冲区: 排名跌出 top_k+3 才卖
        buy_confirm_days=1,
        cost_threshold=0.03,    # 回测中3%门槛 (生产用0.10~0.15)
    )

    # 交易日
    rs = pd.Timestamp(BT_CONFIG["start"])
    re_ = pd.Timestamp(BT_CONFIG["end"])
    all_dates = set()
    for sym in list(all_data.keys())[:200]:
        df = all_data[sym]
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    bt_dates = sorted(all_dates)

    if len(bt_dates) < 50:
        log.error("回测日期不足 (%d)", len(bt_dates))
        return {}

    log.info("  交易日: %d 天", len(bt_dates))

    equity_curve = []
    daily_returns = []
    turnover_history = []
    regime_history = []
    rebalance_count = 0
    total_trades = 0
    pending_decision = None
    prev_equity = float(BT_CONFIG["initial_capital"])

    for di, today in enumerate(bt_dates):
        # T+1 执行
        if pending_decision is not None:
            b, s, trades = bt.execute(pending_decision, today, all_data, rules)
            total_trades += b + s
            pending_decision = None

        # 调仓日
        if di % BT_CONFIG["rebalance_days"] == 0:
            # ═══ 优化 B: 检测 regime, 调整因子权重 ═══
            if use_regime:
                regime = regime_detector.detect(today)
                adapted_factors = regime_detector.adapt_factor_weights(factors, today, regime)
                # ═══ 优化 A: 调整换手参数 ═══
                tp = regime_detector.get_turnover_params(today, regime)
                ranker.hold_thresh = tp["hold_thresh"]
                ranker.n_drop = tp["n_drop"]
                ranker.cost_threshold = tp["cost_threshold"]
                ranker.sell_rank_buffer = tp["sell_rank_buffer"]
                regime_history.append({
                    "date": str(today.date() if hasattr(today, 'date') else today),
                    "regime": regime.value,
                })
            else:
                adapted_factors = factors
                regime = Regime.RANGE

            scores = compute_composite_scores(adapted_factors, all_data, factor_cache, today)
            if scores and len(scores) >= BT_CONFIG["top_k"]:
                tradeable = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][all_data[sym]["date"] <= today].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable[sym] = sc

                if len(tradeable) >= BT_CONFIG["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable, holdings)
                    decision["buy"] = [s for s in decision["buy"]
                                      if s in all_data and rules.can_buy(
                                          s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    decision["sell"] = [s for s in decision["sell"]
                                       if s in all_data and rules.can_sell(
                                           s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    pending_decision = decision
                    rebalance_count += 1
                    n_turn = (len(decision.get("sell", [])) + len(decision.get("buy", []))) / (2 * BT_CONFIG["top_k"])
                    turnover_history.append(n_turn)

            if rebalance_count > 0 and rebalance_count % 6 == 0:
                regime_str = regime.value if use_regime else "N/A"
                log.info("    调仓 #%d: %s, 持仓=%d, regime=%s",
                         rebalance_count,
                         today.date() if hasattr(today, 'date') else today,
                         len(bt.positions), regime_str)

        # Mark-to-market
        close_prices = {}
        for sym in list(bt.positions.keys()):
            if sym in all_data:
                dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

        equity = bt.mark_to_market(close_prices)
        equity_curve.append({"date": str(today.date() if hasattr(today, 'date') else today), "equity": equity})
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

        if (di + 1) % 100 == 0:
            log.info("    Day %d/%d: equity=%.0f", di + 1, len(bt_dates), equity)

    # ── 统计 ──
    equity_arr = np.array([e["equity"] for e in equity_curve])
    daily_ret_arr = np.array(daily_returns)

    total_return = equity_arr[-1] / BT_CONFIG["initial_capital"] - 1
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准
    bench_df = load_benchmark()
    if bench_df is not None:
        bench_mask = (bench_df.index >= rs) & (bench_df.index <= re_)
        bench_sub = bench_df.loc[bench_mask, "close"]
        if len(bench_sub) > 1:
            bench_total = bench_sub.iloc[-1] / bench_sub.iloc[0] - 1
            bench_annual = (1 + bench_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bench_sub.pct_change().dropna().values
        else:
            bench_annual = 0.0
            bench_daily = np.zeros(len(daily_ret_arr))
    else:
        bench_annual = 0.0
        bench_daily = np.zeros(len(daily_ret_arr))

    excess_annual = annual_return - bench_annual

    # Sharpe
    rf_daily = 0.025 / 252
    excess_daily = daily_ret_arr - rf_daily
    sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252) if np.std(excess_daily) > 0 else 0

    # IR
    if len(bench_daily) >= len(daily_ret_arr):
        active_returns = daily_ret_arr - bench_daily[:len(daily_ret_arr)]
    else:
        active_returns = daily_ret_arr - np.pad(bench_daily, (0, len(daily_ret_arr) - len(bench_daily)))
    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0

    # MaxDD
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Calmar
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # 换手
    avg_turnover = np.mean(turnover_history) if turnover_history else 0
    one_way_cost = (BT_CONFIG["slippage_bps"] / 10000 + BT_CONFIG["commission_buy"] + BT_CONFIG["commission_sell"]) / 2
    cost_drag = avg_turnover * one_way_cost * 12 * 2

    # Regime 分布统计
    regime_counts = {}
    for rh in regime_history:
        r = rh["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1

    log.info("  回测结果:")
    log.info("    总收益: %+.1f%%", total_return * 100)
    log.info("    年化收益: %+.1f%%", annual_return * 100)
    log.info("    基准年化 (CSI1000): %+.1f%%", bench_annual * 100)
    log.info("    年化超额: %+.1f%%", excess_annual * 100)
    log.info("    Sharpe: %.2f", sharpe)
    log.info("    IR: %.2f", ir)
    log.info("    最大回撤: %.1f%%", max_drawdown * 100)
    log.info("    Calmar: %.2f", calmar)
    log.info("    月均换手: %.1f%%", avg_turnover * 100)
    log.info("    年化成本: %.2f%%", cost_drag * 100)
    if regime_counts:
        log.info("    Regime分布: %s", regime_counts)

    return {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "benchmark_annual": round(bench_annual, 6),
        "excess_annual": round(excess_annual, 6),
        "sharpe": round(sharpe, 4),
        "ir": round(ir, 4),
        "max_drawdown": round(max_drawdown, 6),
        "calmar": round(calmar, 4),
        "monthly_turnover": round(avg_turnover, 4),
        "cost_drag": round(cost_drag, 6),
        "rebalance_count": rebalance_count,
        "total_trades": total_trades,
        "n_days": len(bt_dates),
        "equity_curve": equity_curve[::5],
        "regime_history": regime_history,
        "_active_returns": active_returns.tolist(),
    }


# ═══════════════════════════════════════════════════════════
#  Part 4: Bootstrap 验证
# ═══════════════════════════════════════════════════════════

def bootstrap_validation(bt_result: dict) -> dict:
    """Bootstrap 重采样验证 IR 显著性。"""
    active = np.array(bt_result.get("_active_returns", []))
    if len(active) < 50:
        return {"n_samples": len(active), "error": "insufficient data"}

    n_bootstrap = 1000
    ir_samples = []
    block_size = 20  # 月度 block bootstrap

    rng = np.random.default_rng(42)
    n_blocks = len(active) // block_size

    for _ in range(n_bootstrap):
        # Block bootstrap
        blocks = [active[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        sample = np.concatenate([blocks[i] for i in chosen])
        ir_b = np.mean(sample) / np.std(sample) * np.sqrt(252) if np.std(sample) > 0 else 0
        ir_samples.append(ir_b)

    ir_arr = np.array(ir_samples)
    ci_lower = float(np.percentile(ir_arr, 2.5))
    ci_upper = float(np.percentile(ir_arr, 97.5))

    result = {
        "n_samples": len(active),
        "n_bootstrap": n_bootstrap,
        "ir_mean": round(float(np.mean(ir_arr)), 4),
        "ir_ci_lower": round(ci_lower, 4),
        "ir_ci_upper": round(ci_upper, 4),
        "ci_excludes_zero": ci_lower > 0 or ci_upper < 0,
    }

    log.info("  Bootstrap: IR=%.3f, 95%% CI=[%.3f, %.3f], 排除0: %s",
             result["ir_mean"], ci_lower, ci_upper, result["ci_excludes_zero"])
    return result


# ═══════════════════════════════════════════════════════════
#  Part 5: Go/No-Go 判定
# ═══════════════════════════════════════════════════════════

def make_verdict(bt_result: dict, bootstrap: dict) -> Tuple[str, List[str]]:
    """判定 Go/No-Go。"""
    reasons = []

    excess = bt_result.get("excess_annual", 0)
    if excess <= GO_CRITERIA["min_excess_annual"]:
        reasons.append(f"FAIL: 年化超额 {excess*100:.1f}% <= {GO_CRITERIA['min_excess_annual']*100:.0f}%")
    else:
        reasons.append(f"PASS: 年化超额 {excess*100:.1f}%")

    ir = bt_result.get("ir", 0)
    if ir <= GO_CRITERIA["min_ir"]:
        reasons.append(f"FAIL: IR {ir:.2f} <= {GO_CRITERIA['min_ir']}")
    else:
        reasons.append(f"PASS: IR {ir:.2f}")

    mdd = bt_result.get("max_drawdown", -1)
    if mdd < GO_CRITERIA["max_drawdown"]:
        reasons.append(f"FAIL: MaxDD {mdd*100:.1f}% < {GO_CRITERIA['max_drawdown']*100:.0f}%")
    else:
        reasons.append(f"PASS: MaxDD {mdd*100:.1f}%")

    ci_ok = bootstrap.get("ci_excludes_zero", False)
    if not ci_ok:
        ci_l = bootstrap.get("ir_ci_lower", 0)
        ci_u = bootstrap.get("ir_ci_upper", 0)
        reasons.append(f"FAIL: Bootstrap CI [{ci_l:.3f}, {ci_u:.3f}] 包含0")
    else:
        reasons.append("PASS: Bootstrap CI 排除0")

    n_pass = sum(1 for r in reasons if r.startswith("PASS"))
    verdict = "GO" if n_pass == len(reasons) else "CONDITIONAL" if n_pass >= 2 else "FAIL"

    return verdict, reasons


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P5 全因子组合验证 (v3)")
    parser.add_argument("--skip-bt", action="store_true", help="跳过回测")
    parser.add_argument("--no-regime", action="store_true", help="禁用regime自适应 (对比用)")
    args = parser.parse_args()

    use_regime = not args.no_regime
    t_start = time.time()
    log.info("=" * 60)
    log.info("  P5 全因子组合验证 (v3 — 四项优化)")
    log.info("  Development 期: %s ~ %s", DEV_START, DEV_END)
    log.info("  CORR_THRESHOLD: %.1f", CORR_THRESHOLD)
    log.info("  MIN_ABS_ICIR: %.1f (优化C)", MIN_ABS_ICIR)
    log.info("  Regime自适应: %s (优化A+B)", use_regime)
    log.info("=" * 60)

    # Part 1: 因子选择
    selected = load_and_select_factors()
    if not selected:
        log.error("无因子通过筛选, 终止")
        sys.exit(1)

    # 加载数据 + 预计算因子
    from data_cache import get_cached_symbols, load
    from factor_scorer import FactorScorer
    from factor_cache import FactorCache

    log.info("加载数据...")
    syms = get_cached_symbols()
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))

    log.info("预计算因子...")
    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    factor_cache = FactorCache(scorer, factor_names)

    symbols = sorted(all_data.keys())
    batch_size = 200
    t0 = time.time()
    for i in range(0, len(symbols), batch_size):
        batch = {s: all_data[s] for s in symbols[i:i+batch_size]}
        factor_cache.precompute(batch)
        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(symbols):
            log.info("  %d/%d (%.0fs)", min(i+batch_size, len(symbols)), len(symbols), time.time()-t0)

    # Part 2: 独立性剪枝
    kept, corr_info = prune_correlated_factors(selected, all_data, factor_cache)

    if args.skip_bt:
        log.info("--skip-bt: 跳过回测")
        bt_result = {}
        bootstrap = {}
        verdict, reasons = "SKIP", ["回测已跳过"]
    else:
        # Part 3: 回测
        bt_result = run_walkforward_backtest(kept, all_data, factor_cache, use_regime=use_regime)

        # Part 4: Bootstrap
        if bt_result:
            bootstrap = bootstrap_validation(bt_result)
        else:
            bootstrap = {}

        # Part 5: 判定
        if bt_result:
            verdict, reasons = make_verdict(bt_result, bootstrap)
        else:
            verdict, reasons = "FAIL", ["回测无结果"]

    # ── 输出报告 ──
    report = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "P5 全因子组合验证报告 (v3 — 四项优化)",
            "optimizations": {
                "A_turnover_control": "regime-adaptive hold_thresh/n_drop/cost_threshold",
                "B_regime_weights": "牛市加动量×1.5, 熊市加防御×1.3",
                "C_threshold": f"MIN_ABS_ICIR={MIN_ABS_ICIR} (原0.3)",
                "D_fundamental": "预留接口 (需网络)",
            },
            "bt_config": BT_CONFIG,
            "corr_threshold": CORR_THRESHOLD,
            "min_abs_icir": MIN_ABS_ICIR,
            "use_regime": use_regime,
            "data_partition": "development",
            "n_stocks": len(all_data),
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "selected_factors": [
            {"name": f["name"], "icir": f["icir"], "category": f["category"],
             "weight_multiplier": f["weight_multiplier"], "n_days": f["n_days"]}
            for f in kept
        ],
        "correlation_matrix": corr_info.get("correlation_matrix", {}),
        "pruned_factors": corr_info.get("pruned", {}),
        "backtest": bt_result,
        "bootstrap": bootstrap,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }

    # 移除内部字段
    if "_active_returns" in report.get("backtest", {}):
        del report["backtest"]["_active_returns"]

    os.makedirs(IC_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("  VERDICT: %s", verdict)
    for r in reasons:
        log.info("    %s", r)
    log.info("  报告: %s", REPORT_PATH)
    log.info("  耗时: %.0fs", time.time() - t_start)
    log.info("=" * 60)

    # 记录实验
    try:
        log_experiment(
            script_name="run_p5_portfolio_validation",
            partition="development",
            config={"corr_threshold": CORR_THRESHOLD, "n_factors": len(kept),
                    "min_icir": MIN_ABS_ICIR, "use_regime": use_regime},
            results={"verdict": verdict, "ir": bt_result.get("ir", 0),
                     "excess": bt_result.get("excess_annual", 0),
                     "turnover": bt_result.get("monthly_turnover", 0)},
            notes="P5 v3: A(turnover)+B(regime)+C(threshold 0.2)",
            experiments_dir=os.path.join(BASE_DIR, "experiments"),
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
