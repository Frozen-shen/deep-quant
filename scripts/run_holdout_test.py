"""
run_holdout_test.py — 测试集 + 模拟盘期确认回测

使用 P5 已锁定的 30 个因子 (ICIR 权重来自 research 期),
在 test 期和 blind 期做纯 out-of-sample 回测。

⚠️ 测试集只跑一次，不可回退。

用法:
  py scripts/run_holdout_test.py              # 跑 test + blind
  py scripts/run_holdout_test.py --test-only  # 仅 test 期
  py scripts/run_holdout_test.py --blind-only # 仅 blind 期

输出:
  data/ic_validation/holdout_results.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gate import load_config
from logger import get_logger

log = get_logger("holdout_test")

# ── 配置 ──
config = load_config(os.path.join(BASE_DIR, "config.yaml"))
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
P5_REPORT = os.path.join(IC_DIR, "p5_portfolio_report.json")
OUTPUT_PATH = os.path.join(IC_DIR, "holdout_results.json")

PARTITIONS = config["data_partition"]
TEST_START = PARTITIONS["test"]["start"]       # 2024-07-01
TEST_END = PARTITIONS["test"]["end"]           # 2025-06-30
BLIND_START = PARTITIONS["blind"]["start"]     # 2025-07-01
BLIND_END = PARTITIONS["blind"]["end"]         # 2026-07-31

BT_CONFIG = {
    "rebalance_days": 20,
    "top_k": config["execution"]["top_k"],
    "initial_capital": config["execution"]["initial_capital"],
    "lot_size": config["execution"]["lot_size"],
    "slippage_bps": config["execution"]["slippage_bps"],
    "commission_buy": config["execution"]["commission_buy"],
    "commission_sell": config["execution"]["commission_sell"],
}


def log_msg(msg: str):
    log.info(msg)


def load_factors() -> List[dict]:
    """从 P5 报告加载已锁定的因子。"""
    with open(P5_REPORT, "r", encoding="utf-8") as f:
        rpt = json.load(f)
    factors = rpt["selected_factors"]
    log_msg(f"加载 P5 锁定因子: {len(factors)} 个")
    return factors


def compute_composite_scores(factors: List[dict], all_data: dict,
                             factor_cache, today) -> Dict[str, float]:
    """IC加权线性组合。"""
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f.get("weight_multiplier", 1.0) for f in factors])
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


def load_benchmark(start: str, end: str):
    """加载 CSI1000 基准收益。"""
    path = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    sub = df.loc[mask, "close"]
    if len(sub) < 2:
        return 0.0, np.array([])
    total_ret = sub.iloc[-1] / sub.iloc[0] - 1
    daily_ret = sub.pct_change().dropna().values
    return total_ret, daily_ret


def run_backtest(factors: List[dict], all_data: dict, factor_cache,
                 start: str, end: str, label: str) -> dict:
    """在指定期间跑回测。"""
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker

    log_msg(f"\n{'='*60}")
    log_msg(f"  {label}: {start} ~ {end}")
    log_msg(f"  因子: {len(factors)} 个, Top-{BT_CONFIG['top_k']}")
    log_msg(f"{'='*60}")

    bt = SimpleBacktest(
        initial_capital=BT_CONFIG["initial_capital"],
        top_k=BT_CONFIG["top_k"],
        lot_size=BT_CONFIG["lot_size"],
        slippage_bps=BT_CONFIG["slippage_bps"],
        turnover_limit_pct=1.0,
    )
    rules = TradingRules()
    ranker = PortfolioRanker(
        top_k=BT_CONFIG["top_k"],
        n_drop=BT_CONFIG["top_k"],
        hold_thresh=1,
        sell_rank_buffer=0,
        buy_confirm_days=1,
        cost_threshold=0.0,
    )

    # 交易日
    rs = pd.Timestamp(start)
    re_ = pd.Timestamp(end)
    all_dates = set()
    for sym in list(all_data.keys())[:200]:
        df = all_data[sym]
        mask = (df["date"] >= rs) & (df["date"] <= re_)
        all_dates.update(df.loc[mask, "date"].tolist())
    bt_dates = sorted(all_dates)

    if len(bt_dates) < 20:
        log_msg(f"  交易日不足 ({len(bt_dates)}), 跳过")
        return {}

    log_msg(f"  交易日: {len(bt_dates)} 天")

    equity_curve = []
    daily_returns = []
    turnover_history = []
    rebalance_count = 0
    total_trades = 0
    pending_decision = None
    prev_equity = float(BT_CONFIG["initial_capital"])

    for di, today in enumerate(bt_dates):
        if pending_decision is not None:
            b, s, trades = bt.execute(pending_decision, today, all_data, rules)
            total_trades += b + s
            pending_decision = None

        if di % BT_CONFIG["rebalance_days"] == 0:
            scores = compute_composite_scores(factors, all_data, factor_cache, today)
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
                log_msg(f"    调仓 #{rebalance_count}: 持仓={len(bt.positions)}")

        close_prices = {}
        for sym in list(bt.positions.keys()):
            if sym in all_data:
                dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

        equity = bt.mark_to_market(close_prices)
        equity_curve.append(equity)
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

        if (di + 1) % 50 == 0:
            log_msg(f"    Day {di+1}/{len(bt_dates)}: equity={equity:,.0f}")

    # 统计
    equity_arr = np.array(equity_curve)
    daily_ret_arr = np.array(daily_returns)

    total_return = equity_arr[-1] / BT_CONFIG["initial_capital"] - 1
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准
    bench_total, bench_daily = load_benchmark(start, end)
    if bench_total is not None:
        bench_annual = (1 + bench_total) ** (1 / max(n_years, 0.1)) - 1
    else:
        bench_annual = 0.0
        bench_daily = np.zeros(len(daily_ret_arr))

    excess_annual = annual_return - bench_annual

    # Sharpe
    rf_daily = 0.025 / 252
    excess_daily = daily_ret_arr - rf_daily
    sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252) if np.std(excess_daily) > 0 else 0

    # IR
    if bench_daily is not None and len(bench_daily) >= len(daily_ret_arr):
        active_returns = daily_ret_arr - bench_daily[:len(daily_ret_arr)]
    elif bench_daily is not None and len(bench_daily) > 0:
        active_returns = daily_ret_arr - np.pad(bench_daily, (0, max(0, len(daily_ret_arr) - len(bench_daily))))
    else:
        active_returns = daily_ret_arr
    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0

    # MaxDD
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Calmar
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # 换手
    avg_turnover = np.mean(turnover_history) if turnover_history else 0

    result = {
        "period": label,
        "start": start,
        "end": end,
        "n_days": len(bt_dates),
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "benchmark_annual": round(bench_annual, 6),
        "excess_annual": round(excess_annual, 6),
        "sharpe": round(sharpe, 4),
        "ir": round(ir, 4),
        "max_drawdown": round(max_drawdown, 6),
        "calmar": round(calmar, 4),
        "monthly_turnover": round(avg_turnover, 4),
        "rebalance_count": rebalance_count,
        "total_trades": total_trades,
    }

    log_msg(f"\n  结果:")
    log_msg(f"    总收益: {total_return*100:+.1f}%")
    log_msg(f"    年化收益: {annual_return*100:+.1f}%")
    log_msg(f"    基准年化 (CSI1000): {bench_annual*100:+.1f}%")
    log_msg(f"    年化超额: {excess_annual*100:+.1f}%")
    log_msg(f"    Sharpe: {sharpe:.2f}")
    log_msg(f"    IR: {ir:.2f}")
    log_msg(f"    最大回撤: {max_drawdown*100:.1f}%")
    log_msg(f"    Calmar: {calmar:.2f}")
    log_msg(f"    月均换手: {avg_turnover*100:.1f}%")
    log_msg(f"    调仓: {rebalance_count} 次, {total_trades} 笔")

    return result


def main():
    parser = argparse.ArgumentParser(description="Holdout 测试集/模拟盘期回测")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--blind-only", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    log_msg("=" * 60)
    log_msg("  Holdout 确认回测")
    log_msg("  ⚠️ 测试集结果不可回退")
    log_msg("=" * 60)

    # 加载因子
    factors = load_factors()

    # 加载数据
    from data_cache import get_cached_symbols, load
    from factor_scorer import FactorScorer
    from factor_cache import FactorCache

    log_msg("加载数据...")
    syms = get_cached_symbols()
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log_msg(f"  有效: {len(all_data)} 只")

    # 预计算因子
    log_msg("预计算因子...")
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
            log_msg(f"  {min(i+batch_size, len(symbols))}/{len(symbols)} ({time.time()-t0:.0f}s)")

    # 回测
    results = {}

    if not args.blind_only:
        results["test"] = run_backtest(
            factors, all_data, factor_cache,
            TEST_START, TEST_END, "TEST (最终确认)")

    if not args.test_only:
        results["blind"] = run_backtest(
            factors, all_data, factor_cache,
            BLIND_START, BLIND_END, "BLIND (模拟盘期)")

    # 输出
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Holdout 确认回测 (P5 锁定因子)",
            "n_factors": len(factors),
            "n_stocks": len(all_data),
            "bt_config": BT_CONFIG,
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "results": results,
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log_msg(f"\n{'='*60}")
    log_msg(f"  结果已保存: {OUTPUT_PATH}")
    log_msg(f"  耗时: {time.time()-t_start:.0f}s")

    # 汇总对比
    log_msg(f"\n  {'期间':<20} {'年化':>8} {'超额':>8} {'IR':>6} {'MaxDD':>8} {'Sharpe':>8}")
    log_msg(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8}")
    for key, r in results.items():
        if r:
            log_msg(f"  {r['period']:<20} {r['annual_return']*100:>+7.1f}% "
                    f"{r['excess_annual']*100:>+7.1f}% {r['ir']:>6.2f} "
                    f"{r['max_drawdown']*100:>7.1f}% {r['sharpe']:>8.2f}")
    log_msg("=" * 60)


if __name__ == "__main__":
    main()
