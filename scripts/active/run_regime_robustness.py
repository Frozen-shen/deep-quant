"""
scripts/active/run_regime_robustness.py — Regime 参数敏感性测试

目的: 检验 regime_detector.py 中硬编码的因子权重乘数和换手参数是否过拟合。
方法: 对 Development 期 (2023-01 ~ 2024-06) 跑多组参数, 比较结果差异。

参数组:
  - Original:     (up_pos=2.0, up_neg=0.3, down_pos=0.5, down_neg=1.5)
  - Perturbed+20%: (2.4, 0.36, 0.6, 1.8)
  - Perturbed-20%: (1.6, 0.24, 0.4, 1.2)
  - Conservative:  (1.3, 0.7, 0.7, 1.3)
  - Aggressive:    (3.0, 0.1, 0.3, 2.0)

简化: 不用 PIT universe, 不用 DateRangeGuard, 不加载分钟数据。
仅测试 regime 参数敏感性, 不是正式回测。

用法:
  py scripts/active/run_regime_robustness.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from regime_detector import RegimeDetector, Regime

log = get_logger("regime_robust")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
REPORT_PATH = os.path.join(IC_DIR, "p5_portfolio_report.json")
BENCH_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
OUTPUT_PATH = os.path.join(IC_DIR, "regime_robustness.json")

# 与 run_corrected_backtest.py 一致
FDR_ELIMINATED = {"fund_ocf_ps", "fund_sp"}
EXPLORATORY_CATEGORIES = {"minute"}

DEV_START = "2023-01-01"
DEV_END = "2024-06-30"

# ── 参数组定义 ──
# (up_pos, up_neg, down_pos, down_neg, turnover_scale)
PARAM_SETS = {
    "Original":      {"up_pos": 2.0, "up_neg": 0.3, "down_pos": 0.5, "down_neg": 1.5, "turn_scale": 1.0},
    "Perturbed+20%": {"up_pos": 2.4, "up_neg": 0.36, "down_pos": 0.6, "down_neg": 1.8, "turn_scale": 1.2},
    "Perturbed-20%": {"up_pos": 1.6, "up_neg": 0.24, "down_pos": 0.4, "down_neg": 1.2, "turn_scale": 0.8},
    "Conservative":  {"up_pos": 1.3, "up_neg": 0.7, "down_pos": 0.7, "down_neg": 1.3, "turn_scale": 1.0},
    "Aggressive":    {"up_pos": 3.0, "up_neg": 0.1, "down_pos": 0.3, "down_neg": 2.0, "turn_scale": 1.0},
}

# 基准换手参数 (与 regime_detector.get_turnover_params 一致)
BASE_TURNOVER = {
    Regime.TREND_UP:   {"hold_thresh": 8,  "n_drop": 8, "cost_threshold": 0.02, "sell_rank_buffer": 2},
    Regime.TREND_DOWN: {"hold_thresh": 12, "n_drop": 5, "cost_threshold": 0.04, "sell_rank_buffer": 3},
    Regime.RANGE:      {"hold_thresh": 10, "n_drop": 6, "cost_threshold": 0.03, "sell_rank_buffer": 2},
}


def load_factors() -> list:
    """加载因子 (与 run_corrected_backtest.py 一致: FDR + 探索性降级)。"""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    factors = report.get("selected_factors", [])
    factors = [f for f in factors if f["name"] not in FDR_ELIMINATED]
    factors = [f for f in factors if f.get("category") not in EXPLORATORY_CATEGORIES]
    return factors


def adapt_weights_custom(factors: list, regime: Regime, params: dict) -> list:
    """自定义因子权重调整 (替代 RegimeDetector.adapt_factor_weights)。"""
    adapted = []
    for f in factors:
        new_f = dict(f)
        icir = f["icir"]
        base_mult = f.get("weight_multiplier", 1.0)

        if regime == Regime.TREND_UP:
            if icir > 0:
                new_f["weight_multiplier"] = base_mult * params["up_pos"]
            else:
                new_f["weight_multiplier"] = base_mult * params["up_neg"]
        elif regime == Regime.TREND_DOWN:
            if icir > 0:
                new_f["weight_multiplier"] = base_mult * params["down_pos"]
            else:
                new_f["weight_multiplier"] = base_mult * params["down_neg"]
        # RANGE: 不调整

        adapted.append(new_f)
    return adapted


def get_turnover_params_custom(regime: Regime, turn_scale: float) -> dict:
    """自定义换手参数 (缩放基准参数)。"""
    base = BASE_TURNOVER[regime]
    return {
        "hold_thresh": max(3, round(base["hold_thresh"] * turn_scale)),
        "n_drop": max(1, round(base["n_drop"] * turn_scale)),
        "cost_threshold": round(base["cost_threshold"] * turn_scale, 4),
        "sell_rank_buffer": base["sell_rank_buffer"],
    }


def compute_composite_scores(factors, all_data, factor_cache, today,
                             fund_panel=None):
    """IC加权合成 (简化版: 价量 + 基本面 + 相对, 无分钟)。"""
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f.get("weight_multiplier", 1.0)
                        for f in factors])
    abs_w = np.sum(np.abs(weights))
    if abs_w < 1e-9:
        return {}

    raw = {n: {} for n in factor_names}

    # 价量
    for sym in all_data:
        feats = factor_cache.get(sym, today)
        if feats is None:
            continue
        for f in factors:
            if f.get("category") == "price_volume":
                n = f["name"]
                if n in feats and not np.isnan(feats[n]):
                    raw[n][sym] = feats[n]

    # 基本面
    if fund_panel:
        try:
            from fundamental_fetcher import compute_fundamental_factors
            fv = compute_fundamental_factors(fund_panel, all_data, today)
            for sym, vals in fv.items():
                for n, v in vals.items():
                    if n in raw and not np.isnan(v):
                        raw[n][sym] = v
        except Exception:
            pass

    # 相对
    try:
        from relative_factors import compute_relative_factors_batch
        rv = compute_relative_factors_batch(all_data, today)
        for sym, vals in rv.items():
            for n, v in vals.items():
                if n in raw and not np.isnan(v):
                    raw[n][sym] = v
    except Exception:
        pass

    # 覆盖率 (>=50%)
    n_f = len(factor_names)
    cov = {}
    for n in factor_names:
        for s in raw[n]:
            cov[s] = cov.get(s, 0) + 1
    valid = sorted(s for s, c in cov.items() if c >= n_f * 0.5)
    if len(valid) < 10:
        return {}

    # z-score + IC加权
    composite = np.zeros(len(valid))
    for fi, n in enumerate(factor_names):
        vals = np.array([raw[n].get(s, np.nan) for s in valid])
        m = ~np.isnan(vals)
        if m.sum() < 10:
            continue
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        if sd < 1e-9:
            continue
        z = np.where(m, (vals - mu) / sd, 0.0)
        composite += weights[fi] * z
    composite /= abs_w
    return dict(zip(valid, composite.tolist()))


def run_single_backtest(factors, all_data, factor_cache, fund_panel,
                        bt_config, regime_det, params, label):
    """单组参数回测 (简化版: 无PIT, 无DateRangeGuard)。"""
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker

    bt = SimpleBacktest(
        initial_capital=bt_config["initial_capital"],
        top_k=bt_config["top_k"],
        lot_size=bt_config["lot_size"],
        slippage_bps=bt_config["slippage_bps"],
        turnover_limit_pct=1.0,
    )
    rules = TradingRules()
    ranker = PortfolioRanker(
        top_k=bt_config["top_k"],
        n_drop=10, hold_thresh=30,
        sell_rank_buffer=3, buy_confirm_days=1,
        cost_threshold=0.08,
    )

    # 交易日历
    all_dates = set()
    for df in all_data.values():
        all_dates.update(pd.to_datetime(df["date"]).dt.date.tolist())
    bt_dates = sorted(d for d in all_dates
                      if pd.Timestamp(DEV_START).date() <= d <= pd.Timestamp(DEV_END).date())
    if not bt_dates:
        return None

    equity_curve = []
    daily_returns = []
    rebalance_count = 0
    pending = None
    prev_equity = float(bt_config["initial_capital"])
    risk_scale = 1.0
    turnover_history = []

    for di, today in enumerate(bt_dates):
        today_ts = pd.Timestamp(today)

        # T+1 执行
        if pending is not None:
            bt.execute(pending, today_ts, all_data, rules)
            pending = None

        # 调仓日
        if di % bt_config["rebalance_days"] == 0:
            regime = regime_det.detect(today_ts)
            adapted = adapt_weights_custom(factors, regime, params)
            tp = get_turnover_params_custom(regime, params["turn_scale"])
            ranker.hold_thresh = tp["hold_thresh"]
            ranker.n_drop = tp["n_drop"]
            ranker.cost_threshold = tp["cost_threshold"]
            ranker.sell_rank_buffer = tp["sell_rank_buffer"]

            # 评分
            scores = compute_composite_scores(
                adapted, all_data, factor_cache, today_ts,
                fund_panel=fund_panel)

            if scores and len(scores) >= bt_config["top_k"]:
                tradeable = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][all_data[sym]["date"] <= today_ts].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable[sym] = sc

                if len(tradeable) >= bt_config["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable, holdings)
                    decision["buy"] = [
                        s for s in decision.get("buy", [])
                        if s in all_data and rules.can_buy(
                            s, all_data[s][all_data[s]["date"] <= today_ts].tail(2))]
                    decision["sell"] = [
                        s for s in decision.get("sell", [])
                        if s in all_data and rules.can_sell(
                            s, all_data[s][all_data[s]["date"] <= today_ts].tail(2))]
                    if risk_scale < 1.0 and decision.get("buy"):
                        n_keep = max(1, int(len(decision["buy"]) * risk_scale))
                        decision["buy"] = decision["buy"][:n_keep]
                    pending = decision
                    rebalance_count += 1
                    n_turn = (len(decision.get("sell", [])) +
                              len(decision.get("buy", []))) / (2 * bt_config["top_k"])
                    turnover_history.append(n_turn)

        # Mark-to-market
        close_prices = {}
        for sym in list(bt.positions.keys()):
            if sym in all_data:
                dt = all_data[sym][all_data[sym]["date"] <= today_ts].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])
        equity = bt.mark_to_market(close_prices)
        equity_curve.append(equity)
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

        # 风控
        peak_equity = max(equity_curve)
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        risk_scale = 0.5 if current_dd < -0.15 else 1.0

    # ── 统计 ──
    eq = np.array(equity_curve)
    rets = np.array(daily_returns)
    total_return = eq[-1] / bt_config["initial_capital"] - 1
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准
    bench_annual = 0.0
    bench_daily = np.zeros(len(rets))
    if os.path.exists(BENCH_PATH):
        bdf = pd.read_parquet(BENCH_PATH)
        bdf["date"] = pd.to_datetime(bdf["date"])
        bdf = bdf.set_index("date")
        rs = pd.Timestamp(DEV_START)
        re_ = pd.Timestamp(DEV_END)
        bm = (bdf.index >= rs) & (bdf.index <= re_)
        bs = bdf.loc[bm, "close"]
        if len(bs) > 1:
            bt_total = bs.iloc[-1] / bs.iloc[0] - 1
            bench_annual = (1 + bt_total) ** (1 / max(n_years, 0.1)) - 1
            bench_daily = bs.pct_change().dropna().values

    excess = annual_return - bench_annual

    # Sharpe (rf=2.5%)
    rf_daily = 0.025 / 252
    excess_daily = rets - rf_daily
    sharpe = (np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252)
              if np.std(excess_daily) > 0 else 0)

    # IR
    if len(bench_daily) >= len(rets):
        active = rets - bench_daily[:len(rets)]
    else:
        active = rets - np.pad(bench_daily, (0, len(rets) - len(bench_daily)))
    ir = (np.mean(active) / np.std(active) * np.sqrt(252)
          if np.std(active) > 0 else 0)

    # MaxDD
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd))

    avg_turnover = np.mean(turnover_history) if turnover_history else 0

    return {
        "label": label,
        "params": {k: v for k, v in params.items()},
        "annual_return": round(annual_return * 100, 2),
        "benchmark_annual": round(bench_annual * 100, 2),
        "excess_annual": round(excess * 100, 2),
        "sharpe": round(sharpe, 3),
        "ir": round(ir, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "n_rebalances": rebalance_count,
        "avg_turnover": round(avg_turnover * 100, 1),
    }


def main():
    t_start = time.time()

    log.info("=" * 60)
    log.info("  Regime 参数敏感性测试")
    log.info("  期间: %s ~ %s (Development)", DEV_START, DEV_END)
    log.info("=" * 60)

    # 加载参数
    import yaml
    with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bt_config = {
        "rebalance_days": 20,
        "top_k": cfg["execution"]["top_k"],
        "initial_capital": cfg["execution"]["initial_capital"],
        "lot_size": cfg["execution"]["lot_size"],
        "slippage_bps": cfg["execution"]["slippage_bps"],
    }
    log.info("  top_k=%d, lot=%d, slippage=%dbps",
             bt_config["top_k"], bt_config["lot_size"], bt_config["slippage_bps"])

    # 加载因子
    factors = load_factors()
    log.info("  因子: %d 个", len(factors))

    # 加载数据
    log.info("加载数据...")
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))

    # 预计算因子
    log.info("预计算因子...")
    from factor_scorer import FactorScorer
    from factor_cache import FactorCache
    scorer = FactorScorer.from_preset("full_auto")
    pv_names = sorted(scorer.factor_weights.keys())
    factor_cache = FactorCache(scorer, pv_names)
    symbols = sorted(all_data.keys())
    t0 = time.time()
    for i in range(0, len(symbols), 200):
        batch = {s: all_data[s] for s in symbols[i:i + 200]}
        factor_cache.precompute(batch)
        elapsed = time.time() - t0
        if (i + 200) % 1000 == 0 or i + 200 >= len(symbols):
            log.info("  %d/%d (%.0fs)", min(i + 200, len(symbols)), len(symbols), elapsed)
    log.info("  因子预计算完成: %.0fs", time.time() - t0)

    # 基本面
    fund_panel = {}
    if any(f.get("category") == "fundamental" for f in factors):
        try:
            from fundamental_fetcher import load_fundamental_panel
            fund_panel = load_fundamental_panel()
            log.info("基本面: %d 只", len(fund_panel))
        except Exception as e:
            log.warning("基本面: %s", e)

    # Regime 检测器
    regime_det = RegimeDetector.from_benchmark_parquet(BENCH_PATH)

    # 跑各组参数
    results = []
    for label, params in PARAM_SETS.items():
        t1 = time.time()
        log.info("")
        log.info("-" * 40)
        log.info("[%s] up=(%.1f, %.2f), down=(%.1f, %.1f), turn_scale=%.1f",
                 label, params["up_pos"], params["up_neg"],
                 params["down_pos"], params["down_neg"], params["turn_scale"])
        r = run_single_backtest(factors, all_data, factor_cache, fund_panel,
                                bt_config, regime_det, params, label)
        if r:
            results.append(r)
            log.info("  年化超额: %+.2f%%  IR: %.3f  MaxDD: %.2f%%  Sharpe: %.3f  (%.0fs)",
                     r["excess_annual"], r["ir"], r["max_drawdown"], r["sharpe"],
                     time.time() - t1)
        else:
            log.warning("  [%s] 回测失败", label)

    # 汇总
    log.info("")
    log.info("=" * 60)
    log.info("  汇总")
    log.info("=" * 60)
    log.info("  %-16s %8s %6s %8s %7s %6s",
             "Params", "Excess", "IR", "MaxDD", "Sharpe", "Turn")
    log.info("  " + "-" * 60)
    for r in results:
        log.info("  %-16s %+7.2f%% %6.3f %7.2f%% %7.3f %5.1f%%",
                 r["label"], r["excess_annual"], r["ir"],
                 r["max_drawdown"], r["sharpe"], r["avg_turnover"])

    # 稳定性分析
    if results:
        excess_vals = [r["excess_annual"] for r in results]
        ir_vals = [r["ir"] for r in results]
        log.info("")
        log.info("  稳定性分析:")
        log.info("    超额收益: mean=%+.2f%%, std=%.2f%%, range=[%+.2f%%, %+.2f%%]",
                 np.mean(excess_vals), np.std(excess_vals),
                 min(excess_vals), max(excess_vals))
        log.info("    IR: mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
                 np.mean(ir_vals), np.std(ir_vals),
                 min(ir_vals), max(ir_vals))
        log.info("    变异系数 (超额): %.1f%%",
                 np.std(excess_vals) / abs(np.mean(excess_vals)) * 100
                 if abs(np.mean(excess_vals)) > 0.01 else float("inf"))

        # 结论
        cv = (np.std(excess_vals) / abs(np.mean(excess_vals)) * 100
              if abs(np.mean(excess_vals)) > 0.01 else float("inf"))
        all_positive = all(e > 0 for e in excess_vals)
        if all_positive and cv < 30:
            conclusion = "ROBUST: 所有参数组均正超额, 变异系数<30%"
        elif all_positive:
            conclusion = "MODERATELY ROBUST: 所有参数组正超额, 但波动较大"
        else:
            conclusion = "FRAGILE: 部分参数组超额为负, 策略对参数敏感"
        log.info("")
        log.info("  结论: %s", conclusion)

    # 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Regime参数敏感性测试 (简化回测, 无PIT)",
            "period": f"{DEV_START} ~ {DEV_END}",
            "n_factors": len(factors),
            "n_stocks": len(all_data),
            "total_runtime_s": round(time.time() - t_start, 1),
        },
        "results": results,
    }
    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("")
    log.info("  总耗时: %.0fs", time.time() - t_start)
    log.info("  结果: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
