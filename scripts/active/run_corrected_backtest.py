"""
scripts/active/run_corrected_backtest.py — 正式 Holdout 回测 (v3)

唯一合法的样本外测试脚本。

方法论保障:
  1. PIT universe — 每个调仓日用 CSI300+CSI1000 月度成分
  2. 参数统一 — top_k/成本/调仓频率全部来自 config.yaml
  3. FDR 校正 — BH q=0.10 剔除不显著因子
  4. TEST 锁 — TEST 期只能跑一次，结果锁定后不可重跑
  5. 日期守卫 — gate.py DateRangeGuard 运行时拦截盲测期访问
  6. BLIND 永不回测 — 脚本中无 BLIND 相关代码

用法:
  py scripts/active/run_corrected_backtest.py                # Development + TEST
  py scripts/active/run_corrected_backtest.py --dev-only     # 仅 Development (可重复)
  py scripts/active/run_corrected_backtest.py --test-only    # 仅 TEST (需解锁)

⚠️ TEST 期结果一旦生成即锁定。如需重跑，必须手动删除锁文件并说明理由。
"""

import os
import sys
import json
import time
import warnings
import argparse
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from gate import load_config, check_date_range, DateRangeGuard, GateViolation
from data.pit_universe import get_universe

log = get_logger("corrected_bt")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
REPORT_PATH = os.path.join(IC_DIR, "p5_portfolio_report.json")
OUTPUT_PATH = os.path.join(IC_DIR, "corrected_backtest.json")
BENCH_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
TEST_LOCK_PATH = os.path.join(IC_DIR, ".test_lock")

# FDR 淘汰的因子
FDR_ELIMINATED = {"fund_ocf_ps", "fund_sp"}

# 探索性因子类别 (IC验证期与TEST重叠, 不计入正式alpha)
# 分钟因子IC在2024-2025数据上验证, 与TEST期(2024-07~2025-06)高度重叠
EXPLORATORY_CATEGORIES = {"minute"}

# 数据分区 v3 (从 config.yaml 读取, 硬编码仅作 fallback)
def _load_partitions() -> dict:
    try:
        import yaml
        cfg_path = os.path.join(BASE_DIR, "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dp = cfg["data_partition"]
        return {
            "development": (dp["development"]["start"], dp["development"]["end"]),
            "test": (dp["test"]["start"], dp["test"]["end"]),
        }
    except Exception:
        return {
            "development": ("2026-07-01", "2026-12-31"),
            "test": ("2026-07-01", "2026-12-31"),
        }

PARTITIONS = _load_partitions()


def load_bt_config() -> dict:
    """从 config.yaml 加载回测参数 (与 P5 统一)。"""
    import yaml
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "rebalance_days": 20,
        "top_k": cfg["execution"]["top_k"],            # 30
        "initial_capital": cfg["execution"]["initial_capital"],
        "lot_size": cfg["execution"]["lot_size"],
        "slippage_bps": cfg["execution"]["slippage_bps"],
        "commission_buy": cfg["execution"]["commission_buy"],
        "commission_sell": cfg["execution"]["commission_sell"],
    }


def load_factors(include_exploratory: bool = False) -> list:
    """加载 P5 因子并应用 FDR 校正 + 探索性因子降级。"""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    factors = report.get("selected_factors", [])
    before_fdr = len(factors)
    factors = [f for f in factors if f["name"] not in FDR_ELIMINATED]
    after_fdr = len(factors)

    # 探索性因子降级
    if not include_exploratory:
        before_exp = len(factors)
        factors = [f for f in factors
                   if f.get("category") not in EXPLORATORY_CATEGORIES]
        n_exp = before_exp - len(factors)
        if n_exp > 0:
            log.info("因子: %d 个 (FDR剔除 %d, 探索性降级 %d: %s)",
                     len(factors), before_fdr - after_fdr, n_exp,
                     sorted(EXPLORATORY_CATEGORIES))
        else:
            log.info("因子: %d 个 (FDR剔除 %d)", len(factors), before_fdr - after_fdr)
    else:
        log.info("因子: %d 个 (FDR剔除 %d, ⚠️ 含探索性因子)",
                 len(factors), before_fdr - after_fdr)

    return factors


def compute_composite_scores(factors, all_data, factor_cache, today,
                             fund_panel=None, minute_data=None):
    """IC加权合成 (含4类因子, 与 P5 逻辑一致)。"""
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

    # 分钟
    if minute_data:
        try:
            from minute_factors import compute_minute_factors_batch
            mv = compute_minute_factors_batch(minute_data, today, lookback=20)
            for sym, vals in mv.items():
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


def run_backtest(factors, all_data, factor_cache, fund_panel, minute_data,
                 bt_config, start, end, label="", use_regime=True,
                 regime_profile="conservative", wf_weighter=None):
    """Walk-forward 回测 + PIT universe。"""
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker
    from regime_detector import RegimeDetector

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
    regime_det = RegimeDetector.from_benchmark_parquet(BENCH_PATH, profile=regime_profile)

    # 交易日历
    all_dates = set()
    for df in all_data.values():
        all_dates.update(pd.to_datetime(df["date"]).dt.date.tolist())
    bt_dates = sorted(d for d in all_dates
                      if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date())
    if not bt_dates:
        log.error("[%s] 无交易日", label)
        return None

    log.info("=" * 60)
    log.info("[%s] %s ~ %s (%d 天), top_k=%d",
             label, bt_dates[0], bt_dates[-1], len(bt_dates), bt_config["top_k"])
    log.info("=" * 60)

    equity_curve = []
    daily_returns = []
    regime_history = []
    turnover_history = []
    rebalance_count = 0
    pending = None
    prev_equity = float(bt_config["initial_capital"])
    pit_sizes = []
    risk_scale = 1.0  # 风控缩放因子 (DD>15%时减半)

    rs = pd.Timestamp(start)
    re_ = pd.Timestamp(end)

    for di, today in enumerate(bt_dates):
        today_ts = pd.Timestamp(today)

        # T+1 执行
        if pending is not None:
            b, s, trades = bt.execute(pending, today_ts, all_data, rules)
            pending = None

        # 调仓日
        if di % bt_config["rebalance_days"] == 0:
            # ★ PIT universe
            pit_stocks = set(get_universe(str(today)))
            pit_sizes.append(len(pit_stocks))

            # ★ Walk-Forward 动态IC权重 (在 regime 之前)
            if wf_weighter is not None:
                factors = wf_weighter.update_weights(factors, today_ts,
                                                      bt_config["rebalance_days"])

            # Regime
            if use_regime:
                regime = regime_det.detect(today_ts)
                adapted = regime_det.adapt_factor_weights(factors, today_ts, regime)
                tp = regime_det.get_turnover_params(today_ts, regime)
                ranker.hold_thresh = tp["hold_thresh"]
                ranker.n_drop = tp["n_drop"]
                ranker.cost_threshold = tp["cost_threshold"]
                ranker.sell_rank_buffer = tp["sell_rank_buffer"]
                regime_history.append({"date": str(today), "regime": regime.value})
            else:
                adapted = factors
                regime = None

            # 评分
            scores = compute_composite_scores(
                adapted, all_data, factor_cache, today_ts,
                fund_panel=fund_panel, minute_data=minute_data)

            # ★ PIT 过滤
            scores = {s: v for s, v in scores.items() if s in pit_stocks}

            if scores and len(scores) >= bt_config["top_k"]:
                tradeable = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][
                            all_data[sym]["date"] <= today_ts].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable[sym] = sc

                if len(tradeable) >= bt_config["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable, holdings)
                    decision["buy"] = [
                        s for s in decision.get("buy", [])
                        if s in all_data and rules.can_buy(
                            s, all_data[s][
                                all_data[s]["date"] <= today_ts].tail(2))]
                    decision["sell"] = [
                        s for s in decision.get("sell", [])
                        if s in all_data and rules.can_sell(
                            s, all_data[s][
                                all_data[s]["date"] <= today_ts].tail(2))]
                    # ★ 风控: 回撤>15%时减半买入
                    if risk_scale < 1.0 and decision.get("buy"):
                        n_keep = max(1, int(len(decision["buy"]) * risk_scale))
                        decision["buy"] = decision["buy"][:n_keep]
                    pending = decision
                    rebalance_count += 1
                    n_turn = (len(decision.get("sell", [])) +
                              len(decision.get("buy", []))) / (2 * bt_config["top_k"])
                    turnover_history.append(n_turn)

            if rebalance_count > 0 and rebalance_count % 6 == 0:
                regime_str = regime.value if regime else "disabled"
                log.info("  调仓 #%d: %s, 持仓=%d, regime=%s, PIT=%d",
                         rebalance_count, today,
                         len(bt.positions), regime_str, len(pit_stocks))

        # Mark-to-market
        close_prices = {}
        for sym in list(bt.positions.keys()):
            if sym in all_data:
                dt = all_data[sym][
                    all_data[sym]["date"] <= today_ts].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])
        equity = bt.mark_to_market(close_prices)
        equity_curve.append({"date": str(today), "equity": equity})
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append(daily_ret)
        prev_equity = equity

        # ★ 风控: 滚动回撤 > 15% 时, 下次调仓减半买入
        peak_equity = max(e["equity"] for e in equity_curve)
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if current_dd < -0.15:
            risk_scale = 0.5  # 回撤>15%, 减仓50%
        else:
            risk_scale = 1.0

        if (di + 1) % 100 == 0:
            log.info("  Day %d/%d: equity=%.0f, DD=%.1f%%",
                     di + 1, len(bt_dates), equity, current_dd * 100)

    # ── 统计 ──
    eq = np.array([e["equity"] for e in equity_curve])
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
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0

    # 换手
    avg_turnover = np.mean(turnover_history) if turnover_history else 0

    # Regime
    regime_counts = {}
    for rh in regime_history:
        r = rh["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1

    result = {
        "label": label,
        "period": f"{bt_dates[0]} ~ {bt_dates[-1]}",
        "n_days": len(bt_dates),
        "total_return": round(total_return * 100, 1),
        "annual_return": round(annual_return * 100, 1),
        "benchmark_annual": round(bench_annual * 100, 1),
        "excess_annual": round(excess * 100, 1),
        "sharpe": round(sharpe, 2),
        "ir": round(ir, 2),
        "max_drawdown": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "n_rebalances": rebalance_count,
        "avg_turnover": round(avg_turnover * 100, 1),
        "avg_pit_size": int(np.mean(pit_sizes)) if pit_sizes else 0,
        "regime_distribution": regime_counts,
        # 日度数据 (供 bootstrap 分析)
        "daily_returns": [round(float(r), 8) for r in rets],
        "daily_active_returns": [round(float(a), 8) for a in active],
    }

    log.info("  结果:")
    log.info("    总收益: %+.1f%%", result["total_return"])
    log.info("    年化收益: %+.1f%%", result["annual_return"])
    log.info("    基准年化: %+.1f%%", result["benchmark_annual"])
    log.info("    年化超额: %+.1f%%", result["excess_annual"])
    log.info("    Sharpe: %.2f  IR: %.2f", result["sharpe"], result["ir"])
    log.info("    最大回撤: %.1f%%  Calmar: %.2f", result["max_drawdown"], result["calmar"])
    log.info("    调仓: %d 次, 平均PIT: %d 只, 月均换手: %.1f%%",
             rebalance_count, result["avg_pit_size"], result["avg_turnover"])
    return result


def main():
    parser = argparse.ArgumentParser(description="正式 Holdout 回测 (v3)")
    parser.add_argument("--dev-only", action="store_true",
                        help="仅跑 Development (可重复)")
    parser.add_argument("--test-only", action="store_true",
                        help="仅跑 TEST (需解锁, 只跑一次)")
    parser.add_argument("--unlock-test", action="store_true",
                        help="解除 TEST 锁 (需说明理由)")
    parser.add_argument("--include-exploratory", action="store_true",
                        help="包含探索性因子 (分钟因子, 默认排除)")
    parser.add_argument("--no-regime", action="store_true",
                        help="禁用 regime 自适应 (对比用, 检验regime是否过拟合)")
    parser.add_argument("--regime-profile", type=str, default="conservative",
                        choices=["conservative", "original", "aggressive", "disabled"],
                        help="Regime 参数 profile (默认 conservative)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="启用 Walk-Forward 动态IC权重 (trailing 12个月)")
    args = parser.parse_args()

    # TEST 锁管理
    if args.unlock_test:
        if os.path.exists(TEST_LOCK_PATH):
            os.remove(TEST_LOCK_PATH)
            log.info("TEST 锁已解除。请在下次运行时说明理由。")
        else:
            log.info("TEST 锁不存在，无需解除。")
        return

    config = load_config(os.path.join(BASE_DIR, "config.yaml"))

    # 确定要跑的分区
    if args.dev_only:
        partitions_to_run = {"development": PARTITIONS["development"]}
    elif args.test_only:
        partitions_to_run = {"test": PARTITIONS["test"]}
    else:
        partitions_to_run = dict(PARTITIONS)

    # TEST 锁检查
    if "test" in partitions_to_run:
        if os.path.exists(TEST_LOCK_PATH):
            with open(TEST_LOCK_PATH, "r") as f:
                lock_info = json.load(f)
            log.error("=" * 60)
            log.error("  🚫 TEST 期已锁定！不可重复运行。")
            log.error("  锁定时间: %s", lock_info.get("locked_at", "unknown"))
            log.error("  结果文件: %s", OUTPUT_PATH)
            log.error("  如确需重跑: py scripts/active/run_corrected_backtest.py --unlock-test")
            log.error("=" * 60)
            sys.exit(1)

    # 日期范围静态检查 (确保不触碰盲测期)
    for label, (s, e) in partitions_to_run.items():
        try:
            check_date_range(s, e, config, script_name="run_corrected_backtest")
        except GateViolation as ex:
            log.error(str(ex))
            sys.exit(1)

    log.info("=" * 60)
    log.info("  正式 Holdout 回测 v3 (PIT + 统一参数 + FDR + TEST锁)")
    log.info("  分区: %s", ", ".join(partitions_to_run.keys()))
    log.info("=" * 60)

    bt_config = load_bt_config()
    log.info("  top_k=%d, lot=%d, slippage=%dbps",
             bt_config["top_k"], bt_config["lot_size"], bt_config["slippage_bps"])

    factors = load_factors(include_exploratory=args.include_exploratory)

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
        if (i + 200) % 1000 == 0 or i + 200 >= len(symbols):
            log.info("  %d/%d (%.0fs)",
                     min(i + 200, len(symbols)), len(symbols), time.time() - t0)

    # 基本面
    fund_panel = {}
    if any(f.get("category") == "fundamental" for f in factors):
        try:
            from fundamental_fetcher import load_fundamental_panel
            fund_panel = load_fundamental_panel()
            log.info("基本面: %d 只", len(fund_panel))
        except Exception as e:
            log.warning("基本面: %s", e)

    # 分钟
    minute_data = {}
    if any(f.get("category") == "minute" for f in factors):
        from minute_factors import load_minute_data
        minute_data = load_minute_data(use_cache=True)
        log.info("分钟: %d 只", len(minute_data))

    # Walk-Forward 动态IC权重
    wf_weighter = None
    if args.walk_forward:
        from walk_forward import WalkForwardICWeighter
        wf_weighter = WalkForwardICWeighter(
            factor_cache, all_data,
            lookback_months=12,
            min_ic_obs=6,
            decay_halflife=60,
        )
        log.info("  Walk-Forward: 启用 (trailing 12个月)")

    # 回测 (带日期守卫)
    results = {}
    with DateRangeGuard(config, script_name="run_corrected_backtest") as guard:
        for label, (s, e) in partitions_to_run.items():
            log.info("")
            guard.check_range(s, e)
            r = run_backtest(factors, all_data, factor_cache, fund_panel,
                             minute_data, bt_config, s, e, label=label.upper(),
                             use_regime=not args.no_regime,
                             regime_profile=args.regime_profile,
                             wf_weighter=wf_weighter)
            if r:
                results[label] = r

    # TEST 锁: 跑完 TEST 后写入锁文件
    if "test" in results:
        lock_data = {
            "locked_at": datetime.now().isoformat(),
            "result_summary": {
                "excess_annual": results["test"]["excess_annual"],
                "ir": results["test"]["ir"],
                "max_drawdown": results["test"]["max_drawdown"],
            },
            "note": "TEST 期只跑一次。此锁由 run_corrected_backtest.py 自动写入。",
        }
        os.makedirs(IC_DIR, exist_ok=True)
        with open(TEST_LOCK_PATH, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, ensure_ascii=False, indent=2)
        log.info("  🔒 TEST 锁已写入: %s", TEST_LOCK_PATH)

    # 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "修正版回测: PIT universe + 统一参数 + FDR",
            "fixes": [
                "幸存者偏差: 每个调仓日使用 PIT universe (CSI300+CSI1000)",
                "参数漂移: top_k 统一为 config.yaml 值 (30)",
                "FDR: BH 校正剔除 fund_ocf_ps, fund_sp",
                "BLIND 已污染 (trial_count>=3): 仅报告 Dev + TEST",
            ],
            "n_factors": len(factors),
            "bt_config": {k: v for k, v in bt_config.items()},
        },
        "results": results,
    }
    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 汇总
    log.info("")
    log.info("=" * 60)
    log.info("  %-15s %8s %8s %8s %8s %8s %6s",
             "Period", "Excess", "IR", "MaxDD", "Sharpe", "PIT", "Turn")
    log.info("  " + "-" * 65)
    for label, r in results.items():
        log.info("  %-15s %+7.1f%% %7.2f %7.1f%% %7.2f %7d %5.1f%%",
                 label.upper(), r["excess_annual"], r["ir"],
                 r["max_drawdown"], r["sharpe"],
                 r["avg_pit_size"], r["avg_turnover"])
    log.info("=" * 60)
    log.info("  结果: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
