"""
Paper Trading v3 — P5全因子组合信号 + 模拟盘执行

替代 run_paper_signal.py, 使用P5验证通过的因子组合:
  - 从 p5_portfolio_report.json 加载因子列表和ICIR权重
  - IC加权线性组合 (无ML, 纯线性)
  - 截面z-score → 排名 → PortfolioRanker → PaperExecutor

用法:
  py scripts/run_paper_signal_v3.py              # 最新交易日
  py scripts/run_paper_signal_v3.py 2026-08-04   # 指定日期
  py scripts/run_paper_signal_v3.py --dry-run    # 仅生成信号, 不执行
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

REPORT_PATH = os.path.join(BASE_DIR, "data", "ic_validation", "p5_portfolio_report.json")
SIGNAL_LOG = os.path.join(BASE_DIR, "data", "paper_signals_v3.jsonl")


# ════════════════════════════════════════════════════════════
#  因子配置加载
# ════════════════════════════════════════════════════════════

def load_factor_config() -> Optional[dict]:
    """
    从 p5_portfolio_report.json 加载因子配置。

    Returns:
      {"factors": [{"name": ..., "icir": ..., "weight_multiplier": ...}], "bt_config": {...}}
      或 None (文件不存在)
    """
    if not os.path.exists(REPORT_PATH):
        return None

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    factors = report.get("selected_factors", [])
    if not factors:
        return None

    return {
        "factors": factors,
        "bt_config": report.get("meta", {}).get("bt_config", {}),
        "verdict": report.get("verdict", "UNKNOWN"),
    }


def build_fallback_config() -> dict:
    """
    当 p5_portfolio_report.json 不存在时的回退配置。
    直接使用P3验证的最强价量因子。
    """
    # 从 p3_alpha158_ic.json 取 horizon=20, |ICIR|>0.3 的因子
    p3_path = os.path.join(BASE_DIR, "data", "ic_validation", "p3_alpha158_ic.json")
    factors = []

    if os.path.exists(p3_path):
        with open(p3_path, "r", encoding="utf-8") as f:
            p3 = json.load(f)
        for r in p3.get("results", []):
            if r["horizon"] == 20 and abs(r["icir"]) > 0.3:
                factors.append({
                    "name": r["factor"],
                    "icir": r["icir"],
                    "category": "price_volume",
                    "weight_multiplier": 1.0,
                })

    # 补充早期验证的因子
    ic_results_path = os.path.join(BASE_DIR, "data", "ic_results.json")
    if os.path.exists(ic_results_path):
        with open(ic_results_path, "r", encoding="utf-8") as f:
            ic_results = json.load(f)
        existing_names = {f["name"] for f in factors}
        for r in ic_results:
            if r["factor"] not in existing_names and abs(r["icir"]) > 0.3:
                factors.append({
                    "name": r["factor"],
                    "icir": r["icir"],
                    "category": "price_volume",
                    "weight_multiplier": 1.0,
                })

    factors.sort(key=lambda x: -abs(x["icir"]))
    return {"factors": factors, "bt_config": {}, "verdict": "FALLBACK"}


# ════════════════════════════════════════════════════════════
#  信号生成
# ════════════════════════════════════════════════════════════

def compute_composite_scores_live(factors: List[dict], all_data: dict,
                                  factor_cache, today,
                                  minute_data: dict = None,
                                  fund_panel: dict = None) -> Dict[str, float]:
    """
    IC加权线性组合 (与回测 DecisionEngine 逻辑一致):
      composite_i = sum(ICIR_j * z_score(factor_j_i)) / sum(|ICIR_j|)

    支持四类因子: price_volume, fundamental, relative, minute
    使用 50% 覆盖率门槛 (与 DecisionEngine 一致)。
    """
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f.get("weight_multiplier", 1.0) for f in factors])
    abs_weight_sum = np.sum(np.abs(weights))
    if abs_weight_sum < 1e-9:
        return {}

    # 收集各类因子原始值
    raw_values = {name: {} for name in factor_names}

    # ── 价量因子 (from FactorCache) ──
    pv_factors = [f for f in factors if f.get("category") == "price_volume"]
    if pv_factors and factor_cache:
        for sym in all_data:
            feats = factor_cache.get(sym, today)
            if feats is None:
                continue
            for f in pv_factors:
                name = f["name"]
                if name in feats and not np.isnan(feats[name]):
                    raw_values[name][sym] = feats[name]

    # ── 基本面因子 ──
    fund_factors = [f for f in factors if f.get("category") == "fundamental"]
    if fund_factors and fund_panel:
        try:
            from fundamental_fetcher import compute_fundamental_factors
            fund_values = compute_fundamental_factors(fund_panel, all_data, today)
            for sym, fvals in fund_values.items():
                for name, val in fvals.items():
                    if name in raw_values and not np.isnan(val):
                        raw_values[name][sym] = val
        except Exception as e:
            print(f"  [WARN] 基本面因子计算失败: {e}", flush=True)

    # ── 相对因子 ──
    rel_factors = [f for f in factors if f.get("category") == "relative"]
    if rel_factors:
        try:
            from relative_factors import compute_relative_factors_batch
            rel_values = compute_relative_factors_batch(all_data, today)
            for sym, fvals in rel_values.items():
                for name, val in fvals.items():
                    if name in raw_values and not np.isnan(val):
                        raw_values[name][sym] = val
        except Exception as e:
            print(f"  [WARN] 相对因子计算失败: {e}", flush=True)

    # ── 分钟频因子 ──
    min_factors = [f for f in factors if f.get("category") == "minute"]
    if min_factors and minute_data:
        try:
            from minute_factors import compute_minute_factors_batch
            min_values = compute_minute_factors_batch(
                minute_data, today, lookback=20)
            for sym, fvals in min_values.items():
                for name, val in fvals.items():
                    if name in raw_values and not np.isnan(val):
                        raw_values[name][sym] = val
        except Exception as e:
            print(f"  [WARN] 分钟因子计算失败: {e}", flush=True)

    # 覆盖率过滤: 至少 50% 因子有值 (与 DecisionEngine 一致)
    n_factors = len(factor_names)
    sym_coverage = {}
    for name in factor_names:
        for sym in raw_values[name]:
            sym_coverage[sym] = sym_coverage.get(sym, 0) + 1
    valid_syms = sorted(
        s for s, c in sym_coverage.items() if c >= n_factors * 0.5)

    if len(valid_syms) < 10:
        return {}

    # 截面 z-score + IC加权
    n = len(valid_syms)
    composite = np.zeros(n)
    for fi, name in enumerate(factor_names):
        vals = np.array([raw_values[name].get(s, np.nan) for s in valid_syms])
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 10:
            continue
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-9:
            continue
        z = np.where(valid_mask, (vals - mean) / std, 0.0)
        composite += weights[fi] * z

    composite /= abs_weight_sum
    return dict(zip(valid_syms, composite.tolist()))


def check_minute_staleness(latest_minute_date, daily_dates,
                           threshold: int = 5):
    """分钟数据陈旧度判定: 落后信号日的交易日数。

    latest_minute_date: date|None (本地分钟数据最新日期)
    daily_dates: 日线交易日序列 (DatetimeIndex/Series/list)
    Returns: (stale: bool, gap: int|None)
      latest_minute_date=None → (False, None) 不判定;
      gap = 分钟最新日期之后的交易日数量, >threshold 视为陈旧。
    """
    if latest_minute_date is None:
        return False, None
    dts = pd.to_datetime(daily_dates)
    m = pd.Timestamp(latest_minute_date)
    gap = int((dts > m).sum())
    return gap > threshold, gap


def generate_signal_v3(date_str: str = None, dry_run: bool = False):
    """
    生成P5因子组合信号并执行。

    Args:
      date_str: '2026-08-04' 或 None=最新交易日
      dry_run: True=仅生成信号不执行
    """
    print("=" * 60, flush=True)
    print("  Paper Trading v3 — P5全因子组合", flush=True)
    print("=" * 60, flush=True)

    # ── 0. 离线守卫: 信号生成只读本地数据 ──
    #    网络统一由 daily_pipeline 数据更新步骤前置完成; 本进程内随后的
    #    paper_executor 同样离线 (分钟数据新鲜度由获取阶段保证, 缺失走回退)。
    from netgate import set_offline_mode
    set_offline_mode(True)

    # ── 1. 加载因子配置 ──
    factor_config = load_factor_config()
    if factor_config is None:
        print("  [WARN] p5_portfolio_report.json 不存在, 使用回退配置", flush=True)
        factor_config = build_fallback_config()

    factors = factor_config["factors"]
    if not factors:
        print("  [ERROR] 无可用因子, 终止", flush=True)
        return None

    # ★ Alpha 衰减自动降级: 读取降级覆盖表, 将衰减因子权重置零
    try:
        from run_ic_monitor import load_downgrade_overrides
        downgrade = load_downgrade_overrides()
        if downgrade:
            n_downgraded = 0
            for f in factors:
                if f["name"] in downgrade:
                    f["weight_multiplier"] = downgrade[f["name"]]
                    n_downgraded += 1
            if n_downgraded > 0:
                print(f"  [降级] {n_downgraded} 个因子已被IC衰减监控降级", flush=True)
            # 过滤掉权重为0的因子
            factors = [f for f in factors if f.get("weight_multiplier", 1.0) != 0.0]
            if not factors:
                print("  [ERROR] 所有因子均已被降级, 终止", flush=True)
                return None
    except ImportError:
        pass  # run_ic_monitor 不可用时跳过

    print(f"  因子数: {len(factors)}", flush=True)
    print(f"  验证状态: {factor_config.get('verdict', 'UNKNOWN')}", flush=True)
    for f in factors[:5]:
        print(f"    {f['name']:<25} ICIR={f['icir']:+.3f} x{f.get('weight_multiplier', 1.0):.1f}", flush=True)
    if len(factors) > 5:
        print(f"    ... 共 {len(factors)} 个", flush=True)

    # ── 2. 确定日期 ──
    from data_cache import get_cached_symbols, load as load_single

    if date_str is None:
        syms = get_cached_symbols()
        latest_date = None
        for sym in syms[:5]:
            df = load_single(sym)
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                last = df["date"].max()
                if latest_date is None or last > latest_date:
                    latest_date = last
        today = latest_date or pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(date_str)

    print(f"  信号日期: {today.date()}", flush=True)

    # ── 3. PIT Universe 过滤 ──
    print("  加载数据...", flush=True)
    from data_cache import load_all
    syms = get_cached_symbols()

    # PIT universe: 只保留信号日当天存在的股票 (消除幸存者偏差)
    try:
        from data.pit_universe import get_universe
        pit_syms = set(get_universe(str(today.date())))
        if pit_syms:
            syms = [s for s in syms if s in pit_syms]
            print(f"  PIT universe: {len(syms)} 只 (过滤幸存者偏差)", flush=True)
    except Exception as e:
        print(f"  [WARN] PIT universe 不可用, 使用全量: {e}", flush=True)

    all_data = load_all(syms)
    all_data = {s: df for s, df in all_data.items() if df is not None and len(df) >= 100}
    print(f"  有效数据: {len(all_data)} 只", flush=True)

    # ── 数据陈旧度标注 (结构性缺失守卫): 分钟数据落后信号日 >5 交易日 → 告警 ──
    _data_as_of = None
    try:
        from data.minute_fetcher import latest_local_minute_date
        m5_dir = os.path.join(BASE_DIR, "data_store", "minute_5m")
        latest_min = latest_local_minute_date(m5_dir)
        daily_dates = None
        for df in list(all_data.values())[:10]:
            if df is not None and len(df) > 0 and "date" in df.columns:
                daily_dates = pd.to_datetime(df["date"])
                break
        stale, gap = check_minute_staleness(latest_min, daily_dates)
        _data_as_of = {
            "minute_5m_latest": str(latest_min) if latest_min else None,
            "stale": bool(stale),
            "lag_trading_days": gap,
        }
        if stale:
            print(f"  ⚠️ 分钟数据滞后 {gap} 个交易日 (最新 {latest_min}), "
                  f"POV 定价将回退日线开盘价", flush=True)
    except Exception as e:
        print(f"  [WARN] 分钟数据陈旧度检查失败: {e}", flush=True)

    # ── 3b. 预计算价量因子 ──
    print("  预计算因子...", flush=True)
    from factor_scorer import FactorScorer
    from factor_cache import FactorCache

    scorer = FactorScorer.from_preset("full_auto")
    pv_names = sorted(scorer.factor_weights.keys())
    factor_cache = FactorCache(scorer, pv_names)
    factor_cache.precompute(all_data)
    print(f"  价量因子预计算完成: {len(pv_names)} 个", flush=True)

    # 保留所有可计算因子 (价量 + 基本面 + 相对 + 分钟)
    computable = set(pv_names)
    computable.update(f["name"] for f in factors
                     if f.get("category") in ("fundamental", "relative", "minute"))
    factors = [f for f in factors if f["name"] in computable]
    if not factors:
        print("  [ERROR] 无可计算因子", flush=True)
        return None

    # ── 3c. 加载分钟数据 (如有分钟因子) ──
    minute_data = {}
    has_minute = any(f.get("category") == "minute" for f in factors)
    if has_minute:
        try:
            from minute_factors import load_minute_data
            minute_data = load_minute_data(use_cache=True)
            print(f"  分钟数据: {len(minute_data)} 只", flush=True)
        except Exception as e:
            print(f"  [WARN] 分钟数据加载失败: {e}", flush=True)

    # ── 3d. 加载基本面数据 (如有基本面因子) ──
    fund_panel = {}
    has_fund = any(f.get("category") == "fundamental" for f in factors)
    if has_fund:
        try:
            from fundamental_fetcher import load_fundamental_panel
            fund_panel = load_fundamental_panel()
            print(f"  基本面数据: {len(fund_panel)} 只", flush=True)
        except Exception as e:
            print(f"  [WARN] 基本面数据加载失败: {e}", flush=True)

    # ── 3e. Regime 自适应权重调整 ──
    try:
        from regime_detector import RegimeDetector
        index_path = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")
        detector = RegimeDetector.from_benchmark_parquet(index_path)
        regime = detector.detect(today)
        factors = detector.adapt_factor_weights(factors, today, regime=regime)
        print(f"  Regime: {regime.value} (权重已自适应调整)", flush=True)
    except Exception as e:
        print(f"  [WARN] Regime 检测不可用: {e}", flush=True)

    # ── 4. 计算复合评分 (四类因子) ──
    print("  计算复合评分...", flush=True)
    scores = compute_composite_scores_live(
        factors, all_data, factor_cache, today,
        minute_data=minute_data, fund_panel=fund_panel)

    if not scores:
        print("  [ERROR] 无法计算评分 (数据不足?)", flush=True)
        return None

    print(f"  有效评分: {len(scores)} 只", flush=True)

    # ── 5. 排名决策 ──
    from portfolio_ranker import PortfolioRanker
    from trading_rules import TradingRules

    # 从config读取参数
    import yaml
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── 4b. 行业动量叠加通道 (v28, 回测已验证): composite += λ×截面z(ind_mom_60)
    #     与回测 score_stocks 行业通道同一数学 (apply_industry_lambda),
    #     广播值取 industry_mom_broadcast_at (与 industry_momentum_panel 逐值一致)。
    #     失败降级为核心分 (WARN), 不阻断信号 — 与分钟/基本面加载同风格。
    try:
        _lam = float(((config.get("styles") or {}).get("industry_lambda")) or 0.0)
    except (TypeError, ValueError):
        _lam = 0.0
    if _lam > 0:
        try:
            from earnings_surprise import (load_industry_map,
                                           industry_mom_broadcast_at,
                                           apply_industry_lambda)
            _imap = load_industry_map()
            if _imap:
                _bcast = industry_mom_broadcast_at(all_data, _imap, today)
                _n_before = len(scores)
                scores = apply_industry_lambda(scores, _bcast, _lam)
                _hit = sum(1 for s in scores if s in _bcast)
                print(f"  行业λ通道: λ={_lam:.2f} 覆盖 {_hit}/{_n_before} 只", flush=True)
            else:
                print("  [WARN] 行业映射缺失, 信号为核心分 (无行业λ叠加)", flush=True)
        except Exception as e:
            print(f"  [WARN] 行业λ通道失败, 信号为核心分: {e}", flush=True)

    top_k = config["execution"].get("top_k", 30)
    n_drop = config["portfolio"].get("n_drop", 2)
    hold_thresh = config["portfolio"].get("hold_thresh", 30)
    sell_rank_buffer = config["portfolio"].get("sell_rank_buffer", 2)
    buy_confirm_days = config["portfolio"].get("buy_confirm_days", 1)
    cost_threshold = config["portfolio"].get("cost_threshold", 0.08)

    rules = TradingRules()

    # 过滤不可交易
    tradeable_scores = {}
    for sym, sc in scores.items():
        if sym in all_data:
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(2)
            if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                tradeable_scores[sym] = sc

    if len(tradeable_scores) < top_k:
        print(f"  [WARN] 可交易股票不足 ({len(tradeable_scores)} < {top_k})", flush=True)

    # 加载当前持仓
    if dry_run:
        holdings = []
    else:
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(
            initial_capital=config["execution"]["initial_capital"],
            top_k=top_k,
            lot_size=config["execution"].get("lot_size", 100),
            slippage_bps=config["execution"].get("slippage_bps", 30),
            turnover_limit_pct=config["execution"].get("turnover_limit_pct", 0.5),
            max_single_pct=config["execution"].get("max_single_pct", 0.25),
            minute_mode=config["execution"].get("minute_mode", False),
            execution_algo=config["execution"].get("execution_algo", "vwap"),
            twap_slices=config["execution"].get("twap_slices", 8),
            residual_bps=config["execution"].get("vwap_residual_bps"),
            overlay=config["execution"].get("overlay"),
        )
        state = executor.load_state()
        holdings = list(state.positions.keys())
        print(f"  当前持仓: {len(holdings)} 只, 现金: {state.cash:,.0f}", flush=True)

    ranker = PortfolioRanker(
        top_k=top_k, n_drop=n_drop,
        hold_thresh=hold_thresh,
        sell_rank_buffer=sell_rank_buffer,
        buy_confirm_days=buy_confirm_days,
        cost_threshold=cost_threshold,
    )

    # 同步持有天数
    if not dry_run:
        for sym in holdings:
            ranker._hold_since[sym] = hold_thresh + 1  # 允许卖出

    decision = ranker.rank(tradeable_scores, holdings)

    # 涨跌停过滤
    decision["buy"] = [s for s in decision["buy"]
                      if s in all_data and rules.can_buy(
                          s, all_data[s][all_data[s]["date"] <= today].tail(2))]
    decision["sell"] = [s for s in decision["sell"]
                       if s in all_data and rules.can_sell(
                           s, all_data[s][all_data[s]["date"] <= today].tail(2))]

    # ── 6. 输出信号 ──
    print(f"\n  {'='*50}", flush=True)
    print(f"  信号 ({today.date()}):", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  买入: {decision.get('buy', [])}", flush=True)
    print(f"  卖出: {decision.get('sell', [])}", flush=True)
    print(f"  持有: {len(decision.get('hold', []))} 只", flush=True)
    print(f"  Top-{top_k}: {decision.get('top_k', [])[:10]}...", flush=True)

    # ── 7. 执行 (非dry-run) ──
    if dry_run:
        print(f"\n  [DRY-RUN] 不执行订单", flush=True)
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "signal_date": str(today.date()),
            "mode": "dry_run",
            "buy": decision.get("buy", []),
            "sell": decision.get("sell", []),
            "hold": decision.get("hold", []),
            "n_factors": len(factors),
            "verdict": factor_config.get("verdict", "UNKNOWN"),
            "data_as_of": _data_as_of,
        }
    else:
        # ★ 熔断检查: HALT 状态下禁止买入, 只允许卖出/持有
        from execution.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb_state = cb.check()
        buy_allowed, cb_reason = cb.allow_buy()

        original_buy = list(decision.get("buy", []))
        if not buy_allowed:
            print(f"\n  🛑 熔断检查: {cb_reason}, 禁止买入", flush=True)
            decision["buy"] = []  # 清空买入列表, 保留卖出
        elif cb_state == "warning":
            _, _, dd = cb.calculate_drawdown()
            print(f"\n  ⚠️ 回撤警告: {dd:.2%}, 继续运行但需关注", flush=True)

        # 构建 close_prices
        close_prices = {}
        for sym in all_data:
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
            if len(dt) > 0:
                close_prices[sym] = float(dt["close"].iloc[-1])

        # 执行
        report = executor.execute_orders(
            buy_list=decision.get("buy", []),
            sell_list=decision.get("sell", []),
            today=today,
            all_data=all_data,
            close_prices=close_prices,
        )

        # 收盘快照
        executor.snapshot(str(today.date()), close_prices)

        print(f"\n  执行报告:", flush=True)
        print(f"    买入: {len(report.buy_filled)}/{len(report.buy_signals)} 成交 ({report.fill_rate_buy:.0%})", flush=True)
        print(f"    卖出: {len(report.sell_filled)}/{len(report.sell_signals)} 成交 ({report.fill_rate_sell:.0%})", flush=True)
        print(f"    手续费: {report.total_commission:.2f}", flush=True)
        print(f"    权益: {report.equity_before:,.0f} -> {report.equity_after:,.0f}", flush=True)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "signal_date": str(today.date()),
            "mode": "live",
            "buy": decision.get("buy", []),
            "sell": decision.get("sell", []),
            "hold": decision.get("hold", []),
            "n_factors": len(factors),
            "verdict": factor_config.get("verdict", "UNKNOWN"),
            "circuit_breaker": cb_state,
            "data_as_of": _data_as_of,
            "execution": {
                "buy_filled": len(report.buy_filled),
                "buy_rejected": len(report.buy_rejected),
                "sell_filled": len(report.sell_filled),
                "sell_rejected": len(report.sell_rejected),
                "total_commission": report.total_commission,
                "equity_before": report.equity_before,
                "equity_after": report.equity_after,
            },
        }

    # ── 8. 落盘 ──
    os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)
    with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  信号已写入: {SIGNAL_LOG}", flush=True)

    return entry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper Trading v3 — P5全因子组合")
    parser.add_argument("date", nargs="?", default=None, help="交易日期 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="仅生成信号, 不执行")
    args = parser.parse_args()
    generate_signal_v3(args.date, dry_run=args.dry_run)
