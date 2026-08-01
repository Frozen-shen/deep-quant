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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                                  factor_cache, today) -> Dict[str, float]:
    """
    IC加权线性组合 (与回测逻辑一致):
      composite_i = sum(ICIR_j * z_score(factor_j_i)) / sum(|ICIR_j|)
    """
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f.get("weight_multiplier", 1.0) for f in factors])
    abs_weight_sum = np.sum(np.abs(weights))
    if abs_weight_sum < 1e-9:
        return {}

    # 收集当日因子值
    raw_values = {name: {} for name in factor_names}
    for sym in all_data:
        feats = factor_cache.get(sym, today)
        if feats is None:
            continue
        for name in factor_names:
            if name in feats and not np.isnan(feats[name]):
                raw_values[name][sym] = feats[name]

    # 找所有因子都有值的股票
    valid_syms = None
    for name in factor_names:
        syms_set = set(raw_values[name].keys())
        if valid_syms is None:
            valid_syms = syms_set
        else:
            valid_syms = valid_syms & syms_set

    if valid_syms is None or len(valid_syms) < 10:
        return {}

    valid_syms = sorted(valid_syms)
    n = len(valid_syms)

    # 截面 z-score + IC加权
    composite = np.zeros(n)
    for fi, name in enumerate(factor_names):
        vals = np.array([raw_values[name][s] for s in valid_syms])
        mean = vals.mean()
        std = vals.std()
        if std < 1e-9:
            continue
        z = (vals - mean) / std
        composite += weights[fi] * z

    composite /= abs_weight_sum
    return dict(zip(valid_syms, composite.tolist()))


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

    # ── 1. 加载因子配置 ──
    factor_config = load_factor_config()
    if factor_config is None:
        print("  [WARN] p5_portfolio_report.json 不存在, 使用回退配置", flush=True)
        factor_config = build_fallback_config()

    factors = factor_config["factors"]
    if not factors:
        print("  [ERROR] 无可用因子, 终止", flush=True)
        return None

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

    # ── 3. 加载数据 + 预计算因子 ──
    print("  加载数据...", flush=True)
    from data_cache import load_all
    syms = get_cached_symbols()
    all_data = load_all(syms)
    all_data = {s: df for s, df in all_data.items() if df is not None and len(df) >= 100}
    print(f"  有效数据: {len(all_data)} 只", flush=True)

    print("  预计算因子...", flush=True)
    from factor_scorer import FactorScorer
    from factor_cache import FactorCache

    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())
    factor_cache = FactorCache(scorer, factor_names)
    factor_cache.precompute(all_data)
    print(f"  因子预计算完成: {len(factor_names)} 个", flush=True)

    # 过滤可计算因子
    computable = set(factor_names)
    factors = [f for f in factors if f["name"] in computable]
    if not factors:
        print("  [ERROR] 无可计算因子", flush=True)
        return None

    # ── 4. 计算复合评分 ──
    print("  计算复合评分...", flush=True)
    scores = compute_composite_scores_live(factors, all_data, factor_cache, today)

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
        }
    else:
        # 分钟数据预拉取
        if config["execution"].get("minute_mode", False):
            try:
                from data.minute_fetcher import MinuteFetcher
                mf = MinuteFetcher()
                trade_symbols = list(set(
                    decision.get('buy', []) +
                    decision.get('sell', []) +
                    holdings
                ))
                print(f"  预拉取分钟数据: {len(trade_symbols)}只...", flush=True)
                mf.fetch_batch(trade_symbols, days=5)
            except Exception as e:
                print(f"  [WARN] 分钟数据预拉取失败: {e}", flush=True)

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
