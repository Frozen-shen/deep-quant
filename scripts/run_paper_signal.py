"""
Paper Trading 日频信号生成器 v2 — 持久化版

每个交易日收盘后运行:
  python scripts/run_paper_signal.py
  python scripts/run_paper_signal.py --date 2026-08-03

v2 改进:
  - 接入 PaperExecutor 实现持久持仓追踪
  - 信号生成后自动执行下单
  - 收盘快照记录权益
  - 同时输出 paper_signals.jsonl + paper_executions.jsonl
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SIGNAL_LOG = os.path.join(BASE_DIR, "data", "paper_signals.jsonl")


def generate_signal(date_str: str = None):
    """生成当天交易信号并执行。

    Args:
      date_str: '2026-07-31' 或 None=最新交易日
    """
    from data_cache import get_cached_symbols, load_all

    if date_str is None:
        # 取数据缓存中最新的交易日
        syms = get_cached_symbols()
        latest_date = None
        for sym in syms[:5]:
            from data_cache import load as load_single
            df = load_single(sym)
            if df is not None and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"])
                last = df["date"].max()
                if latest_date is None or last > latest_date:
                    latest_date = last
        today = latest_date or pd.Timestamp.now().normalize()
        print(f"[PaperTrading] 最新数据日期: {today.date()}")
    else:
        today = pd.Timestamp(date_str)

    # 交易日检查
    from scheduler import is_trading_day
    if not is_trading_day(today):
        print(f"[PaperTrading] {today.date()} 非交易日, 跳过信号生成")
        return None

    print(f"[PaperTrading] 生成 {today.date()} 信号...")

    from model.pipeline import load_config, QuantPipeline

    config = load_config()

    # ── 1. 加载数据 + 预计算因子 ──
    pipeline = QuantPipeline(config, mode="blind")
    pipeline._load_universe()
    pipeline._load_data()
    pipeline._precompute_factors()

    # ── 2. 训练模型 (用最近N天数据) ──
    train_end = today - timedelta(days=1)
    train_start = train_end - pd.DateOffset(months=config["rolling"]["train_months"])

    w = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": today,
        "test_end": today,
    }

    xt, yt, gt = [], [], []
    all_dates = sorted(set().union(*[set(df["date"].tolist())
                                     for df in pipeline._all_data.values()]))
    train_days = [d for d in all_dates
                  if train_start <= d <= train_end - timedelta(
                      days=config["rolling"]["embargo_days"])]

    for d in train_days[::2]:
        fn, labels, syms = pipeline._build_cs(d)
        if fn is not None:
            xt.extend(fn.tolist())
            yt.extend(labels.tolist())
            gt.extend([str(d)] * len(labels))

    if len(xt) < 100:
        print("  ❌ 训练数据不足")
        return None

    model = pipeline._train_model(np.array(xt), np.array(yt, dtype=int),
                                  gt, train_end)

    # ── 3. 生成原始模型预测分数 ──
    from execution.paper_executor import PaperExecutor
    from portfolio_ranker import PortfolioRanker
    from trading_rules import TradingRules

    executor = PaperExecutor(
        initial_capital=config["execution"]["initial_capital"],
        top_k=pipeline.top_k,
        lot_size=pipeline.lot_size,
        slippage_bps=config["execution"].get("slippage_bps", 30),
        turnover_limit_pct=config["execution"].get("turnover_limit_pct", 0.5),
        max_single_pct=config["execution"].get("max_single_pct", 0.25),
        minute_mode=config["execution"].get("minute_mode", False),
        execution_algo=config["execution"].get("execution_algo", "vwap"),
        twap_slices=config["execution"].get("twap_slices", 8),
    )

    state = executor.load_state()
    print(f"  当前持仓: {len(state.positions)} 只, 现金: {state.cash:,.0f}")

    ranker = PortfolioRanker(
        top_k=pipeline.top_k, n_drop=pipeline.n_drop,
        hold_thresh=pipeline.hold_thresh,
        sell_rank_buffer=pipeline.sell_rank_buffer,
        buy_confirm_days=pipeline.buy_confirm_days,
        cost_threshold=pipeline.cost_threshold,
    )
    rules = TradingRules()

    # 同步 ranker 的持有天数状态
    for sym in state.positions:
        ranker._hold_since[sym] = (today - pd.Timestamp(state.last_date)).days if state.last_date else 1

    # ── 3a. 获取原始模型分数 ──
    sd, cpt = {}, {}
    for sym in pipeline._all_data:
        dt = pipeline._all_data[sym][pipeline._all_data[sym]["date"] <= today].tail(120)
        if len(dt) >= 60:
            sd[sym] = dt
            cpt[sym] = dt["close"].iloc[-1]

    if len(sd) < pipeline.top_k:
        print("  ❌ 可交易股票不足")
        return None

    sd, cpt = rules.filter_tradeable(sd, cpt)
    if len(sd) < pipeline.top_k:
        print("  ❌ 过滤后可交易股票不足")
        return None

    sym_feats, swd = [], []
    for sym in sd:
        feats = pipeline._factor_cache.get_features(sym, today)
        if feats is not None:
            if pipeline._fund_cache:
                from data.fundamental_cache_builder import merge_fundamental_to_features
                feats = merge_fundamental_to_features(sym, today, pipeline._fund_cache, feats)
            sym_feats.append(feats)
            swd.append(sym)

    if len(sym_feats) < pipeline.top_k:
        print("  ❌ 有效特征不足")
        return None

    fa = np.array(sym_feats)
    m, s = fa.mean(axis=0), fa.std(axis=0)
    s[s == 0] = 1.0
    fn = (fa - m) / s
    preds = model.predict(fn)
    scores = {swd[i]: float(preds[i]) for i in range(len(swd))}

    # ── 3b. PEAD 事件因子叠加 (统一接口) ──
    pead_weight = config["factors"].get("pead_weight", 0.0)
    if pead_weight > 0:
        try:
            from factors.pead_factor import PEADFactor
            pead = PEADFactor()
            surprise_scores = pead.compute_surprise_scores(str(today.date()))
            if surprise_scores:
                scores = pead.enhance_scores(scores, surprise_scores, weight=pead_weight)
                print(f"  📡 PEAD增强: {len(surprise_scores)}只有效事件, weight={pead_weight}")
        except Exception as e:
            print(f"  ⚠️ PEAD增强跳过: {e}")

    # ── 3c. PEAD 防御: 负面预告强制卖出 ──
    if config["execution"].get("pead_defense", False):
        try:
            from factors.pead_factor import PEADFactor
            pead_def = PEADFactor()
            negative_events = pead_def.compute_surprise_scores(str(today.date()))
            force_sell = [sym for sym, s in negative_events.items()
                         if s < -0.5 and sym in state.positions]
            if force_sell:
                print(f"  🛡️ PEAD防御: 强制卖出 {force_sell}")
                # 直接加入卖出列表 (不走 ranker)
        except Exception:
            pass

    # ── 3d. 排名决策 ──
    holdings = list(state.positions.keys())
    decision = ranker.rank(scores, holdings)

    # 涨跌停过滤
    limit_data = dict(sd)
    unadj_data = getattr(pipeline, '_unadj_data', {})
    if unadj_data:
        limit_data.update({s: unadj_data[s] for s in unadj_data if s in sd})
    decision["buy"] = [s for s in decision["buy"] if s in limit_data and rules.can_buy(s, limit_data[s])]
    decision["sell"] = [s for s in decision["sell"] if s in limit_data and rules.can_sell(s, limit_data[s])]

    # PEAD defense: 强制卖出负面事件持仓
    if config["execution"].get("pead_defense", False):
        try:
            for sym in force_sell:
                if sym not in decision["sell"] and sym in state.positions:
                    decision["sell"].append(sym)
        except NameError:
            pass

    decision["top_k_scores"] = dict(
        sorted(scores.items(), key=lambda x: -x[1])[:pipeline.top_k * 2]
    )
    signal = decision

    if signal is None or (not signal.get("buy") and not signal.get("sell")):
        print("  ⚠️ 今日无交易信号")
        # 即使无信号也做快照
        all_data = pipeline._all_data
        close_prices = {}
        today_dt = pd.Timestamp(today)
        for sym in all_data:
            dt = all_data[sym][all_data[sym]["date"] <= today_dt].tail(1)
            if len(dt) > 0:
                close_prices[sym] = float(dt["close"].iloc[-1])
        executor.snapshot(str(today.date()), close_prices)
        return {"signal_date": str(today.date()), "buy": [], "sell": [], "hold": []}

    # ── 4. 执行订单 ──
    print(f"  买入信号: {decision.get('buy', [])}")
    print(f"  卖出信号: {decision.get('sell', [])}")

    all_data = pipeline._all_data
    unadj_data = getattr(pipeline, '_unadj_data', {})
    today_dt = pd.Timestamp(today)

    # ── 4a. 分钟数据预拉取 (如果启用分钟模式) ──
    if config["execution"].get("minute_mode", False):
        try:
            from data.minute_fetcher import MinuteFetcher
            mf = MinuteFetcher()
            trade_symbols = list(set(
                decision.get('buy', []) +
                decision.get('sell', []) +
                list(state.positions.keys())
            ))
            print(f"  ⏱️  预拉取分钟数据: {len(trade_symbols)}只...")
            mf.fetch_batch(trade_symbols, days=5)
            print(f"  ✅ 分钟数据就绪")
        except Exception as e:
            print(f"  ⚠️ 分钟数据预拉取失败: {e} (将回退日线)")

    # 构建 close_prices
    close_prices = {}
    for sym in all_data:
        dt = all_data[sym][all_data[sym]["date"] <= today_dt].tail(1)
        if len(dt) > 0:
            close_prices[sym] = float(dt["close"].iloc[-1])

    report = executor.execute_orders(
        buy_list=signal.get("buy", []),
        sell_list=signal.get("sell", []),
        today=today_dt,
        all_data=all_data,
        unadjusted_data=unadj_data,
        close_prices=close_prices,
    )

    # ── 5. 收盘快照 ──
    executor.snapshot(str(today.date()), close_prices)

    # ── 6. 落盘信号记录 (append-only) ──
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "signal_date": str(today.date()),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "buy": signal.get("buy", []),
        "sell": signal.get("sell", []),
        "hold": signal.get("hold", []),
        "top_k_scores": dict(
            list((signal or {}).get("top_k_scores", {}).items())[:10]
        ) if signal else {},
        # v2 新增: 执行结果摘要
        "execution": {
            "buy_filled": len(report.buy_filled),
            "buy_rejected": len(report.buy_rejected),
            "sell_filled": len(report.sell_filled),
            "sell_rejected": len(report.sell_rejected),
            "fill_rate_buy": report.fill_rate_buy,
            "fill_rate_sell": report.fill_rate_sell,
            "total_commission": report.total_commission,
        },
    }

    os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)
    with open(SIGNAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  信号已写入 {SIGNAL_LOG}")
    print(f"  📈 执行报告: 买 {len(report.buy_filled)}/{len(report.buy_signals)} 成交 "
          f"({report.fill_rate_buy:.0%}), "
          f"卖 {len(report.sell_filled)}/{len(report.sell_signals)} 成交 "
          f"({report.fill_rate_sell:.0%})")
    print(f"  💰 手续费: ¥{report.total_commission:.2f}")
    print(f"  📊 权益: ¥{report.equity_before:,.0f} → ¥{report.equity_after:,.0f} "
          f"({(report.equity_after/report.equity_before - 1)*100:+.2f}%)")

    return entry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模拟盘信号生成 + 执行")
    parser.add_argument("date", nargs="?", default=None,
                       help="交易日期 YYYY-MM-DD")
    args = parser.parse_args()
    generate_signal(args.date)
