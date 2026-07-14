"""
多股票组合纸面交易 — Top-K排名制

所有股票一起排名,Top-K持有,其余不持有。
替代单股票 paper_trade.py 的逐只循环。

用法:
  python paper_trade_portfolio.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from datetime import datetime
import storage

from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from alt_data import peer_relative_factor
from llm_weight_optimizer import LLMWeightOptimizer
from sector_analyzer import SectorAnalyzer


# 12只港股
SYMBOLS = [
    "01810", "00700", "09988", "09618", "03690", "09999",
    "02020", "02318", "01211", "00981", "02269", "09888",
]
STOCK_NAMES = {
    "01810":"小米", "00700":"腾讯", "09988":"阿里", "09618":"京东",
    "03690":"美团", "09999":"网易", "02020":"安踏", "02318":"平安",
    "01211":"比亚迪","00981":"中芯","02269":"药明","09888":"百度",
}
MARKET = "hk"
START, END = "2024-01-01", "2026-07-10"
TOP_K = 3
INITIAL_CAPITAL = 100_000
MAX_HOLD_DAYS = 30      # ★ 时间止损: 持有超30天不涨→强制卖出
MAX_DD_PCT = 0.15       # ★ 熔断: 从最高点回撤>15%→清仓


def main():
    storage.init_db()
    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL_CAPITAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=1, hold_thresh=10)
    scorer = FactorScorer.from_preset("trend_momentum")
    macro = MacroOverlay(market=MARKET)
    macro.update()

    print(f"\n{'='*60}")
    print(f"  📊 Top-{TOP_K} 排名制组合交易")
    print(f"  {len(SYMBOLS)}只港股 → 始终持有最强的{TOP_K}只")
    print(f"{'='*60}")

    # 1. 拉取所有股票的全量数据
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        print(f"  拉取 {sym} {STOCK_NAMES.get(sym,'')}...")
        try:
            df = fetcher.fetch(sym, "20180101", END.replace("-",""), "qfq", market=MARKET)
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except Exception as e:
            print(f"    ❌ {e}")

    # 2. 获取交易日列表
    start_dt = pd.Timestamp(START)
    end_dt = pd.Timestamp(END)
    sample_df = list(all_data.values())[0]
    trading_days = sample_df[(sample_df["date"] >= start_dt) & (sample_df["date"] <= end_dt)]["date"].tolist()

    print(f"\n  交易日: {len(trading_days)}, 股票: {len(all_data)}只")

    # 2.5. LLM定制每只股票的因子权重
    print("  LLM定制权重...")
    optimizer = LLMWeightOptimizer(backend="mock")
    sector_aly = SectorAnalyzer(market=MARKET)
    per_stock_scorers = {}
    for sym in SYMBOLS:
        if sym not in all_data: continue
        df = all_data[sym]
        ret = df["close"].pct_change().dropna()
        sector_info = sector_aly.get_sector(sym)
        features = {
            "volatility": float(ret.std() * np.sqrt(252)) if len(ret) > 0 else 0.3,
            "trend_adx": 20, "daily_range": float(((df["high"]-df["low"])/df["close"]).mean()),
            "sector": sector_info["sector"],
        }
        opt = optimizer.optimize(sym, features)
        per_stock_scorers[sym] = FactorScorer(
            factor_weights=opt["factor_weights"],
            buy_threshold=opt.get("buy_threshold", 0.15),
            sell_threshold=opt.get("sell_threshold", -0.10),
        )
    print(f"  为{len(per_stock_scorers)}只股票生成了专属权重")

    # 3. 每日循环
    daily_equity = []
    trade_count = 0
    peak_equity = INITIAL_CAPITAL  # 熔断用
    hold_start: dict = {}  # sym → 买入日期

    for day_idx, today in enumerate(trading_days):
        today_str = today.strftime("%Y-%m-%d")

        # 收集今日所有股票的最近数据
        stock_data = {}
        close_prices = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_full = all_data[sym]
            df_today = df_full[df_full["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today
            close_prices[sym] = df_today["close"].iloc[-1]

        if len(stock_data) < TOP_K:
            continue

        # 截面评分 + 同伴相对强度
        try:
            scores = scorer.cross_sectional_score(stock_data)
            peer_scores = peer_relative_factor(SYMBOLS, stock_data)
            for sym in scores:
                scores[sym] = scores[sym] * 0.7 + peer_scores.get(sym, 0) * 0.3
        except Exception as e:
            if day_idx == 0: print(f"  评分异常: {e}")
            continue

        # 宏观叠加
        today_macro = macro.score_at(today)
        for sym in scores:
            scores[sym] *= (1 + today_macro * 0.3)

        # 当前持仓
        state = pm.load()
        holdings = [sym for sym, pos in state.positions.items() if pos["qty"] > 0]

        # 排名决策
        decision = ranker.rank(scores, holdings)

        # 执行卖出 (含风险控制)
        for sym in list(decision["sell"]):
            pos = state.positions.get(sym, {})
            qty = pos.get("qty", 0)
            if qty > 0 and sym in close_prices:
                px = close_prices[sym]
                comm = qty * px * cfg["sell_commission"]
                try:
                    pm.apply_sell(sym, qty, px, commission=comm)
                    trade_count += 1
                    hold_start.pop(sym, None)
                except: pass

        # ★ 时间止损: 持有超30天强制卖出
        for sym in list(hold_start.keys()):
            held_days = (today - hold_start[sym]).days
            if held_days > MAX_HOLD_DAYS:
                pos = state.positions.get(sym, {})
                qty = pos.get("qty", 0)
                if qty > 0 and sym in close_prices:
                    px = close_prices[sym]
                    pm.apply_sell(sym, qty, px, commission=qty*px*cfg["sell_commission"])
                    trade_count += 1
                    hold_start.pop(sym, None)

        # ★ 熔断检查: 权益从高点回撤>15%→清仓
        pm.snapshot(today_str, close_prices)
        summary = pm.get_summary(close_prices)
        current_equity = summary["total_equity"]
        peak_equity = max(peak_equity, current_equity)
        dd_pct = (current_equity - peak_equity) / peak_equity

        if dd_pct < -MAX_DD_PCT:
            print(f"  ⚠️ {today_str}: 熔断! 回撤{dd_pct*100:.1f}% > {MAX_DD_PCT*100:.0f}%, 清仓")
            state = pm.load()  # reload fresh state
            for sym in list(state.positions.keys()):
                pos = state.positions.get(sym, {})
                qty = pos.get("qty", 0)
                if qty > 0 and sym in close_prices:
                    px = close_prices[sym]
                    pm.apply_sell(sym, qty, px, commission=qty*px*cfg["sell_commission"])
            peak_equity = current_equity
            continue

        # 执行买入 (等权重)
        to_buy = decision["buy"]
        if to_buy:
            state = pm.load()
            cash_per_stock = state.cash * 0.9 / len(to_buy)
            for sym in to_buy:
                if sym in close_prices:
                    px = close_prices[sym]
                    qty = int(cash_per_stock / px / cfg["lot_size"]) * cfg["lot_size"]
                    if qty >= cfg["lot_size"]:
                        comm = qty * px * cfg["buy_commission"]
                        if pm.can_buy(sym, qty, px, comm)[0]:
                                pm.apply_buy(sym, qty, px, commission=comm)
                                trade_count += 1
                                hold_start[sym] = today  # ★ 记录买入日

        # 每日快照
        pm.snapshot(today_str, close_prices)
        summary = pm.get_summary(close_prices)

        if day_idx % 30 == 0:
            top3 = decision["ranked"][:3]
            top3_str = " > ".join(f"{STOCK_NAMES.get(s,s)}({scores[s]:+.2f})" for s in top3)
            print(f"  {today_str}: 权益={summary['total_equity']:,.0f} "
                  f"持仓={len(holdings)} Top3=[{top3_str}]")

        daily_equity.append({
            "date": today_str,
            "equity": summary["total_equity"],
        })

    # 4. 结果统计
    final_equity = daily_equity[-1]["equity"] if daily_equity else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    # 等权基准
    bench_returns = []
    for sym in SYMBOLS:
        if sym in all_data:
            df = all_data[sym]
            bench_df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            if len(bench_df) > 0:
                ret = bench_df["close"].iloc[-1] / bench_df["close"].iloc[0] - 1
                bench_returns.append(ret)
    bench_avg = np.mean(bench_returns) * 100 if bench_returns else 0

    print(f"\n{'='*60}")
    print(f"  📊 Top-{TOP_K} 排名制结果")
    print(f"{'='*60}")
    print(f"  策略收益: {total_return:+.2f}%")
    print(f"  等权基准: {bench_avg:+.2f}%")
    print(f"  超额收益: {total_return - bench_avg:+.2f}%")
    print(f"  交易次数: {trade_count}")
    print(f"  最终权益: {cfg['currency']} {final_equity:,.0f}")


if __name__ == "__main__":
    main()
