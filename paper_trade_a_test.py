"""
A股 Top-K 排名制纸面交易 — 20只多板块

用法: python paper_trade_a.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import storage

from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from alt_data import peer_relative_factor
from intraday_executor import IntradayExecutor
from sector_analyzer import SectorAnalyzer

# 20只A股 (科技7 + 新能源4 + 消费3 + 医药3 + 金融2 + 军工1)
SYMBOLS = [
    "688981", "002371", "603986", "002049", "300033", "002230",  # 科技6
    "300750", "002594",  # 新能源2
    "600519", "600036",  # 防御2
]
STOCK_NAMES = {
    "688981":"中芯", "002371":"北华创", "603986":"兆易", "002049":"紫光",
    "300033":"同花顺", "002230":"讯飞",
    "300750":"宁德", "002594":"比亚迪",
    "600519":"茅台", "600036":"招行",
}
# A股板块映射
A_SECTORS = {
    "688981": ("半导体", ["688981","002371","603986","002049"]),
    "002371": ("半导体", ["002371","688981","603986"]),
    "603986": ("半导体", ["603986","002371","688981"]),
    "002049": ("半导体", ["002049","603986","688981"]),
    "688111": ("软件", ["688111","300033","002230"]),
    "300033": ("软件", ["300033","688111","002230"]),
    "002230": ("AI", ["002230","300033","688111"]),
    "300750": ("新能源", ["300750","002594","601012","300274"]),
    "002594": ("新能源", ["002594","300750","601012"]),
    "601012": ("新能源", ["601012","300750","300274"]),
    "300274": ("新能源", ["300274","300750","601012"]),
    "600519": ("白酒", ["600519","000858"]),
    "000858": ("白酒", ["000858","600519"]),
    "002714": ("养殖", ["002714"]),
    "300760": ("医药", ["300760","600276","300122"]),
    "600276": ("医药", ["600276","300760"]),
    "300122": ("医药", ["300122","300760"]),
    "601318": ("金融", ["601318","600036"]),
    "600036": ("金融", ["600036","601318"]),
    "600760": ("军工", ["600760"]),
}

MARKET = "a"
START, END = "2024-01-01", "2026-07-10"
TOP_K = 4
INITIAL_CAPITAL = 100_000
MAX_HOLD_DAYS = 30
MAX_DD_PCT = 0.15

MARKET = "a"
START, END = "2024-01-01", "2026-07-10"
TOP_K = 4
INITIAL_CAPITAL = 100_000
MAX_HOLD_DAYS = 30
MAX_DD_PCT = 0.15


def main():
    storage.init_db()
    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL_CAPITAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=1, hold_thresh=10,
                             sector_neutral=True)
    scorer = FactorScorer.from_preset("ic_optimized")
    macro = MacroOverlay(market=MARKET)
    iexec = IntradayExecutor()  # 日内执行器
    macro.update()

    print(f"\n{'='*60}")
    print(f"  📊 A股 Top-{TOP_K} 排名制 — {len(SYMBOLS)}只多板块")
    print(f"{'='*60}")

    # 拉数据
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        try:
            df = fetcher.fetch(sym, "20180101", END.replace("-",""), "qfq", market=MARKET)
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except Exception as e:
            print(f"  {sym} ❌ {e}")

    start_dt, end_dt = pd.Timestamp(START), pd.Timestamp(END)
    sample = list(all_data.values())[0]
    trading_days = sample[(sample["date"] >= start_dt) & (sample["date"] <= end_dt)]["date"].tolist()
    print(f"  交易日: {len(trading_days)}, 股票: {len(all_data)}只")

    peak_equity = INITIAL_CAPITAL
    hold_start = {}

    for day_idx, today in enumerate(trading_days):
        today_str = today.strftime("%Y-%m-%d")
        stock_data, close_prices = {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_today = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today
            close_prices[sym] = df_today["close"].iloc[-1]

        if len(stock_data) < TOP_K: continue

        # 截面评分
        try:
            scores = scorer.cross_sectional_score(stock_data)
            peer = peer_relative_factor(SYMBOLS, stock_data)
            for sym in scores:
                scores[sym] = scores[sym] * 0.7 + peer.get(sym, 0) * 0.3
            # 日内因子奖励 (最后60天)
            if (end_dt - today).days < 60:
                for sym in scores:
                    if sym in stock_data:
                        try:
                            intra = iexec.fetcher.fetch_intraday_factors(sym, today_str)
                            if intra:
                                scores[sym] += (intra.get("intraday_trend",0)*10
                                              - abs(intra.get("vwap_deviation",0))*5
                                              + intra.get("tail_return",0)*8) * 0.1
                        except: pass
        except: continue

        # 宏观叠加
        today_macro = macro.score_at(today)
        for sym in scores:
            scores[sym] *= (1 + today_macro * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]

        # 带A股板块信息的排名
        # 提取板块名 (A_SECTORS格式: {sym: (name, peers)})
        sector_map = {s: v[0] for s, v in A_SECTORS.items()} if 'A_SECTORS' in dir() else {}
        decision = ranker.rank(scores, holdings, sectors=sector_map)

        # 卖出
        for sym in list(decision["sell"]):
            pos = state.positions.get(sym, {})
            qty = pos.get("qty", 0)
            if qty > 0 and sym in close_prices:
                pm.apply_sell(sym, qty, close_prices[sym],
                              trade_date=today_str, commission=qty*close_prices[sym]*cfg.get("sell_commission", 0.0008))
                hold_start.pop(sym, None)

        # 时间止损
        for sym in list(hold_start.keys()):
            if (today - hold_start[sym]).days > MAX_HOLD_DAYS:
                pos = state.positions.get(sym, {})
                qty = pos.get("qty", 0)
                if qty > 0 and sym in close_prices:
                    pm.apply_sell(sym, qty, close_prices[sym],
                                  trade_date=today_str, commission=qty*close_prices[sym]*cfg.get("sell_commission", 0.0008))
                    hold_start.pop(sym, None)

        # 买入
        to_buy = decision["buy"]
        if to_buy:
            state = pm.load()
            cash_per = state.cash * 0.9 / len(to_buy)
            for sym in to_buy:
                if sym in close_prices:
                    # 日内VWAP执行 (最后60天)
                    if (end_dt - today).days < 60:
                        iexec_px = iexec.get_best_execution_price(sym, today_str, "BUY")
                        px = iexec_px if iexec_px else close_prices[sym]
                    else:
                        px = close_prices[sym]
                    px = close_prices[sym]
                    qty = int(cash_per / px / cfg["lot_size"]) * cfg["lot_size"]
                    if qty >= cfg["lot_size"]:
                        comm = qty * px * cfg.get("buy_commission", 0.0003)
                        if pm.can_buy(sym, qty, px, comm)[0]:
                            pm.apply_buy(sym, qty, px, trade_date=today_str, commission=comm)
                            hold_start[sym] = today

        pm.snapshot(today_str, close_prices)
        summary = pm.get_summary(close_prices)
        current_equity = summary["total_equity"]
        peak_equity = max(peak_equity, current_equity)
        dd_pct = (current_equity - peak_equity) / peak_equity

        if dd_pct < -MAX_DD_PCT:
            print(f"  ⚠️ {today_str}: 熔断! 回撤{dd_pct*100:.1f}%")
            state = pm.load()
            for sym in list(state.positions.keys()):
                pos = state.positions.get(sym, {})
                qty = pos.get("qty", 0)
                if qty > 0 and sym in close_prices:
                    pm.apply_sell(sym, qty, close_prices[sym],
                                  trade_date=today_str, commission=qty*close_prices[sym]*cfg.get("sell_commission", 0.0008))
            peak_equity = current_equity
            hold_start.clear()
            continue

        if day_idx % 30 == 0:
            top_n = decision["ranked"][:TOP_K]
            ts = " > ".join(f"{STOCK_NAMES.get(s,s)}({scores[s]:+.2f})" for s in top_n)
            print(f"  {today_str}: 权益={current_equity:,.0f} "
                  f"持仓={len(holdings)} Top{TOP_K}=[{ts}]")

    # 结果
    final_equity = summary["total_equity"] if 'summary' in dir() else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            df = all_data[sym]
            bdf = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            if len(bdf) > 0:
                bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100

    print(f"\n{'='*60}")
    print(f"  📊 A股 Top-{TOP_K} 排名制结果")
    print(f"{'='*60}")
    print(f"  策略收益: {total_return:+.2f}%")
    print(f"  等权基准: {bench_avg:+.2f}%")
    print(f"  超额收益: {total_return - bench_avg:+.2f}%")


if __name__ == "__main__":
    main()
