"""
快速策略验证 — 参数网格扫描

用法:
  python quick_validate.py → 扫描top_k+hold_thresh → 输出最优组合
"""

import os, sys, itertools
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
import storage
from portfolio_ranker import PortfolioRanker
from paper_trade_portfolio import SYMBOLS, STOCK_NAMES, MARKET, START, END, TOP_K, INITIAL_CAPITAL
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from macro_overlay import MacroOverlay


def quick_validate(param_grid: dict = None):
    """快速扫描参数组合。"""
    if param_grid is None:
        param_grid = {
            "top_k": [2, 3, 4],
            "hold_thresh": [5, 10, 15],
        }

    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        try:
            df = fetcher.fetch(sym, "20180101", END.replace("-",""), "qfq", market=MARKET)
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except: pass

    start_dt, end_dt = pd.Timestamp(START), pd.Timestamp(END)
    sample = list(all_data.values())[0]
    trading_days = sample[(sample["date"] >= start_dt) & (sample["date"] <= end_dt)]["date"].tolist()

    # 对所有交易日计算截面分数 (只算一次)
    print("计算截面分数...")
    scorer = FactorScorer.from_preset("trend_momentum")
    all_scores = {}
    for i, today in enumerate(trading_days[:60]):  # 只跑前60天快速验证
        stock_data = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_today = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today
        if len(stock_data) < 2: continue
        try:
            scores = scorer.cross_sectional_score(stock_data)
            all_scores[today] = scores
        except: pass

    print(f"计算完成: {len(all_scores)} 天")

    # 扫描参数
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        db_path = os.path.join(os.path.dirname(__file__), "quant.db")
        if os.path.exists(db_path): os.remove(db_path)
        storage.init_db()

        pm = PortfolioManager(market=MARKET)
        ranker = PortfolioRanker(**params)

        macro = MacroOverlay(market=MARKET)
        macro.update()

        cfg = MARKET_CONFIG[MARKET]

        for today in trading_days[:60]:
            if today not in all_scores: continue
            scores = all_scores[today]
            state = pm.load()
            holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
            decision = ranker.rank(scores, holdings)

            for sym in decision["sell"]:
                pos = state.positions.get(sym, {})
                qty = pos.get("qty", 0)
                if qty > 0:
                    pm.apply_sell(sym, qty, df_today["close"].iloc[-1] if sym in stock_data else 0,
                                  commission=0)

            to_buy = decision["buy"]
            if to_buy:
                state = pm.load()
                cash_per = state.cash * 0.9 / len(to_buy)
                for sym in to_buy:
                    px = stock_data[sym]["close"].iloc[-1] if sym in stock_data else 0
                    if px > 0:
                        qty = int(cash_per / px / 200) * 200
                        if qty >= 200:
                            pm.apply_buy(sym, qty, px, commission=0)

        summary = pm.get_summary({s: stock_data[s]["close"].iloc[-1]
                                  for s in SYMBOLS if s in stock_data})
        final_equity = summary["total_equity"]
        results.append({**params, "equity": final_equity,
                       "return": (final_equity / INITIAL_CAPITAL - 1) * 100})

    # 输出最优
    df_res = pd.DataFrame(results).sort_values("return", ascending=False)
    print(f"\n{'='*60}")
    print("  参数扫描结果 (前60天)")
    print(f"{'='*60}")
    print(df_res.head(10).to_string(index=False))
    print(f"\n  最优: top_k={df_res.iloc[0]['top_k']}, "
          f"hold_thresh={df_res.iloc[0]['hold_thresh']}, "
          f"收益={df_res.iloc[0]['return']:+.1f}%")


if __name__ == "__main__":
    quick_validate()
