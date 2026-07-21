"""
ML增强版纸面交易 — LightGBM排序替代固定权重

用法: python paper_trade_ml.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
import storage
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from alt_data import peer_relative_factor
from ml_ranker import MLRanker

# 10只A股
SYMBOLS = ["688981","002371","603986","002049","300033","002230",
           "300750","002594","600519","600036"]
STOCK_NAMES = {"688981":"中芯","002371":"北华创","603986":"兆易","002049":"紫光",
    "300033":"同花顺","002230":"讯飞","300750":"宁德","002594":"比亚迪",
    "600519":"茅台","600036":"招行"}
MARKET, START, END = "a", "2024-01-01", "2026-07-10"
TOP_K, INITIAL_CAPITAL = 4, 100_000


def train_ml_ranker(all_data: dict, scorer: FactorScorer, days: list):
    """用前60天数据训练LightGBM。"""
    print("  [训练] LightGBM排序模型...")
    X_list, y_list, groups_list = [], [], []
    group_id = 0

    for today in days[:60]:  # 前60天
        stock_data = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_today = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today

        if len(stock_data) < 5: continue
        try:
            # 截面评分得到每个股票的因子值和未来收益
            scores = scorer.cross_sectional_score(stock_data)
            # 收集因子值(简化:直接用scores作为输入)
            features = []
            targets = []
            syms = []
            for sym in stock_data:
                close = stock_data[sym]["close"].values
                if len(close) < 6: continue
                # 5日未来收益作为标签
                fwd_ret = close[-1] / close[-6] - 1 if len(close) >= 6 else 0
                # 用因子评分作为特征(简化版)
                f = scores.get(sym, 0)
                features.append([f])
                targets.append(fwd_ret)
                syms.append(sym)
            if len(features) >= 5:
                X_list.extend(features)
                y_list.extend(targets)
                groups_list.extend([group_id] * len(features))
                group_id += 1
        except: pass

    if len(X_list) < 50:
        print(f"  训练数据不足({len(X_list)}),用规则引擎")
        return None

    ranker = MLRanker(n_estimators=100, max_depth=4, learning_rate=0.1)
    ranker.fit(np.array(X_list), np.array(y_list), np.array(groups_list))
    print(f"  LightGBM训练完成 ({len(X_list)}样本)")
    return ranker


def main():
    storage.init_db()
    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL_CAPITAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=1, hold_thresh=10)
    scorer = FactorScorer.from_preset("trend_momentum")
    macro = MacroOverlay(market=MARKET)
    macro.update()

    print(f"  📊 A股 Top-{TOP_K} — LightGBM排序")

    # 拉数据
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

    # 训练ML排序器
    ml_ranker = train_ml_ranker(all_data, scorer, trading_days)

    # 每日循环
    for day_idx, today in enumerate(trading_days):
        stock_data, close_prices = {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_today = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_today) < 50: continue
            stock_data[sym] = df_today
            close_prices[sym] = df_today["close"].iloc[-1]
        if len(stock_data) < TOP_K: continue

        # 评分: 因子(base) + ML调整
        scores = scorer.cross_sectional_score(stock_data)

        if ml_ranker and day_idx > 60:
            # ML增强: 用LightGBM预测替代原始分数
            syms = list(scores.keys())
            X_day = np.array([[scores[s]] for s in syms])
            ml_scores = ml_ranker.predict(X_day)
            for i, s in enumerate(syms):
                scores[s] = scores[s] * 0.5 + ml_scores[i] * 0.5  # 混合

        peer = peer_relative_factor(SYMBOLS, stock_data)
        for sym in scores:
            scores[sym] = scores[sym] * 0.7 + peer.get(sym, 0) * 0.3

        today_macro = macro.score_at(today)
        for sym in scores:
            scores[sym] *= (1 + today_macro * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)

        # 卖出
        for sym in decision["sell"]:
            pos = state.positions.get(sym, {})
            qty = pos.get("qty", 0)
            if qty > 0 and sym in close_prices:
                pm.apply_sell(sym, qty, close_prices[sym],
                              commission=qty*close_prices[sym]*cfg.get("sell_commission",0.0008))

        # 买入
        to_buy = decision["buy"]
        if to_buy:
            state = pm.load()
            cash_per = state.cash * 0.9 / len(to_buy)
            for sym in to_buy:
                if sym in close_prices:
                    px = close_prices[sym]
                    qty = int(cash_per/px/cfg["lot_size"])*cfg["lot_size"]
                    if qty >= cfg["lot_size"]:
                        comm = qty*px*cfg.get("buy_commission",0.0003)
                        if pm.can_buy(sym, qty, px, comm)[0]:
                            pm.apply_buy(sym, qty, px, commission=comm)

        pm.snapshot(today.strftime("%Y-%m-%d"), close_prices)

        if day_idx % 30 == 0:
            summary = pm.get_summary(close_prices)
            top_n = decision["ranked"][:TOP_K]
            ts = " > ".join(f"{STOCK_NAMES.get(s,s)}({scores[s]:+.2f})" for s in top_n)
            print(f"  {today.strftime('%Y-%m-%d')}: 权益={summary['total_equity']:,.0f} Top{TOP_K}=[{ts}]")

    # 结果
    summary = pm.get_summary(close_prices)
    final_equity = summary["total_equity"]
    total_return = (final_equity/INITIAL_CAPITAL-1)*100
    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            df = all_data[sym]
            bdf = df[(df["date"]>=start_dt)&(df["date"]<=end_dt)]
            if len(bdf)>0: bench_rets.append(bdf["close"].iloc[-1]/bdf["close"].iloc[0]-1)
    bench_avg = np.mean(bench_rets)*100

    print(f"\n  LightGBM排序结果: 策略{total_return:+.1f}% 基准{bench_avg:+.1f}% 超额{total_return-bench_avg:+.1f}%")


if __name__ == "__main__":
    main()
