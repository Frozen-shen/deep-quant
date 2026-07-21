"""
诚实版: 25+因子 → LightGBM lambdarank → 训练/验证/测试分离

用法: python test_20_real.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from scipy import stats
import storage
from data_fetcher import MARKET_CONFIG
from data_cache import load_all
from factor_scorer import FactorScorer
from portfolio import PortfolioManager
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from alt_data import peer_relative_factor
from ml_ranker import MLRanker

SYMBOLS = ["688981","002371","603986","002049","300033","002230","300750","002594","600519","600036"]
MARKET, TOP_K, INITIAL = "a", 4, 100_000


def main():
    print("="*60)
    print("  诚实版: 25因子+LightGBM lambdarank")
    print("="*60)

    # 加载数据
    print("\n[加载] 数据缓存...")
    all_data = load_all(SYMBOLS)
    if not all_data:
        print("缓存为空,请先: python data_cache.py --fetch")
        return
    print(f"  {len(all_data)}只, {len(all_data[SYMBOLS[0]])}天")

    # IC分析(训练期)
    print("\n[IC] 训练期2020-2023...")
    scorer = FactorScorer.from_preset("ic_optimized")
    factor_names = sorted(scorer.factor_weights.keys())
    print(f"  因子数: {len(factor_names)}")

    # 构建训练面板
    print("\n[训练] LightGBM lambdarank...")
    X_list, y_list, group_list = [], [], []
    days = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
    train_days = [d for d in days if pd.Timestamp("2020-01-01")<=d<=pd.Timestamp("2023-12-31")]
    train_days = train_days[::3]

    for ti, today in enumerate(train_days):
        sd = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"]<=today].tail(120)
            if len(dt) < 60: continue
            sd[sym] = dt
        if len(sd) < 5: continue
        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if f is None or len(f) == 0: continue
            row = f.iloc[-1]; close = sd[sym]["close"].values
            if len(close) < 6: continue
            try:
                feats = [float(row.get(fn, 0)) for fn in factor_names]
            except: continue
            fwd = close[-1] / close[-6] - 1
            X_list.append(feats); y_list.append(fwd)
            group_list.append(today)
        if ti == 0:
            print(f"  第1天: sd={len(sd)}只, 收集{len(X_list)}条")

    if len(X_list)<100:
        print(f"  训练数据不足({len(X_list)}条)!")
        return

    X, y = np.array(X_list), np.array(y_list)
    groups = pd.Series(group_list).astype(str).factorize()[0]
    print(f"  训练集: {len(X)}条, 验证集: {int(len(X)*0.2)}条")

    model = MLRanker(n_estimators=200, max_depth=6, learning_rate=0.05)
    model.feature_names = factor_names
    model.fit(X, y, groups, val_ratio=0.2)

    # 验证+测试
    for period_name, start, end in [
        ("验证期(2024-2025H1)", "2024-01-01", "2025-06-30"),
        ("测试期(2025H2-2026H1)", "2025-07-01", "2026-07-10"),
    ]:
        print(f"\n[{period_name}]")
        run_backtest(model, scorer, factor_names, all_data, start, end, period_name)


def run_backtest(model, scorer, factor_names, all_data, start, end, label):
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()

    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=2, hold_thresh=10)
    macro = MacroOverlay(market=MARKET); macro.update()

    days = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
    days = [d for d in days if pd.Timestamp(start)<=d<=pd.Timestamp(end)]
    trade_count = 0

    for ti, today in enumerate(days):
        ts = today.strftime("%Y-%m-%d")
        sd, cp, scores = {}, {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"]<=today].tail(120)
            if len(dt)<60: continue; sd[sym]=dt; cp[sym]=dt["close"].iloc[-1]
        if len(sd)<TOP_K: continue

        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if len(f)==0: continue
            row = f.iloc[-1]
            feats = np.array([[float(row.get(fn,0)) for fn in factor_names]])
            try:
                scores[sym] = float(model.predict(feats)[0])
            except: scores[sym] = 0.0

        if ti == 0:
            print(f"  Day1: {len(sd)} stocks, {len(scores)} scores, spread={max(scores.values())-min(scores.values()):.4f}")

        if len(scores)<TOP_K: continue
        for s in scores: scores[s] *= (1+macro.score_at(today)*0.3)

        state = pm.load()
        holdings = [s for s,p in state.positions.items() if p["qty"]>0]
        decision = ranker.rank(scores, holdings)

        for s in decision["sell"]:
            pos = state.positions.get(s,{}); qty=pos.get("qty",0)
            if qty>0 and s in cp:
                pm.apply_sell(s,qty,cp[s],trade_date=ts,commission=qty*cp[s]*0.0008)
                trade_count+=1
        for s in decision["buy"]:
            if s in cp:
                state=pm.load(); cash_p=state.cash*0.9/max(1,len(decision["buy"]))
                px=cp[s]; qty=int(cash_p/px/100)*100
                if qty>=100:
                    pm.apply_buy(s,qty,px,trade_date=ts,commission=qty*px*0.0003)
                    trade_count+=1
        pm.snapshot(ts, cp)

    summary = pm.get_summary(cp)
    ret = (summary["total_equity"]/INITIAL-1)*100
    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            bdf = all_data[sym][(all_data[sym]["date"]>=pd.Timestamp(start))&(all_data[sym]["date"]<=pd.Timestamp(end))]
            if len(bdf)>0: bench_rets.append(bdf["close"].iloc[-1]/bdf["close"].iloc[0]-1)
    bench_avg = np.mean(bench_rets)*100
    print(f"  策略: {ret:+.1f}%  基准: {bench_avg:+.1f}%  超额: {ret-bench_avg:+.1f}%  ({trade_count}笔)")


if __name__ == "__main__":
    main()
