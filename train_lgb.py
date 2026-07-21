"""
LightGBM训练 — 替代网格搜索,学非线性因子组合

方法: 用训练期(2020-2023)的因子面板训练LightGBM回归
      验证/测试期用训练好的模型预测分数

用法: python train_lgb.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from scipy import stats
import storage, json
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from ml_ranker import MLRanker

SYMBOLS = ["688981","002371","603986","002049","300033","002230",
           "300750","002594","600519","600036"]
MARKET, TOP_K, INITIAL = "a", 4, 100_000
FACTOR_NAMES = [
    "volatility_20d","ma5_ma20_spread","ma10_ma20_spread","ma20_ma60_spread",
    "ma5_cross_ma20","vol_ratio","kmid2","klen","ksft2",
    "rsv_9","cntd_20","rank_20","turnover_ratio",
]


def build_panel(start: str, end: str, max_days: int = 200) -> pd.DataFrame:
    """构建训练面板: 每天×每只股票的因子值+未来收益。"""
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        df = fetcher.fetch(sym, "20180101", end.replace("-",""), "qfq", market=MARKET)
        df["date"] = pd.to_datetime(df["date"])
        all_data[sym] = df

    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    days = days[::max(1, len(days)//max_days)][:max_days]  # 采样
    
    scorer = FactorScorer.from_preset("ic_optimized")
    panel = []
    print(f"  处理 {len(days)} 个交易日...")

    for today in days:
        sd = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(dt) < 50: continue; sd[sym] = dt
        if len(sd) < 5: continue
        
        # 直接用截面评分获取因子值
        try:
            factors_dict = {}
            for sym in sd:
                f = scorer.compute_factors(sd[sym])
                if len(f) == 0: continue
                row = f.iloc[-1]
                factors_dict[sym] = {fn: row.get(fn, np.nan) for fn in FACTOR_NAMES}
            
            for sym, fvals in factors_dict.items():
                close = sd[sym]["close"].values
                fwd = close[-1]/close[-6]-1 if len(close)>=6 else 0
                rec = {"date": today, "symbol": sym, "fwd_ret": fwd}
                rec.update(fvals)
                panel.append(rec)
        except Exception as e:
            if len(panel) == 0: print(f"  err: {e}")
            continue

    df = pd.DataFrame(panel)
    print(f"  面板: {len(df)}条, 列={len(df.columns)}")
    return df


def backtest_lgb(model, start: str, end: str, label: str) -> dict:
    """用LightGBM模型跑回测。"""
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()

    cfg, pm = MARKET_CONFIG[MARKET], PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K)
    macro = MacroOverlay(market=MARKET); macro.update()

    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        df = fetcher.fetch(sym, "20180101", end.replace("-",""), "qfq", market=MARKET)
        df["date"] = pd.to_datetime(df["date"]); all_data[sym] = df

    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    scorer = FactorScorer.from_preset("ic_optimized")

    for today in days:
        ts = today.strftime("%Y-%m-%d")
        sd, cp, scores = {}, {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(dt) < 50: continue; sd[sym] = dt; cp[sym] = dt["close"].iloc[-1]
        if len(sd) < TOP_K: continue

        # LightGBM预测: 用因子值→ML预测分数
        try:
            for sym in sd:
                factors = scorer.compute_factors(sd[sym])
                if len(factors) == 0: continue
                feats = [factors[fn].iloc[-1] if fn in factors.columns else 0 for fn in FACTOR_NAMES]
                scores[sym] = float(model.predict(np.array([feats]))[0])
        except: continue
        if len(scores) < TOP_K: continue

        for s in scores: scores[s] *= (1 + macro.score_at(today) * 0.3)
        state = pm.load(); holdings = [s for s,p in state.positions.items() if p["qty"]>0]
        decision = ranker.rank(scores, holdings)
        for s in decision["sell"]:
            pos = state.positions.get(s,{}); qty = pos.get("qty",0)
            if qty>0 and s in cp: pm.apply_sell(s,qty,cp[s],trade_date=ts,commission=qty*cp[s]*0.0008)
        for s in decision["buy"]:
            if s in cp:
                state=pm.load(); cash_p=state.cash*0.9/max(1,len(decision["buy"]))
                px=cp[s]; qty=int(cash_p/px/100)*100
                if qty>=100: pm.apply_buy(s,qty,px,trade_date=ts,commission=qty*px*0.0003)
        pm.snapshot(ts, cp)

    summary = pm.get_summary(cp)
    ret = (summary["total_equity"]/INITIAL-1)*100
    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            bdf = all_data[sym][(all_data[sym]["date"]>=pd.Timestamp(start))&(all_data[sym]["date"]<=pd.Timestamp(end))]
            if len(bdf)>0: bench_rets.append(bdf["close"].iloc[-1]/bdf["close"].iloc[0]-1)
    bench_avg = np.mean(bench_rets)*100
    print(f"  [{label}] 策略={ret:+.1f}% 基准={bench_avg:+.1f}% 超额={ret-bench_avg:+.1f}%")
    return {"strategy": ret, "benchmark": bench_avg, "excess": ret-bench_avg}


def main():
    print("=" * 70)
    print("  LightGBM 训练 — 非线性因子组合")
    print("=" * 70)

    # 训练
    print("\n[训练] 构建因子面板 (2020-2023)...")
    df_train = build_panel("2020-01-01", "2023-12-31")
    print(f"  面板: {len(df_train)}条")

    X = df_train[FACTOR_NAMES].fillna(0).values
    y = df_train["fwd_ret"].values
    groups = df_train.groupby("date").ngroup().values

    print(f"  训练 LightGBM ({len(X)}样本, {len(FACTOR_NAMES)}特征)...")
    model = MLRanker(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.fit(X, y, groups)
    print(f"  训练完成")

    # 验证
    print(f"\n[验证] 2024-01 ~ 2025-06")
    r1 = backtest_lgb(model, "2024-01-01", "2025-06-30", "验证期")

    # 测试
    print(f"\n[测试] 2025-07 ~ 2026-07")
    r2 = backtest_lgb(model, "2025-07-01", "2026-07-10", "测试期")

    print(f"\n{'='*70}")
    print(f"  LightGBM 结果")
    print(f"{'='*70}")
    print(f"  测试期超额: {r2['excess']:+.1f}%")
    model.save(os.path.join(os.path.dirname(__file__), "lgb_model.pkl"))

    # 对比之前的默认权重
    print(f"\n  对比: 默认权重 +30.3% → LightGBM {r2['excess']:+.1f}%")


if __name__ == "__main__":
    main()
