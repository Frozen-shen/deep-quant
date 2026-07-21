"""
完整优化训练 — 多特征LightGBM + 20只A股 + 动态权重

用法: python optimize_v2.py
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
import lightgbm as lgb

# ① 20只A股
SYMBOLS = [
    "688981","002371","603986","002049",  # 半导体
    "300033","002230","688111",            # 软件/AI
    "300750","002594","601012",            # 新能源
    "600519","000858",                      # 白酒
    "601318","600036",                      # 金融
    "300760","600276",                      # 医药
    "600760","000625",                      # 军工+汽车
    "601668","601899",                      # 基建+矿业
]
MARKET, TOP_K, INITIAL = "a", 4, 100_000

# ② 13个因子
FACTORS = [
    "volatility_20d","ma5_ma20_spread","ma10_ma20_spread","ma20_ma60_spread",
    "ma5_cross_ma20","vol_ratio","kmid2","klen","ksft2",
    "rsv_9","cntd_20","rank_20","turnover_ratio",
]


def build_multi_feature_panel(start: str, end: str, max_days: int = 100):
    """构建13因子面板。"""
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        df = fetcher.fetch(sym, start, end, "qfq", market=MARKET)
        if len(df) > 0: df["date"] = pd.to_datetime(df["date"]); all_data[sym] = df

    days = sorted(set().union(*[set(df["date"]) for df in all_data.values()]))
    days = [d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    days = days[::max(1, len(days)//max_days)][:max_days]

    scorer = FactorScorer.from_preset("ic_optimized")
    panel = []
    print(f"  处理 {len(days)}天, {len(all_data)}只股票...")

    for today in days:
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_t = all_data[sym][all_data[sym]["date"] <= today]
            if len(df_t) < 60: continue
            close = df_t["close"].values
            if len(close) < 6: continue

            f = scorer.compute_factors(df_t.tail(120))
            if len(f) == 0: continue
            row = f.iloc[-1]

            feat_vals = []
            for fn in FACTORS:
                v = row.get(fn, np.nan)
                feat_vals.append(0.0 if pd.isna(v) else float(v))

            fwd = close[-1] / close[-6] - 1
            panel.append({"date": today, "symbol": sym, "fwd_ret": fwd,
                          "features": feat_vals})

        if len(panel) % 500 == 0:
            print(f"    已收集 {len(panel)} 条...")

    X = np.array([p["features"] for p in panel])
    y = np.array([p["fwd_ret"] for p in panel])
    dates = [p["date"] for p in panel]
    return X, y, dates


def train_lgb(X, y, dates) -> MLRanker:
    """训练LightGBM回归模型。"""
    group_ids = pd.Series(dates).astype(str).factorize()[0]
    model = MLRanker(n_estimators=200, max_depth=8, learning_rate=0.03)
    model.feature_names = FACTORS
    model.fit(X, y, group_ids)
    return model


def backtest_lgb(model, start: str, end: str, label: str,
                 dynamic: bool = False) -> dict:
    """用LightGBM模型跑回测。可选动态权重。"""
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
        if len(df) > 0: df["date"] = pd.to_datetime(df["date"]); all_data[sym] = df

    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    scorer = FactorScorer.from_preset("ic_optimized")
    trade_count = 0

    for today in days:
        ts = today.strftime("%Y-%m-%d")
        sd, cp, scores = {}, {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(dt) < 60: continue; sd[sym] = dt; cp[sym] = dt["close"].iloc[-1]
        if len(sd) < TOP_K: continue

        # LightGBM预测分数
        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if len(f) == 0: continue
            row = f.iloc[-1]
            feats = np.array([[float(row.get(fn, 0)) for fn in FACTORS]])
            scores[sym] = float(model.predict(feats)[0])

        if len(scores) < TOP_K: continue

        # 动态权重: 牛市加momentum,熊市加contrarian
        today_macro = macro.score_at(today)
        for s in scores:
            if dynamic:
                boost = 1 + today_macro * 0.5
                scores[s] *= boost
            else:
                scores[s] *= (1 + today_macro * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)

        for s in decision["sell"]:
            pos = state.positions.get(s, {}); qty = pos.get("qty", 0)
            if qty > 0 and s in cp:
                pm.apply_sell(s, qty, cp[s], trade_date=ts, commission=qty*cp[s]*0.0008)
                trade_count += 1
        for s in decision["buy"]:
            if s in cp:
                state = pm.load()
                cash_p = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp[s]; qty = int(cash_p / px / 100) * 100
                if qty >= 100:
                    pm.apply_buy(s, qty, px, trade_date=ts, commission=qty*px*0.0003)
                    trade_count += 1
        pm.snapshot(ts, cp)

    summary = pm.get_summary(cp)
    ret = (summary["total_equity"] / INITIAL - 1) * 100
    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            bdf = all_data[sym][(all_data[sym]["date"] >= pd.Timestamp(start)) &
                                (all_data[sym]["date"] <= pd.Timestamp(end))]
            if len(bdf) > 0: bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100
    print(f"  [{label}] {ret:+.1f}% vs {bench_avg:+.1f}% = 超额{ret - bench_avg:+.1f}% ({trade_count}笔)")
    return {"strategy": ret, "benchmark": bench_avg, "excess": ret - bench_avg, "trades": trade_count}


def main():
    print("=" * 70)
    print("  完整优化: 多特征LightGBM + 20只A股 + 动态权重")
    print("=" * 70)

    # Phase 1: 训练
    print("\n[训练] 2020-2023, 多特征面板...")
    X, y, dates = build_multi_feature_panel("2020-01-01", "2023-12-31", max_days=150)
    print(f"  面板: {len(X)}条, {X.shape[1]}特征")
    model = train_lgb(X, y, dates)
    model.save(os.path.join(os.path.dirname(__file__), "lgb_v2.pkl"))

    # Phase 2: 验证 (静态)
    print(f"\n[验证] 2024-2025H1")
    r1 = backtest_lgb(model, "2024-01-01", "2025-06-30", "验证(静态)", dynamic=False)

    # Phase 3: 测试 (动态)
    print(f"\n[测试] 2025H2-2026H1")
    r2 = backtest_lgb(model, "2025-07-01", "2026-07-10", "测试(动态)", dynamic=True)

    print(f"\n{'='*70}")
    print(f"  最终结果")
    print(f"{'='*70}")
    print(f"  验证期超额: {r1['excess']:+.1f}% ({r1['trades']}笔)")
    print(f"  测试期超额: {r2['excess']:+.1f}% ({r2['trades']}笔)")
    print(f"  对比10只基线: +30.3% → 20只+LightGBM+动态: {r2['excess']:+.1f}%")


if __name__ == "__main__":
    main()
