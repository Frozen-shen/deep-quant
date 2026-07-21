"""
诚实版 v3: 28因子 + LightGBM + 训练/测试分离
单文件,无函数调用,避免变量污染
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
from ml_ranker import MLRanker

SYMBOLS = ["688981","002371","603986","002049","300033","002230","300750","002594","600519","600036"]
MARKET, TOP_K, INITIAL = "a", 4, 100_000

print("="*60)
print("  诚实版 v3 — 28因子 + LightGBM")
print("="*60)

# ════════════════════════════════════════
# 0. 加载数据 (只在最外层加载一次)
# ════════════════════════════════════════
print("\n[加载] 数据缓存...")
ALL_DATA = load_all(SYMBOLS)
print(f"  加载: {len(ALL_DATA)}只, 日期范围={ALL_DATA[SYMBOLS[0]]['date'].min().date()}~{ALL_DATA[SYMBOLS[0]]['date'].max().date()}")

# ════════════════════════════════════════
# 1. 训练期: 构建因子面板 + 训练LightGBM
# ════════════════════════════════════════
print("\n[训练] 2020-2023...")
scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())
print(f"  因子: {len(factor_names)}个")

X_list, y_list, group_list = [], [], []
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))
train_days = [d for d in all_days if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2023-12-31")]
train_days = train_days[::3]
print(f"  训练日: {len(train_days)}天")

for ti, today in enumerate(train_days):
    sd = {}
    for sym in SYMBOLS:
        df = ALL_DATA[sym]
        dt = df[df["date"] <= today].tail(120)
        if len(dt) >= 60:
            sd[sym] = dt
    if len(sd) < 5:
        continue
    for sym in sd:
        f = scorer.compute_factors(sd[sym])
        if len(f) == 0: continue
        row = f.iloc[-1]
        close_vals = sd[sym]["close"].values
        if len(close_vals) < 6: continue
        feats = [float(row.get(fn, 0)) for fn in factor_names]
        fwd = close_vals[-1] / close_vals[-6] - 1
        X_list.append(feats)
        y_list.append(fwd)
        group_list.append(today)
    if len(X_list) >= 5000:
        break

print(f"  面板: {len(X_list)}条")

if len(X_list) < 100:
    print("  训练数据不足!")
    sys.exit(1)

X = np.array(X_list)
y = np.array(y_list)
groups = pd.Series(group_list).astype(str).factorize()[0]

model = MLRanker(n_estimators=200, max_depth=6, learning_rate=0.05)
model.feature_names = factor_names
model.fit(X, y, groups, val_ratio=0.2)

# ════════════════════════════════════════
# 2. 测试期回测
# ════════════════════════════════════════
for period_name, start, end in [
    ("验证期", "2024-01-01", "2025-06-30"),
    ("★ 测试期", "2025-07-01", "2026-07-10"),
]:
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()

    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=2, hold_thresh=10)
    macro = MacroOverlay(market=MARKET)
    macro.update()

    days = [d for d in all_days if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    trade_count = 0
    close_prices = {}

    print(f"\n[{period_name}] {len(days)}天")

    for ti, today in enumerate(days):
        ts = today.strftime("%Y-%m-%d")
        sd, cp, scores = {}, {}, {}

        for sym in SYMBOLS:
            df = ALL_DATA[sym]
            dt = df[df["date"] <= today].tail(120)
            if len(dt) >= 60:
                sd[sym] = dt
                cp[sym] = dt["close"].iloc[-1]

        if len(sd) < TOP_K:
            continue

        # LightGBM 预测分数
        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if len(f) == 0: continue
            row = f.iloc[-1]
            feats = np.array([[float(row.get(fn, 0)) for fn in factor_names]])
            try:
                scores[sym] = float(model.predict(feats)[0])
            except:
                scores[sym] = 0.0

        if len(scores) < TOP_K:
            continue

        for s in scores:
            scores[s] *= (1 + macro.score_at(today) * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)

        for s in decision["sell"]:
            pos = state.positions.get(s, {})
            qty = pos.get("qty", 0)
            if qty > 0 and s in cp:
                pm.apply_sell(s, qty, cp[s], trade_date=ts, commission=qty * cp[s] * 0.0008)
                trade_count += 1

        for s in decision["buy"]:
            if s in cp:
                state = pm.load()
                cash_per = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp[s]
                qty = int(cash_per / px / 100) * 100
                if qty >= cfg["lot_size"]:
                    pm.apply_buy(s, qty, px, trade_date=ts, commission=qty * px * 0.0003)
                    trade_count += 1

        pm.snapshot(ts, cp)
        close_prices = cp

        if ti == 0:
            print(f"  Day1: {len(sd)}stocks, {len(scores)}scores, {trade_count}trades")

    summary = pm.get_summary(close_prices)
    ret = (summary["total_equity"] / INITIAL - 1) * 100

    bench_rets = []
    for sym in SYMBOLS:
        df = ALL_DATA[sym]
        bdf = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        if len(bdf) > 0:
            bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100

    print(f"  策略: {ret:+.1f}%  基准: {bench_avg:+.1f}%  超额: {ret - bench_avg:+.1f}%  ({trade_count}笔)")

print("\n✅ 完成")
