"""
滚动重训练 — 每6个月用最近3年数据重新训练LightGBM

honest result = 所有窗口测试期超额的均值

用法: python test_rolling.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from datetime import timedelta
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
TRAIN_YEARS = 3
TEST_MONTHS = 12  # 每12个月一个窗口

print("="*60)
print(f"  滚动重训练 — 每{TEST_MONTHS}月用近{TRAIN_YEARS}年数据重训")
print("="*60)

# ════════════════════════════════════════
ALL_DATA = load_all(SYMBOLS)
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))
scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())
cfg = MARKET_CONFIG[MARKET]

# ════════════════════════════════════════
#  生成滚动窗口
# ════════════════════════════════════════
test_start = pd.Timestamp("2021-01-01")
test_end = pd.Timestamp("2026-07-10")
windows = []

current = test_start
while current < test_end:
    test_period_end = min(current + pd.DateOffset(months=TEST_MONTHS), test_end)
    train_start = current - pd.DateOffset(years=TRAIN_YEARS)
    windows.append({
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
        "test_start": current.strftime("%Y-%m-%d"),
        "test_end": test_period_end.strftime("%Y-%m-%d"),
    })
    current = test_period_end

print(f"  窗口数: {len(windows)}")

# ════════════════════════════════════════
#  每个窗口: 训练 → 测试
# ════════════════════════════════════════
all_results = []

for wi, w in enumerate(windows):
    print(f"\n{'='*50}")
    print(f"  窗口 {wi+1}/{len(windows)}: 训练{w['train_start'][:7]}~{w['train_end'][:7]} → 测试{w['test_start'][:7]}~{w['test_end'][:7]}")
    
    # ---- 训练 ----
    train_days = [d for d in all_days if pd.Timestamp(w["train_start"]) <= d <= pd.Timestamp(w["train_end"])]
    train_days = train_days[::3]
    
    X_list, y_list, group_list = [], [], []
    for today in train_days:
        sd = {}
        for sym in SYMBOLS:
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 60: sd[sym] = dt
        if len(sd) < 5: continue
        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if len(f) == 0: continue
            row = f.iloc[-1]; close = sd[sym]["close"].values
            if len(close) < 6: continue
            feats = [float(row.get(fn, 0)) for fn in factor_names]
            X_list.append(feats); y_list.append(close[-1]/close[-6]-1)
            group_list.append(today)
        if len(X_list) >= 1500: break
    
    if len(X_list) < 100: continue
    
    X, y = np.array(X_list), np.array(y_list)
    groups = pd.Series(group_list).astype(str).factorize()[0]
    model = MLRanker(n_estimators=50, max_depth=5, learning_rate=0.05)
    model.fit(X, y, groups, val_ratio=0.2)
    
    # ---- 测试 ----
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()
    
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=2, hold_thresh=10)
    macro = MacroOverlay(market=MARKET); macro.update()
    
    test_days = [d for d in all_days if pd.Timestamp(w["test_start"]) <= d <= pd.Timestamp(w["test_end"])]
    trades = 0; cp = {}
    
    for today in test_days:
        ts = today.strftime("%Y-%m-%d")
        sd, cp_today = {}, {}
        for sym in SYMBOLS:
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 60: sd[sym] = dt; cp_today[sym] = dt["close"].iloc[-1]
        if len(sd) < TOP_K: continue
        
        scores = {}
        for sym in sd:
            f = scorer.compute_factors(sd[sym])
            if len(f) == 0: continue
            row = f.iloc[-1]
            feats = np.array([[float(row.get(fn, 0)) for fn in factor_names]])
            try: scores[sym] = float(model.predict(feats)[0])
            except: scores[sym] = 0.0
        
        if len(scores) < TOP_K: continue
        for s in scores: scores[s] *= (1 + macro.score_at(today) * 0.3)
        
        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)
        
        for s in decision["sell"]:
            pos = state.positions.get(s, {}); q = pos.get("qty", 0)
            if q > 0 and s in cp_today:
                pm.apply_sell(s, q, cp_today[s], trade_date=ts, commission=q*cp_today[s]*0.0008)
                trades += 1
        for s in decision["buy"]:
            if s in cp_today:
                state = pm.load()
                cash_p = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp_today[s]; q = int(cash_p / px / 100) * 100
                if q >= 100:
                    pm.apply_buy(s, q, px, trade_date=ts, commission=q*px*0.0003)
                    trades += 1
        pm.snapshot(ts, cp_today)
        cp = cp_today
    
    summary = pm.get_summary(cp)
    ret = (summary["total_equity"] / INITIAL - 1) * 100
    
    bench_rets = []
    for sym in SYMBOLS:
        bdf = ALL_DATA[sym][(ALL_DATA[sym]["date"] >= pd.Timestamp(w["test_start"])) & 
                            (ALL_DATA[sym]["date"] <= pd.Timestamp(w["test_end"]))]
        if len(bdf) > 0: bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100
    
    excess = ret - bench_avg
    all_results.append({"window": wi+1, "train": f'{w["train_start"][:7]}~{w["train_end"][:7]}',
                        "test": f'{w["test_start"][:7]}~{w["test_end"][:7]}',
                        "strategy": ret, "benchmark": bench_avg, "excess": excess,
                        "trades": trades})
    mark = "✅" if excess > 0 else "❌"
    print(f"  策略: {ret:+.1f}%  基准: {bench_avg:+.1f}%  超额: {excess:+.1f}%  {trades}笔 {mark}")


# ════════════════════════════════════════
#  汇总
# ════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  滚动重训练 — 最终结果")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
print(f"\n  正超额窗口: {(df['excess']>0).sum()}/{len(df)}")
print(f"  平均超额: {df['excess'].mean():+.1f}%")
print(f"  中位数超额: {df['excess'].median():+.1f}%")
print(f"  超额标准差: {df['excess'].std():.1f}%")
print(f"\n  ★ 滚动重训练的诚实能力: {df['excess'].mean():+.1f}%")
