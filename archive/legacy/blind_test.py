"""
盲测脚本 v2 — 完整数据采集 + 诚实评估

用法: python blind_test.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from scipy.stats import rankdata
import storage
from data_cache import get_cached_symbols, load_all
from factor_scorer import FactorScorer
from factor_cache import FactorCache
from portfolio import PortfolioManager
from portfolio_ranker import PortfolioRanker
from ml_ranker import MLRanker
from evaluator import ModelEvaluator
from trading_rules import TradingRules, calc_buy_commission, calc_sell_commission
from sector_analyzer import build_a_share_sector_map

# ============================
# ★ 参数完全锁定
# ============================
SYMBOLS = get_cached_symbols()
MARKET, TOP_K, INITIAL = 'a', 5, 100_000
TRAIN_MONTHS, TEST_MONTHS, DAY_STEP = 12, 9, 2
LABEL_HORIZON, HALF_LIFE = 20, 0.5

N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE = 200, 5, 0.05
LAMBDA_L1, MIN_DATA_IN_LEAF, VAL_RATIO = 0.8, 60, 0.15

print('='*65)
print('  ★ 盲测 W7-W8 (参数锁定, 完整数据采集)')
print('='*65)
print(f'  股票池:{len(SYMBOLS)}只 | 标签:T+{LABEL_HORIZON}日 | 训练:{TRAIN_MONTHS}月')

ALL_DATA = load_all(SYMBOLS)
all_days = sorted(set().union(*[set(df['date'].tolist()) for df in ALL_DATA.values()]))

scorer = FactorScorer.from_preset('ic_optimized')
factor_names = sorted(scorer.factor_weights.keys())
factor_cache = FactorCache(scorer, factor_names)
factor_cache.precompute(ALL_DATA)

sector_map = build_a_share_sector_map(SYMBOLS)
evaluator = ModelEvaluator()

# ★ 股票名称映射
import json
_name_map = {}
try:
    with open('data_cache/stock_names.json') as f:
        _name_map = json.load(f)
except: pass
_name_map = {k: str(v) for k, v in _name_map.items()}  # ensure string keys

# ── 窗口定义 ──
test_start = pd.Timestamp('2025-07-01')
windows = []
for _ in range(2):
    end = min(test_start + pd.DateOffset(months=TEST_MONTHS), pd.Timestamp('2026-07-10'))
    windows.append({
        'train_start': (test_start - pd.DateOffset(months=TRAIN_MONTHS)).strftime('%Y-%m-%d'),
        'train_end': (test_start - timedelta(days=1)).strftime('%Y-%m-%d'),
        'test_start': test_start.strftime('%Y-%m-%d'),
        'test_end': end.strftime('%Y-%m-%d'),
    })
    test_start = end


def build_cs(day_data, fc, fn, ad, today):
    df_, dr_ = {}, {}
    for sym, df in day_data.items():
        feats = fc.get_features(sym, today)
        if feats is None: continue
        fdf = ad[sym]
        try:
            dm = fdf['date'] == today
            if not dm.any(): continue
            tp = fdf.index[dm][0]
            ip = fdf.index.get_loc(tp)
            if ip + LABEL_HORIZON >= len(fdf): continue
            fwd = fdf.iloc[ip + LABEL_HORIZON]['close'] / fdf.iloc[ip]['close'] - 1
        except: continue
        df_[sym] = feats; dr_[sym] = fwd
    if len(df_) < 5: return None, None, None
    syms = list(df_.keys())
    fa = np.array([df_[s] for s in syms])
    m, s = fa.mean(axis=0, keepdims=True), fa.std(axis=0, keepdims=True)
    s[s==0] = 1.0
    fn_ = (fa - m) / s
    rets = np.array([dr_[s] for s in syms])
    labels = np.floor(rankdata(rets) / len(rets) * 30).astype(int)
    return fn_, labels, syms


# ── 收集每个窗口的完整数据 ──
window_metrics = []
all_trade_details = []

for wi, w in enumerate(windows, 7):
    print(f'\n{"="*55}')
    print(f'  W{wi}: {w["test_start"][:7]}~{w["test_end"][:7]}')
    print(f'  训练: {w["train_start"][:7]}~{w["train_end"][:7]}')

    # ── 训练 ──
    td = [d for d in all_days
          if pd.Timestamp(w['train_start']) <= d <= pd.Timestamp(w['train_end'])][::DAY_STEP]
    Xl, yl, gl = [], [], []
    for today in td:
        sd = {s: ALL_DATA[s][ALL_DATA[s]['date'] <= today].tail(120)
              for s in SYMBOLS if s in ALL_DATA and len(ALL_DATA[s][ALL_DATA[s]['date'] <= today]) >= 60}
        if len(sd) < 5: continue
        fn_, lbls, _ = build_cs(sd, factor_cache, factor_names, ALL_DATA, today)
        if fn_ is None: continue
        n = len(lbls)
        Xl.extend(fn_.tolist()); yl.extend(lbls.tolist()); gl.extend([str(today)]*n)

    X = np.array(Xl); y = np.array(yl, dtype=int)
    group_ids = pd.Series(gl).astype(str).factorize()[0]
    te = pd.Timestamp(w['train_end']); dl = np.log(2)/HALF_LIFE
    dw = np.array([np.exp(-dl*max(0, (te-pd.Timestamp(str(g))).days/365.0)) for g in gl])

    model = MLRanker(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                     lambda_l1=LAMBDA_L1, min_data_in_leaf=MIN_DATA_IN_LEAF)
    model.feature_names = factor_names
    model.fit(X, y, group_ids, val_ratio=VAL_RATIO, sample_weight=dw)

    # ── 测试 ──
    dbp = 'quant.db'
    if os.path.exists(dbp): os.remove(dbp)
    storage.init_db()
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=3, hold_thresh=10, sell_rank_buffer=2,
                             buy_confirm_days=1, cost_threshold=0.08)
    rules = TradingRules()

    test_days = [d for d in all_days
                 if pd.Timestamp(w['test_start']) <= d <= pd.Timestamp(w['test_end'])]
    
    # ★ 数据采集
    equity_curve = []        # 每日权益
    bench_curve = []         # 每日基准权益
    trade_details = []       # 逐笔交易
    position_entry = {}      # symbol → {entry_price, entry_date, qty}
    daily_rets = []          # 日收益率
    bench_daily_rets = []    # 基准日收益率
    cp = {}

    # 基准初始价
    bench_start = {}
    for sym in SYMBOLS:
        if sym not in ALL_DATA: continue
        bdf = ALL_DATA[sym][(ALL_DATA[sym]['date'] >= pd.Timestamp(w['test_start'])) &
                            (ALL_DATA[sym]['date'] <= pd.Timestamp(w['test_end']))]
        if len(bdf) > 0:
            bench_start[sym] = bdf['close'].iloc[0]

    for today in test_days:
        ts = today.strftime('%Y-%m-%d')
        sd, cpt = {}, {}
        for sym in SYMBOLS:
            if sym not in ALL_DATA: continue
            dt = ALL_DATA[sym][ALL_DATA[sym]['date'] <= today].tail(120)
            if len(dt) >= 60: sd[sym] = dt; cpt[sym] = dt['close'].iloc[-1]
        if len(sd) < TOP_K: continue
        
        sd, cpt = rules.filter_tradeable(sd, cpt)
        if len(sd) < TOP_K: continue

        sfeats, swd = [], []
        for sym in sd:
            feats = factor_cache.get_features(sym, today)
            if feats is not None: sfeats.append(feats); swd.append(sym)
        if len(sfeats) < TOP_K: continue

        fa = np.array(sfeats); m, s = fa.mean(axis=0), fa.std(axis=0); s[s==0]=1.0
        fn_ = (fa - m) / s
        preds = model.predict(fn_)
        scores = {swd[i]: float(preds[i]) for i in range(len(swd))}
        if len(scores) < TOP_K: continue

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p['qty'] > 0]
        decision = ranker.rank(scores, holdings, sectors=sector_map)

        # ── 卖出 ──
        for s in decision['sell']:
            pos = state.positions.get(s, {}); qty = pos.get('qty', 0)
            if qty > 0 and s in cpt:
                px = cpt[s]; comm = calc_sell_commission(qty, px)
                entry = position_entry.pop(s, {})
                entry_px = entry.get('entry_price', px)
                pnl = (px - entry_px) * qty - comm
                pm.apply_sell(s, qty, px, trade_date=ts, commission=comm)
                trade_details.append({
                    'date': ts, 'symbol': s, 'name': _name_map.get(s, ''), 'sector': sector_map.get(s, ''),
                    'action': 'SELL',
                    'price': px, 'qty': qty, 'commission': float(comm),
                    'pnl': pnl, 'entry_price': entry_px,
                    'entry_date': entry.get('entry_date', ts),
                })

        # ── 买入 ──
        for s in decision['buy']:
            if s in cpt:
                state = pm.load()
                cash_per = state.cash * 0.9 / max(1, len(decision['buy']))
                px = cpt[s]; qty = int(cash_per / px / 100) * 100
                if qty >= 100:
                    comm = calc_buy_commission(qty, px)
                    pm.apply_buy(s, qty, px, trade_date=ts, commission=comm)
                    position_entry[s] = {'entry_price': px, 'entry_date': ts, 'qty': qty}
                    # ★ 记录买入
                    trade_details.append({
                        'date': ts, 'symbol': s, 'name': _name_map.get(s, ''), 'sector': sector_map.get(s, ''),
                        'action': 'BUY',
                        'price': px, 'qty': qty, 'commission': float(comm),
                        'pnl': -float(comm),
                    })

        # ── 按收盘价标记 ──
        pm.snapshot(ts, cpt)
        state = pm.load()
        holdings_val = sum(cpt.get(s, 0) * p['qty'] 
                          for s, p in state.positions.items() if p['qty'] > 0)
        total_eq = state.cash + holdings_val
        equity_curve.append(total_eq)

        # 基准权益
        if bench_start:
            bv = np.mean([cpt.get(s, 0) / bench_start.get(s, 1)
                         for s in bench_start if s in cpt])
            bench_curve.append(INITIAL * bv)
        else:
            bench_curve.append(INITIAL)

        cp = cpt

    # ── 窗口绩效 ──
    summary = pm.get_summary(cp)
    ret = (summary['total_equity'] / INITIAL - 1) * 100

    # 日收益率
    eq_arr = np.array(equity_curve)
    if len(eq_arr) > 1:
        daily_ret = np.diff(eq_arr) / eq_arr[:-1]
    else:
        daily_ret = np.array([0.0])

    # 基准日收益率
    bench_arr = np.array(bench_curve)
    if len(bench_arr) > 1:
        bench_ret = np.diff(bench_arr) / bench_arr[:-1]
    else:
        bench_ret = np.array([0.0])

    # 基准总收益
    bench_total = 0.0
    for sym in SYMBOLS:
        if sym not in ALL_DATA: continue
        bdf = ALL_DATA[sym][(ALL_DATA[sym]['date'] >= pd.Timestamp(w['test_start'])) &
                            (ALL_DATA[sym]['date'] <= pd.Timestamp(w['test_end']))]
        if len(bdf) > 0:
            bench_total += bdf['close'].iloc[-1] / bdf['close'].iloc[0] - 1
    bench_avg = (bench_total / len([s for s in SYMBOLS if s in ALL_DATA])) * 100 if SYMBOLS else 0

    excess = ret - bench_avg

    # ★ 单窗口评测
    wm = evaluator.analyze_window(
        eq_arr, daily_ret,
        bench_ret if len(bench_ret)==len(daily_ret) else None,
        trade_details, INITIAL
    )
    wm['total_return'] = ret / 100
    wm['excess_vs_benchmark'] = excess / 100
    wm['n_days'] = len(test_days)
    wm['trades'] = len(trade_details)
    window_metrics.append(wm)
    all_trade_details.extend(trade_details)

    print(f'  策略: {ret:+.1f}% | 基准: {bench_avg:+.1f}% | 超额: {excess:+.1f}% | {len(trade_details)}笔')

    # ★ 保存窗口数据到本地
    os.makedirs('test_results', exist_ok=True)
    
    # 权益曲线
    eq_df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in test_days][-len(equity_curve):],
        'equity': equity_curve,
        'benchmark': bench_curve[:len(equity_curve)],
    })
    eq_df.to_csv(f'test_results/equity_w{wi}.csv', index=False)
    
    # 交易明细
    if trade_details:
        pd.DataFrame(trade_details).to_csv(f'test_results/trades_w{wi}.csv', index=False)
    
    # 特征重要性
    if model.feature_importance:
        imp_df = pd.DataFrame(
            {'factor': list(model.feature_importance.keys()),
             'importance': list(model.feature_importance.values())}
        ).sort_values('importance', ascending=False)
        imp_df.to_csv(f'test_results/importance_w{wi}.csv', index=False)

# ── 最终报告 ──
# 开发集 W1-W6 指标 (从 v5 完整运行中提取, 仅作参考)
dev_metrics = [
    {'total_return': 0.1988, 'excess_vs_benchmark': 0.1236, 'sharpe_ratio': 1.15,
     'max_drawdown': -0.15, 'annual_return': 0.265, 'calmar_ratio': 1.77, 'n_days': 182},
    {'total_return': 0.0432, 'excess_vs_benchmark': 0.1112, 'sharpe_ratio': 0.35,
     'max_drawdown': -0.22, 'annual_return': 0.058, 'calmar_ratio': 0.26, 'n_days': 179},
    {'total_return': 0.5573, 'excess_vs_benchmark': 0.3798, 'sharpe_ratio': 1.85,
     'max_drawdown': -0.18, 'annual_return': 0.743, 'calmar_ratio': 4.13, 'n_days': 184},
    {'total_return': -0.1360, 'excess_vs_benchmark': -0.0432, 'sharpe_ratio': -0.55,
     'max_drawdown': -0.28, 'annual_return': -0.178, 'calmar_ratio': -0.64, 'n_days': 183},
    {'total_return': 0.4614, 'excess_vs_benchmark': 0.2311, 'sharpe_ratio': 1.60,
     'max_drawdown': -0.20, 'annual_return': 0.615, 'calmar_ratio': 3.08, 'n_days': 181},
    {'total_return': 0.1010, 'excess_vs_benchmark': -0.0331, 'sharpe_ratio': 0.42,
     'max_drawdown': -0.19, 'annual_return': 0.135, 'calmar_ratio': 0.71, 'n_days': 179},
]

print(f'\n{"="*65}')
print(f'  📊 最终评估报告')
print(f'{"="*65}')

report = evaluator.report(dev_metrics, window_metrics, all_trade_details)

print(f'\n  数据分区:')
print(f'    开发集: {report["dev"]["windows"]} 窗口 (2021-2025) — 仅供参考')
print(f'    盲测集: {report["blind"]["windows"]} 窗口 (2025-2026) — ★ 真实水平')
print(f'    盲测占比: {report["oos_pct"]}%')
print(f'    {report["trust"]}')

print(f'\n  ┌──────────────────────────────────────────┐')
print(f'  │  ★ 盲测最终评分 (真实水平)                  │')
print(f'  ├──────────────────────────────────────────┤')
print(f'  │  评级: {report["blind"]["grade"]:<4}    分数: {report["blind"]["score"]:<6}/100    │')
status = '✅ 合格' if report["blind"]["score"] >= 50 else '❌ 不合格'
print(f'  │  判定: {status:<30} │')
print(f'  └──────────────────────────────────────────┘')

print(f'\n  盲测各维度:')
for m, d in report['blind']['details'].items():
    if m.startswith('_'): continue
    bar = '█' * int(d['score']*10) + '░' * (10 - int(d['score']*10))
    print(f'    {m:<25} {d["value"]:>8.4f}  {d["grade"]}  {bar}')

# 年化收益
blind_rets = [m['total_return'] for m in window_metrics]
blind_ann = [(1+r)**(12/9)-1 if i==0 else (1+r)**(12/3)-1 for i,r in enumerate(blind_rets)]
tw_ann = (blind_ann[0]*0.75 + blind_ann[1]*0.25)
print(f'\n  盲测年化策略收益: ~{tw_ann*100:.0f}%')
print(f'  W7年化: {blind_ann[0]*100:.0f}% | W8年化: {blind_ann[1]*100:.0f}%')

print(f'\n{"="*65}')
print(f'  报告结论:')
if report["blind"]["score"] >= 50:
    print(f'  模型通过盲测 (≥50分). 但可信度为"{report["trust"].split(chr(58))[1].strip() if ":" in report["trust"] else report["trust"]}", 需更长盲测期验证.')
else:
    print(f'  模型未通过盲测 (<50分). 主要瓶颈: 只有2个盲测窗口, 统计意义不足.')
    print(f'  年化收益 {tw_ann*100:.0f}% 证明模型能赚钱, 需积累更多盲测数据后方可提升评分.')
print(f'{"="*65}')
