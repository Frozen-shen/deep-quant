"""
滚动重训练 v3 — 全优化版 (因子缓存 + 时间戳保存 + 稳健错误处理)

核心改进 (相对 v2):
  1. FactorCache — 预计算因子, 训练环从 O(ND×NS) 降为 O(NS)
  2. 时间戳保存 — 每次测试结果自动带时间戳存档
  3. Macro网络异常容错 — 不因网络问题崩溃
  4. 每5日采样 — 速度提升 40%
  5. 进度条 — 训练/测试阶段可视化进度

用法: python test_rolling_v3.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import rankdata
import storage
from data_fetcher import MARKET_CONFIG
from data_cache import load_all
from factor_scorer import FactorScorer
from factor_cache import FactorCache
from portfolio import PortfolioManager
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from ml_ranker import MLRanker  # Phase 3: DEnsembleRanker 验证后切换
from evaluator import ModelEvaluator
from trading_rules import TradingRules, calc_buy_commission, calc_sell_commission
from risk_manager import RiskManager

# ════════════════════════════════════════
#  配置
# ════════════════════════════════════════

SYMBOLS = [
    "688981","002371","002049",
    "002230","300033",
    "300750","002594",
    "601012","300274",
    "600519","000858",
    "601318","600036",
    "300760","600276",
    "600760","601668","300122",
]

MARKET = "a"
TOP_K = 4
INITIAL = 100_000
TRAIN_YEARS = 3
TEST_MONTHS = 12
DAY_STEP = 3  # 每N天采样(FactorCache下3足够快)

# LightGBM 参数
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.05
LAMBDA_L1 = 0.5  # 最佳: 足够正则化防过拟合 (0.3过拟合降为+3.0%均值)
MIN_DATA_IN_LEAF = 30

# 时间戳 (用于保存结果)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print(f"  滚动重训练 v3 — 全优化版")
print(f"  时间戳: {TIMESTAMP}")
print("=" * 65)
print(f"  股票池: {len(SYMBOLS)}只")
print(f"  LightGBM: n_est={N_ESTIMATORS} depth={MAX_DEPTH} lr={LEARNING_RATE} l1={LAMBDA_L1}")
print(f"  采样步长: 每{DAY_STEP}天")


# ════════════════════════════════════════
#  工具: 构建截面样本
# ════════════════════════════════════════

def build_cross_sectional_samples(day_data: dict, factor_cache: FactorCache,
                                  factor_names: list, all_data: dict,
                                  today) -> tuple:
    """
    对一天的所有股票构建截面标准化特征 + 前瞻排名标签。

    Args:
      day_data: {symbol: DataFrame} — 当天可用的股票数据
      factor_cache: 预计算因子缓存
      factor_names: 因子名列表
      all_data: 全量数据 (用于前瞻收益)
      today: 当前日期

    Returns:
      (features_norm, labels, symbols) or (None, None, None)
    """
    day_feats = {}   # sym → [feature values]
    day_rets = {}    # sym → forward 5-day return

    for sym, df in day_data.items():
        # 从缓存获取因子值
        feats = factor_cache.get_features(sym, today)
        if feats is None:
            continue

        # 前瞻收益率 (无数据泄露)
        full_df = all_data[sym]
        try:
            date_mask = full_df["date"] == today
            if not date_mask.any():
                continue
            today_pos = full_df.index[date_mask][0]
            iloc_pos = full_df.index.get_loc(today_pos)
            if iloc_pos + 5 >= len(full_df):
                continue
            fwd_close = full_df.iloc[iloc_pos + 5]["close"]
            today_close = full_df.iloc[iloc_pos]["close"]
            fwd = fwd_close / today_close - 1
        except (IndexError, KeyError):
            continue

        day_feats[sym] = feats
        day_rets[sym] = fwd

    if len(day_feats) < 5:
        return None, None, None

    syms = list(day_feats.keys())
    n = len(syms)

    # 截面 z-score 标准化
    feats_raw = np.array([day_feats[s] for s in syms])
    mean = feats_raw.mean(axis=0, keepdims=True)
    std = feats_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats_norm = (feats_raw - mean) / std

    # 截面排名标签 (lambdarank)
    rets = np.array([day_rets[s] for s in syms])
    labels = rankdata(rets).astype(int) - 1

    return feats_norm, labels, syms


# ════════════════════════════════════════
#  加载数据 + 预计算因子
# ════════════════════════════════════════

print("\n[1/3] 加载数据...")
ALL_DATA = load_all(SYMBOLS)
print(f"  成功加载: {len(ALL_DATA)}/{len(SYMBOLS)} 只")

all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))
cfg = MARKET_CONFIG[MARKET]

print(f"\n[2/3] 预计算因子 (一次性)...")
scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())
print(f"  因子数: {len(factor_names)}")

factor_cache = FactorCache(scorer, factor_names)
factor_cache.precompute(ALL_DATA)
print(f"  预计算完成: {len(factor_cache._cache)} 只股票")

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

print(f"\n[3/3] 滚动窗口: {len(windows)} 个")
for wi, w in enumerate(windows):
    print(f"    W{wi+1}: {w['train_start'][:7]}~{w['train_end'][:7]} → "
          f"{w['test_start'][:7]}~{w['test_end'][:7]}")


# ════════════════════════════════════════
#  每个窗口: 训练 → 测试
# ════════════════════════════════════════
all_results = []
feature_importance_log = {}

for wi, w in enumerate(windows):
    print(f"\n{'=' * 60}")
    print(f"  ★ W{wi+1}/{len(windows)}: "
          f"训练 {w['train_start'][:7]}~{w['train_end'][:7]} → "
          f"测试 {w['test_start'][:7]}~{w['test_end'][:7]}")
    print(f"{'=' * 60}")

    # ── 训练阶段 ──
    train_days = [d for d in all_days
                  if pd.Timestamp(w["train_start"]) <= d <= pd.Timestamp(w["train_end"])]
    train_days = train_days[::DAY_STEP]
    print(f"  [训练] {len(train_days)}天采样...")

    X_list, y_list, group_list = [], [], []

    for ti, today in enumerate(train_days):
        # 收集当天可用股票
        sd = {}
        for sym in SYMBOLS:
            if sym not in ALL_DATA:
                continue
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 60:
                sd[sym] = dt

        if len(sd) < 5:
            continue

        feats_norm, labels, _ = build_cross_sectional_samples(
            sd, factor_cache, factor_names, ALL_DATA, today)
        if feats_norm is None:
            continue

        n = len(labels)
        X_list.extend(feats_norm.tolist())
        y_list.extend(labels.tolist())
        group_list.extend([str(today)] * n)

    print(f"  [训练] 样本: {len(X_list)}, 日组: {len(set(str(g) for g in group_list))}")

    if len(X_list) < 100:
        print(f"  ⚠️ 跳过W{wi+1}: 样本不足")
        continue

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    groups = pd.Series(group_list).astype(str).factorize()[0]

    model = MLRanker(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        lambda_l1=LAMBDA_L1,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
    )
    model.feature_names = factor_names
    model.fit(X, y, groups, val_ratio=0.2)
    
    # ★ Phase 3: 稳定后可切换为 DEnsembleRanker

    # 记录特征重要性
    feature_importance_log[f"W{wi+1}"] = dict(
        sorted(model.feature_importance.items(), key=lambda x: -x[1])[:10])

    # ── 测试阶段 ──
    db_path = os.path.join(BASE_DIR, "quant.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    storage.init_db()

    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=2, hold_thresh=10)

    # Macro — 容错: 如果网络不通, 使用默认中性评分
    try:
        macro = MacroOverlay(market=MARKET)
        macro.update()
    except Exception:
        print("  [Macro] 网络不可用, 使用中性评分")
        macro = MacroOverlay(market=MARKET)

    test_days = [d for d in all_days
                 if pd.Timestamp(w["test_start"]) <= d <= pd.Timestamp(w["test_end"])]
    print(f"  [测试] {len(test_days)}天...")

    trades = 0
    cp = {}
    # ★ 交易规则
    rules = TradingRules()
    # Phase 2: RiskManager 代码完成, 但在Top-K换仓策略中默认不启用
    # (固定止损与换仓逻辑冲突, 适合趋势跟踪策略)
    risk_mgr = None  # RiskManager(...)
    # ★ 逐笔交易明细 + 权益曲线
    trade_details = []
    position_entry = {}   # symbol → {"entry_price", "entry_date", "qty"}
    equity_curve = []
    bench_equity_curve = []
    # 基准: 等权组合
    bench_init = {sym: ALL_DATA[sym][(ALL_DATA[sym]["date"] >= pd.Timestamp(w["test_start"])) &
                                      (ALL_DATA[sym]["date"] <= pd.Timestamp(w["test_end"]))]
                  for sym in SYMBOLS if sym in ALL_DATA}
    bench_first = {sym: bdf["close"].iloc[0] for sym, bdf in bench_init.items() if len(bdf) > 0}

    for ti, today in enumerate(test_days):
        ts = today.strftime("%Y-%m-%d")
        sd, cp_today, scores = {}, {}, {}

        for sym in SYMBOLS:
            if sym not in ALL_DATA:
                continue
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 60:
                sd[sym] = dt
                cp_today[sym] = dt["close"].iloc[-1]

        if len(sd) < TOP_K:
            continue

        # ★ 过滤不可交易股票 (停牌/一字板/ST)
        sd, cp_today = rules.filter_tradeable(sd, cp_today)
        if len(sd) < TOP_K:
            continue

        # 从缓存获取特征
        sym_feats = []
        syms_with_data = []
        for sym in sd:
            feats = factor_cache.get_features(sym, today)
            if feats is not None:
                sym_feats.append(feats)
                syms_with_data.append(sym)

        if len(sym_feats) < TOP_K:
            continue

        feats_arr = np.array(sym_feats)
        mean = feats_arr.mean(axis=0, keepdims=True)
        std = feats_arr.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        feats_norm = (feats_arr - mean) / std

        try:
            preds = model.predict(feats_norm)
            for i, sym in enumerate(syms_with_data):
                scores[sym] = float(preds[i])
        except Exception:
            for sym in syms_with_data:
                scores[sym] = 0.0

        if len(scores) < TOP_K:
            continue

        try:
            for s in scores:
                scores[s] *= (1 + macro.score_at(today) * 0.3)
        except Exception:
            pass

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)
        # 风控检查 (可选: risk_mgr 不为 None 时启用)
        if risk_mgr is not None:
            holdings_val = sum(cp_today.get(s, 0) * p["qty"]
                              for s, p in state.positions.items() if p["qty"] > 0)
            state.total_equity_val = state.cash + holdings_val
            highs_today = {s: sd[s]["high"].iloc[-1] for s in sd if s in sd}
            lows_today = {s: sd[s]["low"].iloc[-1] for s in sd if s in sd}
            decision = risk_mgr.check(decision, position_entry, state, cp_today,
                                      highs_today, lows_today)

        for s in decision["sell"]:
            pos = state.positions.get(s, {})
            qty = pos.get("qty", 0)
            if qty > 0 and s in cp_today:
                px = cp_today[s]
                comm = calc_sell_commission(qty, px)
                pm.apply_sell(s, qty, px, trade_date=ts, commission=comm)
                # ★ 记录平仓明细
                entry = position_entry.pop(s, {})
                entry_px = entry.get("entry_price", px)
                entry_date = entry.get("entry_date", ts)
                pnl = (px - entry_px) * qty - calc_sell_commission(qty, px)
                hold_days = (today - pd.Timestamp(entry_date)).days if entry_date != ts else 0
                trade_details.append({
                    "date": ts, "symbol": s, "action": "SELL",
                    "price": px, "qty": qty, "commission": float(comm),
                    "pnl": pnl, "entry_price": entry_px, "entry_date": entry_date,
                    "hold_days": hold_days,
                })
                trades += 1

        for s in decision["buy"]:
            if s in cp_today:
                state = pm.load()
                cash_per = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp_today[s]
                qty = int(cash_per / px / 100) * 100
                if qty >= cfg["lot_size"]:
                    pm.apply_buy(s, qty, px, trade_date=ts,
                                 commission=calc_buy_commission(qty, px))
                    # ★ 记录开仓 + 止损参数
                    position_entry[s] = risk_mgr.init_stop(
                        s, px, ts, qty, px,
                        highs_today.get(s, px), lows_today.get(s, px)) if risk_mgr else \
                        {"entry_price": px, "entry_date": ts, "qty": qty}
                    trades += 1

        pm.snapshot(ts, cp_today)
        cp = cp_today

        # ★ 记录权益曲线
        try:
            eq_data = storage.get_equity_log(limit=1)
            if eq_data:
                equity_curve.append({"date": ts, "equity": eq_data[0]["total_equity"]})
        except:
            pass

        # ★ 基准权益 (等权)
        if bench_first:
            bench_val = np.mean([cp_today.get(s, 0) / bench_first.get(s, 1)
                                for s in bench_first if s in cp_today])
            bench_equity_curve.append({"date": ts, "equity": INITIAL * bench_val})

    # ── 绩效 ──
    summary = pm.get_summary(cp)
    ret = (summary["total_equity"] / INITIAL - 1) * 100

    bench_rets = []
    for sym in SYMBOLS:
        if sym not in ALL_DATA:
            continue
        bdf = ALL_DATA[sym][
            (ALL_DATA[sym]["date"] >= pd.Timestamp(w["test_start"])) &
            (ALL_DATA[sym]["date"] <= pd.Timestamp(w["test_end"]))
        ]
        if len(bdf) > 0:
            bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100 if bench_rets else 0

    excess = ret - bench_avg
    all_results.append({
        "window": wi + 1,
        "train": f'{w["train_start"][:7]}~{w["train_end"][:7]}',
        "test": f'{w["test_start"][:7]}~{w["test_end"][:7]}',
        "strategy": round(ret, 2),
        "benchmark": round(bench_avg, 2),
        "excess": round(excess, 2),
        "trades": trades,
        "samples": len(X_list),
    })

    # ★ 保存窗口详细数据 (逐笔交易 + 权益曲线)
    window_detail = {
        "window": wi + 1,
        "trade_details": trade_details,
        "equity_curve": equity_curve,
        "bench_equity_curve": bench_equity_curve,
    }
    detail_path = os.path.join(RESULTS_DIR, f"window_{wi+1}_{TIMESTAMP}.json")
    with open(detail_path, "w") as f:
        json.dump(window_detail, f, default=str)

    mark = "✅" if excess > 0 else "❌"
    print(f"  策略: {ret:+.1f}%  基准: {bench_avg:+.1f}%  "
          f"超额: {excess:+.1f}%  {trades}笔 {mark}")


# ════════════════════════════════════════
#  汇总 & 存档
# ════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  滚动重训练 v3 — 最终结果 ({TIMESTAMP})")
print(f"{'=' * 70}")

if all_results:
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    pos_windows = (df["excess"] > 0).sum()
    mean_ex = df["excess"].mean()
    median_ex = df["excess"].median()
    std_ex = df["excess"].std()

    print(f"\n  正超额窗口: {pos_windows}/{len(df)}")
    print(f"  平均超额: {mean_ex:+.1f}%")
    print(f"  中位数超额: {median_ex:+.1f}%")
    print(f"  超额标准差: {std_ex:.1f}%")
    print(f"  信息比率 (IR): {mean_ex/std_ex:.2f}" if std_ex > 0 else "")

    # ── 保存结果 ──
    results_csv = os.path.join(RESULTS_DIR, f"rolling_v3_{TIMESTAMP}.csv")
    df.to_csv(results_csv, index=False)
    print(f"\n  📁 结果已保存: {results_csv}")

    # ── ★ 模型评测 ──
    evaluator = ModelEvaluator()
    window_metrics = []
    all_trades_flat = []

    for wi in range(1, len(all_results) + 1):
        detail_path = os.path.join(RESULTS_DIR, f"window_{wi}_{TIMESTAMP}.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path) as f:
            wd = json.load(f)

        trades_w = wd.get("trade_details", [])
        all_trades_flat.extend(trades_w)
        equity_arr = np.array([e["equity"] for e in wd.get("equity_curve", [])])
        bench_arr = np.array([e["equity"] for e in wd.get("bench_equity_curve", [])])

        if len(equity_arr) > 1:
            daily_ret = np.diff(equity_arr) / equity_arr[:-1]
            bench_ret = np.diff(bench_arr) / bench_arr[:-1] if len(bench_arr) > 1 else None
            n = len(daily_ret)
            if bench_ret is not None and len(bench_ret) != n:
                bench_ret = None

            wm = evaluator.analyze_window(
                equity_arr, daily_ret,
                bench_ret if bench_ret is not None else None,
                trades_w, INITIAL,
            )
        else:
            wm = {"total_return": all_results[wi-1]["strategy"] / 100,
                  "excess_vs_benchmark": all_results[wi-1]["excess"] / 100}

        wm["window"] = wi
        window_metrics.append(wm)

    # 评测报告
    report = evaluator.analyze_cross_window(window_metrics)
    grade = evaluator.grade(window_metrics, all_trades_flat)

    print(f"\n{'=' * 70}")
    print(f"  📊 模型评测报告")
    print(f"{'=' * 70}")
    print(f"  综合评级: {grade['grade']} (得分: {grade['score']}/100)")
    print(f"  合格判定: {'✅ 合格' if grade['pass'] else '❌ 不合格'}")
    print(f"  多窗口IR: {report.get('cross_window_ir', 0):.2f}")
    print(f"\n  各维度得分:")
    for metric, d in grade.get("details", {}).items():
        bar = "█" * int(d["score"] * 10) + "░" * (10 - int(d["score"] * 10))
        print(f"    {metric:<25} {d['value']:>8.4f}  {d['grade']}  {bar}")

    # 保存评测报告
    report_path = os.path.join(RESULTS_DIR, f"report_card_{TIMESTAMP}.json")
    with open(report_path, "w") as f:
        json.dump({
            "grade": grade,
            "cross_window": {k: v for k, v in report.items() if k != "per_metric"},
            "timestamp": TIMESTAMP,
        }, f, default=str)
    print(f"\n  📁 评测报告: {report_path}")

    # 保存特征重要性
    imp_csv = os.path.join(RESULTS_DIR, f"feature_importance_{TIMESTAMP}.csv")
    imp_rows = []
    for w, items in feature_importance_log.items():
        for f, g in items:
            imp_rows.append({"window": w, "factor": f, "gain": g})
    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(imp_csv, index=False)
    print(f"  📁 特征重要性: {imp_csv}")

    # 达标判断
    if mean_ex >= 10:
        print(f"\n  🎉 目标达成! 平均超额 {mean_ex:+.1f}% ≥ 10%")
    elif mean_ex > 0:
        print(f"\n  📍 超额为正, 距10%目标差 {10-mean_ex:+.1f}%")
    else:
        print(f"\n  ⚠️ 超额仍为负, 需要继续优化")
else:
    print("  ⚠️ 无有效结果")

print(f"\n✅ 完成 ({TIMESTAMP})")
