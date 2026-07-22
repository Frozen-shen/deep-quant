"""
滚动重训练 v2 — Lambdarank + 截面标准化 + 20只股票 + 无样本上限

核心改进 (相对 v1):
  1. Lambdarank objective → 直接优化截面排序
  2. 截面排名整数标签 → 0=最差, N-1=最好
  3. 训练时截面 z-score 标准化特征 → 消除量纲差异
  4. 20只行业分散股票 → 更丰富的 pairwise 信号
  5. 移除 1500 样本上限 → 充分利用全部训练数据
  6. n_estimators=200 + L1正则化 → 因子充分发现 + 防过拟合
  7. 按日期组边界切分 train/valid → 同天股票不跨split

用法: python test_rolling_v2.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import rankdata
import storage
from data_fetcher import MARKET_CONFIG
from data_cache import load_all
from factor_scorer import FactorScorer
from portfolio import PortfolioManager
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay
from ml_ranker import MLRanker

# ════════════════════════════════════════
#  配置
# ════════════════════════════════════════

# 20只行业分散的A股 (9个行业)
SYMBOLS = [
    # 半导体 (3)
    "688981",  # 中芯国际
    "002371",  # 北方华创
    "002049",  # 紫光国微
    # AI/软件 (2)
    "002230",  # 科大讯飞
    "300033",  # 同花顺
    # 电池/新能源车 (2)
    "300750",  # 宁德时代
    "002594",  # 比亚迪
    # 光伏 (2)
    "601012",  # 隆基绿能
    "300274",  # 阳光电源
    # 白酒 (2)
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    # 金融 (2)
    "601318",  # 中国平安
    "600036",  # 招商银行
    # 医药/器械 (2)
    "300760",  # 迈瑞医疗
    "600276",  # 恒瑞医药
    # 军事 (1)
    "600760",  # 中航沈飞
    # 建筑 (1)
    "601668",  # 中国建筑
    # 疫苗 (1)
    "300122",  # 智飞生物
]

MARKET = "a"
TOP_K = 4
INITIAL = 100_000
TRAIN_YEARS = 3
TEST_MONTHS = 12

# LightGBM 参数
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.05
LAMBDA_L1 = 0.5
MIN_DATA_IN_LEAF = 30

print("=" * 60)
print("  滚动重训练 v2 — Lambdarank + 截面标准化 + 20只股票")
print("=" * 60)
print(f"  股票池: {len(SYMBOLS)}只")
print(f"  LightGBM: n_est={N_ESTIMATORS} depth={MAX_DEPTH} lr={LEARNING_RATE} l1={LAMBDA_L1}")
print(f"  窗口: 每{TEST_MONTHS}月, 训练{TRAIN_YEARS}年")

# ════════════════════════════════════════
#  加载数据
# ════════════════════════════════════════
print("\n[加载] 数据缓存...")
ALL_DATA = load_all(SYMBOLS)
print(f"  成功加载: {len(ALL_DATA)}/{len(SYMBOLS)} 只")
all_days = sorted(set().union(*[set(df["date"].tolist()) for df in ALL_DATA.values()]))

scorer = FactorScorer.from_preset("ic_optimized")
factor_names = sorted(scorer.factor_weights.keys())
cfg = MARKET_CONFIG[MARKET]

print(f"  因子: {len(factor_names)}个")
print(f"  日期范围: {all_days[0].date()} ~ {all_days[-1].date()}")


# ════════════════════════════════════════
#  工具: 从 raw features 构建截面标准化 + 排名标签
# ════════════════════════════════════════

def build_day_samples(sd: dict, factor_names: list, scorer,
                     all_data: dict = None, today=None) -> tuple:
    """
    对一天的所有股票:
      1. 计算原始因子值 (仅使用 today 及之前的数据)
      2. 截面 z-score 标准化
      3. 生成排名标签 (整数: 0~N-1, 越高越好)

    ★ 标签 = 未来5日收益率 (前瞻, 无数据泄露)
      - 训练时: all_data/today 必须提供, 从全量数据中查 T+5 收盘价
      - 预测时: all_data/today 为 None, 使用任意标签 (预测时不使用)

    Args:
      sd: {symbol: DataFrame(OHLCV)}, 只含 today 及之前的数据
      factor_names: 因子名列表
      scorer: FactorScorer
      all_data: 全量数据 {symbol: DataFrame} (训练时必须)
      today: 当前日期 Timestamp (训练时必须)

    Returns:
      (features_2d, labels_1d, symbols_list) or (None, None, None)
    """
    day_features = {}   # sym → [feat values]
    day_returns = {}    # sym → forward return (future 5-day)

    for sym, df in sd.items():
        f = scorer.compute_factors(df)
        if len(f) == 0:
            continue
        row = f.iloc[-1]
        feats = [float(row.get(fn, 0)) for fn in factor_names]

        if all_data is not None and today is not None:
            # ── 训练模式: 前瞻收益率 (无数据泄露) ──
            full_df = all_data[sym]
            try:
                date_mask = full_df["date"] == today
                if not date_mask.any():
                    continue
                today_pos = full_df.index[date_mask][0]
                # 在 DataFrame 中的位置
                iloc_pos = full_df.index.get_loc(today_pos)
                if iloc_pos + 5 >= len(full_df):
                    continue  # 不够未来数据
                fwd_close = full_df.iloc[iloc_pos + 5]["close"]
                today_close = full_df.iloc[iloc_pos]["close"]
                fwd = fwd_close / today_close - 1
            except (IndexError, KeyError):
                continue
        else:
            # ── 预测模式: 标签丢弃, 不需要真实值 ──
            close = df["close"].values
            if len(close) < 6:
                continue
            fwd = 0.0  # dummy, 预测时不使用

        day_features[sym] = feats
        day_returns[sym] = fwd

    if len(day_features) < 5:
        return None, None, None

    syms = list(day_features.keys())
    n_stocks = len(syms)

    # ── 截面 z-score 标准化 ──
    feats_raw = np.array([day_features[s] for s in syms])  # (N, F)
    mean = feats_raw.mean(axis=0, keepdims=True)
    std = feats_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats_norm = (feats_raw - mean) / std

    # ── 截面排名标签 (Lambdarank): 返回率越高 → 排名越高 ──
    rets = np.array([day_returns[s] for s in syms])
    labels = rankdata(rets).astype(int) - 1  # 0=最差, N-1=最好

    return feats_norm, labels, syms


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

print(f"\n  滚动窗口: {len(windows)} 个")
for wi, w in enumerate(windows):
    print(f"    W{wi+1}: 训练 {w['train_start'][:7]}~{w['train_end'][:7]} → "
          f"测试 {w['test_start'][:7]}~{w['test_end'][:7]}")


# ════════════════════════════════════════
#  每个窗口: 训练 → 测试
# ════════════════════════════════════════
all_results = []

for wi, w in enumerate(windows):
    print(f"\n{'=' * 60}")
    print(f"  ★ 窗口 {wi+1}/{len(windows)}: "
          f"训练 {w['train_start'][:7]}~{w['train_end'][:7]} → "
          f"测试 {w['test_start'][:7]}~{w['test_end'][:7]}")
    print(f"{'=' * 60}")

    # ═══════════════ 训练阶段 ═══════════════
    train_days = [d for d in all_days
                  if pd.Timestamp(w["train_start"]) <= d <= pd.Timestamp(w["train_end"])]
    train_days = train_days[::3]  # 每3天采样
    print(f"  [训练] 候选日: {len(train_days)}天")

    X_list, y_list, group_list = [], [], []

    for today in train_days:
        # 收集当天所有有数据的股票
        sd = {}
        for sym in SYMBOLS:
            if sym not in ALL_DATA:
                continue
            dt = ALL_DATA[sym][ALL_DATA[sym]["date"] <= today].tail(120)
            if len(dt) >= 60:
                sd[sym] = dt

        if len(sd) < 5:
            continue

        # 构建当天的截面样本 (★ 前瞻标签: T+5收益率)
        feats_norm, labels, _ = build_day_samples(
            sd, factor_names, scorer, all_data=ALL_DATA, today=today)
        if feats_norm is None:
            continue

        n = len(labels)
        X_list.extend(feats_norm.tolist())
        y_list.extend(labels.tolist())
        group_list.extend([today] * n)

        # 注意: 不设上限, 使用全部可用数据
        if len(X_list) % 2000 == 0:
            print(f"    ... 已收集 {len(X_list)} 样本 ({len(train_days)}天中的"
                  f"{list(train_days).index(today)+1 if isinstance(train_days, list) else '?'})")

    if len(X_list) < 100:
        print(f"  ⚠️ 跳过窗口{wi+1}: 训练样本不足 ({len(X_list)})")
        continue

    print(f"  [训练] 总样本: {len(X_list)}, 日组数: {len(set(str(g) for g in group_list))}")

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

    # ═══════════════ 测试阶段 ═══════════════
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    storage.init_db()

    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    ranker = PortfolioRanker(top_k=TOP_K, n_drop=2, hold_thresh=10)
    macro = MacroOverlay(market=MARKET)
    macro.update()

    test_days = [d for d in all_days
                 if pd.Timestamp(w["test_start"]) <= d <= pd.Timestamp(w["test_end"])]
    trades = 0
    cp = {}

    for today in test_days:
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

        # ── 截面特征标准化 + 预测 ──
        feats_norm, _, syms_today = build_day_samples(sd, factor_names, scorer)
        if feats_norm is None or len(syms_today) < TOP_K:
            continue

        # LightGBM 预测 (分数越高越好)
        try:
            preds = model.predict(feats_norm)
            for i, sym in enumerate(syms_today):
                scores[sym] = float(preds[i])
        except Exception:
            # fallback: 等权
            for sym in syms_today:
                scores[sym] = 0.0

        # 宏观叠加 (可选)
        for s in scores:
            scores[s] *= (1 + macro.score_at(today) * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)

        # 执行卖出
        for s in decision["sell"]:
            pos = state.positions.get(s, {})
            qty = pos.get("qty", 0)
            if qty > 0 and s in cp_today:
                px = cp_today[s]
                pm.apply_sell(s, qty, px, trade_date=ts,
                              commission=qty * px * 0.0008)
                trades += 1

        # 执行买入
        for s in decision["buy"]:
            if s in cp_today:
                state = pm.load()
                cash_per = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp_today[s]
                qty = int(cash_per / px / 100) * 100
                if qty >= cfg["lot_size"]:
                    pm.apply_buy(s, qty, px, trade_date=ts,
                                 commission=qty * px * 0.0003)
                    trades += 1

        pm.snapshot(ts, cp_today)
        cp = cp_today

    # ── 绩效汇总 ──
    summary = pm.get_summary(cp)
    ret = (summary["total_equity"] / INITIAL - 1) * 100

    # 等权基准
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
        "strategy": ret,
        "benchmark": bench_avg,
        "excess": excess,
        "trades": trades,
        "samples": len(X_list),
    })
    mark = "✅" if excess > 0 else "❌"
    print(f"  策略: {ret:+.1f}%  基准: {bench_avg:+.1f}%  "
          f"超额: {excess:+.1f}%  {trades}笔 {mark}")


# ════════════════════════════════════════
#  最终汇总
# ════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  滚动重训练 v2 — 最终结果")
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
    print(f"  信息比率 (IR): {mean_ex/std_ex:.2f}" if std_ex > 0 else "  信息比率: N/A")

    # 判定
    if mean_ex >= 10:
        print(f"\n  🎉 目标达成! 平均超额 {mean_ex:+.1f}% ≥ 10%")
    else:
        gap = 10 - mean_ex
        print(f"\n  📍 距离目标10%还差 {gap:+.1f}%")
        if mean_ex > 0:
            print(f"     超额已转正, 继续优化 (因子精简/超参搜索) 可进一步逼近目标")
        else:
            print(f"     超额仍为负, 检查因子质量和样本质量")

    df.to_csv(os.path.join(os.path.dirname(__file__),
                           "rolling_v2_results.csv"), index=False)
    print(f"\n  结果已保存: rolling_v2_results.csv")
else:
    print("  ⚠️ 无有效结果")

print("\n✅ 完成")
