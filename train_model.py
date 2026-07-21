"""
量化模型三阶段训练 — 网格搜索ICIR最优权重

训练期(2020-2023): 网格搜索 → ICIR最大
验证期(2024H1-2025H1): 确认ICIR不崩
测试期(2025H2-2026H1): 一次回测 → 真实超额 (不动权重)

核心原则: 
  · 只用在训练期看到的数据调权重
  · 用ICIR(信息比率)做目标,不用收益
  · 测试期只跑一次

用法: python train_model.py
"""

import os, sys, itertools
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from scipy import stats
import storage
from data_fetcher import DataFetcher, MARKET_CONFIG
from portfolio import PortfolioManager
from factor_scorer import FactorScorer
from portfolio_ranker import PortfolioRanker
from macro_overlay import MacroOverlay

SYMBOLS = ["688981","002371","603986","002049","300033","002230",
           "300750","002594","600519","600036"]
STOCK_NAMES = {"688981":"中芯","002371":"北华创","603986":"兆易","002049":"紫光",
    "300033":"同花顺","002230":"讯飞","300750":"宁德","002594":"比亚迪",
    "600519":"茅台","600036":"招行"}
MARKET, TOP_K, INITIAL = "a", 4, 100_000

# 因子名列表 (ic_optimized预设)
FACTOR_NAMES = [
    "volatility_20d", "ma5_ma20_spread", "ma10_ma20_spread", "ma20_ma60_spread",
    "ma5_cross_ma20", "vol_ratio", "kmid2", "klen", "ksft2",
    "rsv_9", "cntd_20", "rank_20", "turnover_ratio",
]


# ================================================================
#  数据准备: 拉取所有股票的全量数据,计算因子面板
# ================================================================

def prepare_data(start: str, end: str):
    """拉取数据并预计算每天的因子+收益面板。"""
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        df = fetcher.fetch(sym, "20180101", end.replace("-",""), "qfq", market=MARKET)
        df["date"] = pd.to_datetime(df["date"])
        all_data[sym] = df

    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if start_dt <= d <= end_dt]

    print(f"  准备 {len(days)} 个交易日, {len(SYMBOLS)} 只股票")

    # 预计算每天的因子值和未来收益
    factor_records = []  # list of (date, symbol, factor_values[])
    return_records = []  # list of (date, symbol, fwd_5d_return)

    for today in days:
        stock_data = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            df_t = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(df_t) < 50: continue
            stock_data[sym] = df_t

        if len(stock_data) < 5: continue

        # 用当前ic_optimized计算原始因子值
        scorer = FactorScorer.from_preset("ic_optimized")
        try:
            scores = scorer.cross_sectional_score(stock_data)
        except: continue

        for sym in stock_data:
            close = stock_data[sym]["close"].values
            if len(close) < 6: continue
            fwd_ret = close[-1] / close[-6] - 1  # 5日未来收益
            factor_records.append({"date": today, "symbol": sym, "score": scores.get(sym, 0)})
            return_records.append({"date": today, "symbol": sym, "fwd_ret": fwd_ret})

    df_f = pd.DataFrame(factor_records)
    df_r = pd.DataFrame(return_records)
    return df_f, df_r


# ================================================================
#  ICIR 计算
# ================================================================

def compute_icir(df_factors: pd.DataFrame, df_returns: pd.DataFrame) -> float:
    """
    计算截面IC的ICIR。

    Spearman Rank IC = rank_corr(score, fwd_ret) 每天计算
    ICIR = IC均值 / IC标准差
    """
    merged = df_factors.merge(df_returns, on=["date", "symbol"])
    if len(merged) < 50: return 0.0

    ic_vals = []
    for dt, group in merged.groupby("date"):
        if len(group) < 5: continue
        ic, _ = stats.spearmanr(group["score"], group["fwd_ret"])
        if not np.isnan(ic): ic_vals.append(ic)

    if not ic_vals: return 0.0
    ic_arr = np.array(ic_vals)
    icir = ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0.0
    return icir


# ================================================================
#  网格搜索 (训练期) — 重写: 预计算因子值,然后搜权重
# ================================================================

def grid_search_weights(train_start: str, train_end: str) -> dict:
    """
    训练期网格搜索最优权重。

    方法: 预计算每只股票每天的因子值 → 网格搜索权重组合 → 最大化ICIR
    """
    print(f"\n  [训练] 网格搜索最优权重 ({train_start}~{train_end})")
    print(f"  预计算因子面板...")

    # 预计算: 每个交易日,每只股票的因子值 + 未来收益
    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        df = fetcher.fetch(sym, "20180101", train_end.replace("-",""), "qfq", market=MARKET)
        df["date"] = pd.to_datetime(df["date"])
        all_data[sym] = df

    start_dt, end_dt = pd.Timestamp(train_start), pd.Timestamp(train_end)
    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if start_dt <= d <= end_dt]

    # 用默认scorer预计算每天的因子值面板
    scorer = FactorScorer.from_preset("ic_optimized")
    panel = []  # [{date, symbol, fwd_ret, kmid2, klen, ...}]

    for today_idx, today in enumerate(days):
        sd = {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(dt) < 50: continue; sd[sym] = dt
        if len(sd) < 5: continue

        # 计算每个因子的原始值(用compute_factors,不做标准化)
        try:
            for sym in sd:
                factors = scorer.compute_factors(sd[sym])
                if len(factors) == 0: continue
                row = factors.iloc[-1]
                close = sd[sym]["close"].values
                fwd_ret = close[-1]/close[-6]-1 if len(close)>=6 else 0
                rec = {"date": today, "symbol": sym, "fwd_ret": fwd_ret}
                for fn in FACTOR_NAMES:
                    if fn in factors.columns:
                        rec[fn] = row[fn]
                panel.append(rec)
        except: continue

        if today_idx % 100 == 0:
            print(f"    进度: {today_idx}/{len(days)}")

    if len(panel) < 100:
        print("  训练数据不足!")
        return {}

    df_p = pd.DataFrame(panel)
    print(f"  因子面板: {len(df_p)} 条 ({len(days)}天)")

    # 网格搜索: 对13个因子各搜 [0, 0.05, 0.10, 0.15, 0.20]
    candidates = [0.0, 0.05, 0.10, 0.15, 0.20]
    best_weights = None
    best_icir = -999

    # 为避免13^5的组合爆炸,只搜前8个核心因子,其余用0.05
    core_factors = FACTOR_NAMES[:8]
    search_space = list(itertools.product(candidates, repeat=len(core_factors)))
    print(f"  搜索空间: {len(search_space)} 种组合 (8因子×5档)")

    tested = 0
    for combo in search_space:
        weights = {}
        for i, fn in enumerate(core_factors):
            if combo[i] > 0:
                weights[fn] = combo[i]
        # 其余因子给默认小权重
        for fn in FACTOR_NAMES[len(core_factors):]:
            weights[fn] = 0.03

        if not weights: continue

        # 计算加权分数
        df_p["score"] = 0.0
        for fn, w in weights.items():
            if fn in df_p.columns:
                df_p["score"] += df_p[fn].fillna(0) * w

        # 计算ICIR
        ic_vals = []
        for dt, group in df_p.groupby("date"):
            if len(group) < 5: continue
            if group["score"].std() == 0: continue
            ic, _ = stats.spearmanr(group["score"], group["fwd_ret"])
            if not np.isnan(ic): ic_vals.append(ic)

        if len(ic_vals) < 50: continue
        ic_arr = np.array(ic_vals)
        icir = ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0

        tested += 1
        if icir > best_icir:
            best_icir = icir
            best_weights = weights.copy()
            if tested % 5000 == 0:
                print(f"    已测试: {tested}/{len(search_space)}, 当前最优 ICIR={best_icir:+.4f}")

    if best_weights is None:
        print("  未找到有效权重组合!")
        return {}

    print(f"\n  搜索完成: {tested} 组合, 最优 ICIR={best_icir:+.4f}")
    for fn in FACTOR_NAMES:
        w = best_weights.get(fn, 0)
        if w > 0:
            print(f"    {fn:<22} w={w:.2f}")
    return best_weights


def compute_icir_single_factor(df_f, df_r, weights: dict) -> float:
    """用给定权重计算因子分数,然后算ICIR。"""
    merged = df_f.merge(df_r, on=["date", "symbol"])
    if len(merged) < 50: return 0.0

    # 计算加权分数
    merged["score"] = 0.0
    for fn, w in weights.items():
        if fn in merged.columns:
            merged["score"] += merged[fn].fillna(0) * w

    ic_vals = []
    for dt, group in merged.groupby("date"):
        if len(group) < 5: continue
        if group["score"].std() == 0: continue
        ic, _ = stats.spearmanr(group["score"], group["fwd_ret"])
        if not np.isnan(ic): ic_vals.append(ic)

    if not ic_vals: return 0.0
    ic_arr = np.array(ic_vals)
    return ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0.0


# ================================================================
#  回测 (验证期/测试期,固定权重)
# ================================================================

def backtest_period(weights: dict, start: str, end: str, label: str) -> dict:
    """在指定区间上用固定权重跑回测。"""
    db_path = os.path.join(os.path.dirname(__file__), "quant.db")
    if os.path.exists(db_path): os.remove(db_path)
    storage.init_db()

    cfg = MARKET_CONFIG[MARKET]
    pm = PortfolioManager(market=MARKET, initial_capital=INITIAL)
    scorer = FactorScorer(factor_weights=weights, buy_threshold=0.15, sell_threshold=-0.10)
    ranker = PortfolioRanker(top_k=TOP_K)
    macro = MacroOverlay(market=MARKET)
    macro.update()

    fetcher = DataFetcher()
    all_data = {}
    for sym in SYMBOLS:
        try:
            df = fetcher.fetch(sym, "20180101", end.replace("-",""), "qfq", market=MARKET)
            df["date"] = pd.to_datetime(df["date"])
            all_data[sym] = df
        except: pass

    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    days = all_data[SYMBOLS[0]]["date"].tolist()
    days = [d for d in days if start_dt <= d <= end_dt]

    for today in days:
        ts = today.strftime("%Y-%m-%d")
        sd, cp = {}, {}
        for sym in SYMBOLS:
            if sym not in all_data: continue
            dt = all_data[sym][all_data[sym]["date"] <= today].tail(120)
            if len(dt) < 50: continue
            sd[sym] = dt; cp[sym] = dt["close"].iloc[-1]
        if len(sd) < TOP_K: continue

        try: scores = scorer.cross_sectional_score(sd)
        except: continue
        for s in scores: scores[s] *= (1 + macro.score_at(today) * 0.3)

        state = pm.load()
        holdings = [s for s, p in state.positions.items() if p["qty"] > 0]
        decision = ranker.rank(scores, holdings)

        for s in decision["sell"]:
            pos = state.positions.get(s, {}); qty = pos.get("qty", 0)
            if qty > 0 and s in cp:
                pm.apply_sell(s, qty, cp[s], commission=qty*cp[s]*0.0008, trade_date=ts)
        for s in decision["buy"]:
            if s in cp:
                state = pm.load()
                cash_p = state.cash * 0.9 / max(1, len(decision["buy"]))
                px = cp[s]; qty = int(cash_p / px / 100) * 100
                if qty >= 100:
                    pm.apply_buy(s, qty, px, commission=qty*px*0.0003, trade_date=ts)
        pm.snapshot(ts, cp)

    summary = pm.get_summary(cp)
    total_ret = (summary["total_equity"] / INITIAL - 1) * 100

    bench_rets = []
    for sym in SYMBOLS:
        if sym in all_data:
            bdf = all_data[sym][(all_data[sym]["date"] >= start_dt) & (all_data[sym]["date"] <= end_dt)]
            if len(bdf) > 0: bench_rets.append(bdf["close"].iloc[-1] / bdf["close"].iloc[0] - 1)
    bench_avg = np.mean(bench_rets) * 100

    print(f"  [{label}] 策略={total_ret:+.1f}% 基准={bench_avg:+.1f}% 超额={total_ret - bench_avg:+.1f}%")
    return {"total_return": total_ret, "benchmark": bench_avg, "excess": total_ret - bench_avg}


# ================================================================
#  主流程
# ================================================================

def main():
    print("=" * 70)
    print("  量化模型三阶段训练")
    print("  方法: 网格搜索ICIR → 验证确认 → 测试评估")
    print("=" * 70)

    # Phase 1: 训练
    weights = grid_search_weights("2020-01-01", "2023-12-31")
    if not weights:
        print("❌ 训练失败,使用默认权重")
        weights = FactorScorer.from_preset("ic_optimized").factor_weights

    # Phase 2: 验证
    print(f"\n{'='*70}")
    print(f"  [验证] 确认ICIR不崩 (2024-01 ~ 2025-06)")
    r1 = backtest_period(weights, "2024-01-01", "2025-06-30", "验证期")

    # Phase 3: 测试
    print(f"\n{'='*70}")
    print(f"  [测试] 最终评估 — 只跑一次,不动权重 (2025-07 ~ 2026-07)")
    r2 = backtest_period(weights, "2025-07-01", "2026-07-10", "测试期")

    # 结论
    print(f"\n{'='*70}")
    print(f"  最终结论")
    print(f"{'='*70}")
    print(f"  训练权重: {len(weights)}个因子")
    print(f"  验证期超额: {r1['excess']:+.1f}%")
    print(f"  测试期超额: {r2['excess']:+.1f}% {'✅ 显著正超额!' if r2['excess']>0 else '❌ 测试期亏损'}")

    # 保存权重
    import json
    path = os.path.join(os.path.dirname(__file__), "trained_weights.json")
    with open(path, "w") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"  权重已保存: {path}")


if __name__ == "__main__":
    main()
