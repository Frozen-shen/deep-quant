"""
P5: 全因子组合验证 + Walk-Forward回测

将P1-P4验证通过的因子合并为统一alpha信号, 运行滚动回测并做统计检验。

注: 本脚本已取代 scripts/run_p5_portfolio_validation.py (该脚本已归档至 scripts/archive/, 不再使用)。

用法:
  py scripts/run_factor_portfolio.py              # 完整流程
  py scripts/run_factor_portfolio.py --skip-bt    # 跳过回测, 仅因子选择+独立性
  py scripts/run_factor_portfolio.py --fast       # 快速模式 (减少采样)

输出:
  data/ic_validation/p5_portfolio_report.json

成功标准 (Go/No-Go):
  - 年化超额 > 5%
  - IR > 0.5
  - MaxDD < 20%
  - Bootstrap 95% CI 不含 0
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
REPORT_PATH = os.path.join(IC_DIR, "p5_portfolio_report.json")

# 因子筛选阈值
THRESHOLDS = {
    "price_volume": {"min_abs_icir": 0.3, "label": "价量因子"},
    "fundamental": {"min_abs_icir": 0.15, "label": "基本面因子"},
    "event": {"min_abs_icir": 0.1, "label": "事件因子", "direction_consistent": True},
    "money_flow": {"skip": True, "label": "资金流因子 (历史不足, 跳过)"},
}

# 回测参数 (方案C v4: 仅限 research 期 2015-01~2024-12, 终极TEST/BLIND 禁入)
BT_CONFIG = {
    "start": "2019-01-01",       # 回测起始 (2018用于因子warmup)
    "end": "2024-12-31",         # 回测结束 (research 期内, 禁止触碰 TEST/BLIND)
    "rebalance_days": 20,        # 每20个交易日调仓
    "top_k": 30,                 # 持仓数量
    "ic_lookback_months": 12,    # IC估计回溯窗口
    "initial_capital": 1_000_000,  # 初始资金 (100万, 方便30只分仓)
    "lot_size": 100,
    "slippage_bps": 30,          # 滑点30bp
    "commission_buy": 0.00025,   # 万2.5
    "commission_sell": 0.00075,  # 万2.5 + 千0.5印花税
}

# 市场阶段划分 (仅 research 期)
REGIMES = [
    {"name": "bull_2019_2020", "start": "2019-01-01", "end": "2020-12-31"},
    {"name": "bear_2021_2022", "start": "2021-01-01", "end": "2022-12-31"},
    {"name": "recovery_2023_2024", "start": "2023-01-01", "end": "2024-12-31"},
]

# 事件因子低功效折扣
EVENT_POWER_DISCOUNT = 0.5

# 独立性检验
CORR_THRESHOLD = 0.4
CORR_SAMPLE_INTERVAL = 20  # 每20个交易日采样一次


# ════════════════════════════════════════════════════════════
#  Part 1: 因子选择
# ════════════════════════════════════════════════════════════

def load_ic_results() -> Dict[str, dict]:
    """
    加载所有IC验证结果。

    Returns:
      {factor_name: {"icir": float, "ic_mean": float, "n_days": int,
                     "category": str, "horizon": int, "pos_ratio": float}}
    """
    all_factors = {}

    # ── P3: Alpha158 价量因子 ──
    p3_path = os.path.join(IC_DIR, "p3_alpha158_ic.json")
    if os.path.exists(p3_path):
        with open(p3_path, "r", encoding="utf-8") as f:
            p3 = json.load(f)
        # 取每个因子在 horizon=20 的结果 (与回测label_horizon一致)
        for r in p3.get("results", []):
            if r["horizon"] == 20:
                name = r["factor"]
                all_factors[name] = {
                    "icir": r["icir"],
                    "ic_mean": r["ic_mean"],
                    "n_days": r["n_days"],
                    "category": "price_volume",
                    "horizon": 20,
                    "pos_ratio": r["pos_ratio"],
                    "abs_ic_mean": r["abs_ic_mean"],
                }
        print(f"  [P3] 加载 {sum(1 for v in all_factors.values() if v['category'] == 'price_volume')} 个价量因子", flush=True)

    # ── P4: 事件因子 ──
    p4_path = os.path.join(IC_DIR, "p4_analyst_event_ic.json")
    if os.path.exists(p4_path):
        with open(p4_path, "r", encoding="utf-8") as f:
            p4 = json.load(f)
        ic_by_factor = p4.get("ic_by_factor", {})
        for fname, horizons in ic_by_factor.items():
            # 取 h20 结果
            h20 = horizons.get("h20", {})
            if h20.get("status") == "insufficient_data":
                continue
            if "icir" not in h20:
                continue
            all_factors[fname] = {
                "icir": h20["icir"],
                "ic_mean": h20["ic_mean"],
                "n_days": h20["n_days"],
                "category": "event",
                "horizon": 20,
                "pos_ratio": h20["pos_ratio"],
                "abs_ic_mean": abs(h20["ic_mean"]),
            }
        print(f"  [P4] 加载 {sum(1 for v in all_factors.values() if v['category'] == 'event')} 个事件因子", flush=True)

    # ── P1: 基本面因子 (检查是否有IC验证结果) ──
    p1_path = os.path.join(IC_DIR, "p1_earnings_ic.json")
    if os.path.exists(p1_path):
        try:
            with open(p1_path, "r", encoding="utf-8") as f:
                p1 = json.load(f)
            # 格式可能与P3类似
            results = p1.get("results", p1.get("ic_by_factor", []))
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and "factor" in r and "icir" in r:
                        all_factors[r["factor"]] = {
                            "icir": r["icir"],
                            "ic_mean": r.get("ic_mean", 0),
                            "n_days": r.get("n_days", 0),
                            "category": "fundamental",
                            "horizon": r.get("horizon", 20),
                            "pos_ratio": r.get("pos_ratio", 0.5),
                            "abs_ic_mean": abs(r.get("ic_mean", 0)),
                        }
            elif isinstance(results, dict):
                for fname, val in results.items():
                    if isinstance(val, dict) and "icir" in val:
                        all_factors[fname] = {
                            "icir": val["icir"],
                            "ic_mean": val.get("ic_mean", 0),
                            "n_days": val.get("n_days", 0),
                            "category": "fundamental",
                            "horizon": 20,
                            "pos_ratio": val.get("pos_ratio", 0.5),
                            "abs_ic_mean": abs(val.get("ic_mean", 0)),
                        }
            n_fund = sum(1 for v in all_factors.values() if v['category'] == 'fundamental')
            print(f"  [P1] 加载 {n_fund} 个基本面因子", flush=True)
        except Exception as e:
            print(f"  [P1] 加载失败: {e}, 跳过基本面因子", flush=True)
    else:
        print(f"  [P1] 无IC验证文件 (fundamental_cache未构建), 跳过", flush=True)

    # ── 补充: 从 ic_results.json 加载 turnover_vol 等早期因子 ──
    ic_results_path = os.path.join(BASE_DIR, "data", "ic_results.json")
    if os.path.exists(ic_results_path):
        with open(ic_results_path, "r", encoding="utf-8") as f:
            ic_results = json.load(f)
        for r in ic_results:
            name = r["factor"]
            if name not in all_factors and abs(r["icir"]) > 0.3:
                all_factors[name] = {
                    "icir": r["icir"],
                    "ic_mean": r["ic_mean"],
                    "n_days": r["n_days"],
                    "category": "price_volume",
                    "horizon": 20,
                    "pos_ratio": r["pos_ratio"],
                    "abs_ic_mean": r["abs_ic_mean"],
                }

    return all_factors


def select_factors(all_factors: Dict[str, dict]) -> List[dict]:
    """
    按阈值筛选因子。

    Returns:
      通过筛选的因子列表 [{"name": ..., "icir": ..., "category": ..., "weight_multiplier": 1.0}, ...]
    """
    selected = []

    for name, info in all_factors.items():
        cat = info["category"]
        threshold = THRESHOLDS.get(cat, {})

        # 资金流跳过
        if threshold.get("skip", False):
            continue

        min_icir = threshold.get("min_abs_icir", 0.3)
        abs_icir = abs(info["icir"])

        if abs_icir < min_icir:
            continue

        # 事件因子: 检查方向一致性 (pos_ratio偏离0.5)
        if threshold.get("direction_consistent", False):
            pos_ratio = info.get("pos_ratio", 0.5)
            # 正IC因子: pos_ratio应>0.5; 负IC因子: pos_ratio应<0.5
            if info["icir"] > 0 and pos_ratio < 0.5:
                continue
            if info["icir"] < 0 and pos_ratio > 0.5:
                continue

        # 事件因子低功效折扣
        weight_mult = 1.0
        if cat == "event" and info["n_days"] < 30:
            weight_mult = EVENT_POWER_DISCOUNT

        selected.append({
            "name": name,
            "icir": info["icir"],
            "abs_icir": abs_icir,
            "ic_mean": info["ic_mean"],
            "category": cat,
            "n_days": info["n_days"],
            "pos_ratio": info.get("pos_ratio", 0.5),
            "weight_multiplier": weight_mult,
        })

    # 按 |ICIR| 降序排列
    selected.sort(key=lambda x: -x["abs_icir"])
    return selected


def print_factor_selection(selected: List[dict]):
    """打印因子选择结果。"""
    print(f"\n{'='*70}")
    print(f"  Part 1: 因子选择结果 — 共 {len(selected)} 个因子通过筛选")
    print(f"{'='*70}")
    print(f"  {'因子名':<25} {'类别':<12} {'ICIR':>8} {'IC均值':>10} {'天数':>6} {'权重':>6}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")
    for f in selected:
        cat_label = {"price_volume": "价量", "event": "事件", "fundamental": "基本面"}.get(f["category"], f["category"])
        mult_str = f"{f['weight_multiplier']:.1f}x" if f["weight_multiplier"] != 1.0 else "1.0x"
        print(f"  {f['name']:<25} {cat_label:<12} {f['icir']:>+8.3f} {f['ic_mean']:>+10.5f} {f['n_days']:>6} {mult_str:>6}")
    print(f"{'='*70}\n")


# ════════════════════════════════════════════════════════════
#  Part 2: 独立性检验
# ════════════════════════════════════════════════════════════

def check_independence(selected: List[dict], all_data: dict,
                       factor_cache, sample_interval: int = 20) -> Tuple[List[dict], dict]:
    """
    对选中因子做 pairwise Spearman 相关性检验, 剪枝高相关因子。

    Args:
      selected: 因子列表
      all_data: {symbol: DataFrame}
      factor_cache: FactorCache 实例
      sample_interval: 采样间隔 (交易日)

    Returns:
      (pruned_factors, correlation_info)
    """
    if len(selected) <= 1:
        return selected, {}

    print(f"\n{'='*70}")
    print(f"  Part 2: 独立性检验 (|corr| > {CORR_THRESHOLD} 剪枝)")
    print(f"{'='*70}")

    # 收集采样日期的截面因子值
    all_dates = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
    # 限制在回测期内采样
    bt_start = pd.Timestamp(BT_CONFIG["start"])
    bt_end = pd.Timestamp(BT_CONFIG["end"])
    sample_dates = [d for d in all_dates if bt_start <= d <= bt_end][::sample_interval]

    if len(sample_dates) < 10:
        print(f"  采样日不足 ({len(sample_dates)}), 跳过独立性检验", flush=True)
        return selected, {}

    print(f"  采样日期: {len(sample_dates)} 天", flush=True)

    factor_names = [f["name"] for f in selected]

    # 收集所有采样日的截面rank值
    # 对每个因子, 收集所有 (date, symbol) 的rank
    factor_ranks = {name: {} for name in factor_names}  # {name: {date: {symbol: rank}}}

    for di, today in enumerate(sample_dates):
        day_values = {name: {} for name in factor_names}
        for sym in all_data:
            feats = factor_cache.get(sym, today)
            if feats is None:
                continue
            for name in factor_names:
                if name in feats:
                    val = feats[name]
                    if not np.isnan(val):
                        day_values[name][sym] = val

        # 转为rank
        for name in factor_names:
            vals = day_values[name]
            if len(vals) < 30:
                continue
            syms = list(vals.keys())
            arr = np.array([vals[s] for s in syms])
            ranks = rankdata(arr)
            factor_ranks[name][today] = dict(zip(syms, ranks))

        if (di + 1) % 50 == 0:
            print(f"    进度: {di+1}/{len(sample_dates)}", flush=True)

    # 计算 pairwise Spearman 相关
    n_factors = len(factor_names)
    corr_matrix = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n_factors):
            name_i, name_j = factor_names[i], factor_names[j]
            # 收集公共日期的rank向量
            common_dates = set(factor_ranks[name_i].keys()) & set(factor_ranks[name_j].keys())
            if len(common_dates) < 5:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
                continue

            corrs = []
            for d in common_dates:
                ranks_i = factor_ranks[name_i][d]
                ranks_j = factor_ranks[name_j][d]
                common_syms = set(ranks_i.keys()) & set(ranks_j.keys())
                if len(common_syms) < 30:
                    continue
                syms = list(common_syms)
                ri = np.array([ranks_i[s] for s in syms])
                rj = np.array([ranks_j[s] for s in syms])
                c, _ = spearmanr(ri, rj)
                if not np.isnan(c):
                    corrs.append(c)

            avg_corr = np.mean(corrs) if corrs else 0.0
            corr_matrix[i, j] = avg_corr
            corr_matrix[j, i] = avg_corr

    # 贪心剪枝: 按|ICIR|降序, 遇到与已选因子相关>阈值的则剔除
    # selected 已按 |ICIR| 降序排列
    kept = []
    kept_indices = []
    pruned_info = {}

    for idx, f in enumerate(selected):
        should_prune = False
        for kept_idx in kept_indices:
            if abs(corr_matrix[idx, kept_idx]) > CORR_THRESHOLD:
                should_prune = True
                pruned_info[f["name"]] = {
                    "pruned_by": selected[kept_idx]["name"],
                    "correlation": float(corr_matrix[idx, kept_idx]),
                }
                break
        if not should_prune:
            kept.append(f)
            kept_indices.append(idx)

    # 打印结果
    print(f"\n  相关性矩阵 (Top-10 因子对):")
    pairs = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            pairs.append((abs(corr_matrix[i, j]), factor_names[i], factor_names[j], corr_matrix[i, j]))
    pairs.sort(reverse=True)
    for abs_c, n1, n2, c in pairs[:10]:
        marker = " ← 剪枝" if (n1 in pruned_info or n2 in pruned_info) else ""
        print(f"    {n1:<20} vs {n2:<20}: {c:+.3f}{marker}")

    if pruned_info:
        print(f"\n  剪枝结果:")
        for name, info in pruned_info.items():
            print(f"    ✂ {name} (与 {info['pruned_by']} 相关 {info['correlation']:+.3f})")
    else:
        print(f"\n  无因子被剪枝 (所有因子对 |corr| < {CORR_THRESHOLD})")

    print(f"  最终保留: {len(kept)} 个因子", flush=True)
    print(f"{'='*70}\n")

    # 构建相关性矩阵 dict 用于报告
    corr_dict = {}
    for i in range(min(n_factors, 15)):  # 只保存前15个
        for j in range(i + 1, min(n_factors, 15)):
            key = f"{factor_names[i]}_vs_{factor_names[j]}"
            corr_dict[key] = round(float(corr_matrix[i, j]), 4)

    return kept, {"correlation_matrix": corr_dict, "pruned": pruned_info}


# ════════════════════════════════════════════════════════════
#  Part 3: 复合评分
# ════════════════════════════════════════════════════════════

def compute_composite_scores(factors: List[dict], all_data: dict,
                             factor_cache, today) -> Dict[str, float]:
    """
    IC加权线性组合:
      composite_i = sum(ICIR_j * z_score(factor_j_i)) / sum(|ICIR_j|)

    负ICIR因子自动获得负权重 (做空信号方向)。

    Args:
      factors: 选中的因子列表
      all_data: {symbol: DataFrame}
      factor_cache: FactorCache
      today: 当前日期

    Returns:
      {symbol: composite_score}
    """
    factor_names = [f["name"] for f in factors]
    weights = np.array([f["icir"] * f["weight_multiplier"] for f in factors])
    abs_weight_sum = np.sum(np.abs(weights))
    if abs_weight_sum < 1e-9:
        return {}

    # 收集当日所有股票的因子值
    raw_values = {name: {} for name in factor_names}
    for sym in all_data:
        feats = factor_cache.get(sym, today)
        if feats is None:
            continue
        for name in factor_names:
            if name in feats and not np.isnan(feats[name]):
                raw_values[name][sym] = feats[name]

    # 找到所有因子都有值的股票
    valid_syms = None
    for name in factor_names:
        syms_set = set(raw_values[name].keys())
        if valid_syms is None:
            valid_syms = syms_set
        else:
            valid_syms = valid_syms & syms_set

    if valid_syms is None or len(valid_syms) < BT_CONFIG["top_k"]:
        return {}

    valid_syms = sorted(valid_syms)
    n = len(valid_syms)

    # 截面 z-score 并加权求和
    composite = np.zeros(n)
    for fi, name in enumerate(factor_names):
        vals = np.array([raw_values[name][s] for s in valid_syms])
        mean = vals.mean()
        std = vals.std()
        if std < 1e-9:
            continue
        z = (vals - mean) / std
        composite += weights[fi] * z

    composite /= abs_weight_sum

    return dict(zip(valid_syms, composite.tolist()))


# ════════════════════════════════════════════════════════════
#  Part 4: Walk-Forward 回测
# ════════════════════════════════════════════════════════════

def run_walkforward_backtest(factors: List[dict], all_data: dict,
                             factor_cache, fast: bool = False) -> dict:
    """
    Walk-Forward 回测:
      - 月度调仓 (每20个交易日)
      - Top-K=30 等权
      - T+1开盘价成交, 滑点30bp, 佣金万2.5+印花税
      - 基准: 全池等权

    Returns:
      回测结果 dict
    """
    print(f"\n{'='*70}")
    print(f"  Part 4: Walk-Forward 回测")
    print(f"{'='*70}")
    print(f"  期间: {BT_CONFIG['start']} ~ {BT_CONFIG['end']}")
    print(f"  调仓: 每{BT_CONFIG['rebalance_days']}交易日, Top-{BT_CONFIG['top_k']}等权")
    print(f"  成本: 滑点{BT_CONFIG['slippage_bps']}bp + 佣金万2.5 + 印花税千0.5")
    print(f"  因子: {len(factors)} 个")
    print(f"{'='*70}\n")

    # 收集交易日
    all_dates = sorted(set().union(*[set(df["date"].tolist()) for df in all_data.values()]))
    bt_start = pd.Timestamp(BT_CONFIG["start"])
    bt_end = pd.Timestamp(BT_CONFIG["end"])
    bt_dates = [d for d in all_dates if bt_start <= d <= bt_end]

    if len(bt_dates) < 100:
        print(f"  回测日期不足 ({len(bt_dates)}), 终止", flush=True)
        return {}

    print(f"  交易日: {len(bt_dates)} 天", flush=True)

    # ── 使用 SimpleBacktest 引擎 ──
    from model.engine import SimpleBacktest
    from trading_rules import TradingRules
    from portfolio_ranker import PortfolioRanker

    bt = SimpleBacktest(
        initial_capital=BT_CONFIG["initial_capital"],
        top_k=BT_CONFIG["top_k"],
        lot_size=BT_CONFIG["lot_size"],
        slippage_bps=BT_CONFIG["slippage_bps"],
        turnover_limit_pct=1.0,  # 回测中不限制换手
    )
    rules = TradingRules()
    ranker = PortfolioRanker(
        top_k=BT_CONFIG["top_k"],
        n_drop=BT_CONFIG["top_k"],  # 允许大幅换仓
        hold_thresh=1,              # 回测中不限制持有期
        sell_rank_buffer=0,
        buy_confirm_days=1,
        cost_threshold=0.0,         # 回测中不设成本门槛
    )

    # ── 逐日模拟 ──
    equity_curve = []
    daily_returns = []
    monthly_returns = []
    turnover_history = []
    rebalance_count = 0
    total_trades = 0
    pending_decision = None

    # 预计算每日收盘价 (用于mark_to_market)
    # 为性能, 只在需要时计算
    prev_equity = float(BT_CONFIG["initial_capital"])
    prev_month = None
    month_start_equity = prev_equity

    step = 2 if fast else 1  # fast模式跳过部分日期

    for di, today in enumerate(bt_dates):
        # ── T+1 执行: 昨天生成的信号今天执行 ──
        if pending_decision is not None:
            b, s, trades = bt.execute(pending_decision, today, all_data, rules)
            total_trades += b + s
            pending_decision = None

        # ── 调仓日: 生成新信号 ──
        if di % BT_CONFIG["rebalance_days"] == 0:
            scores = compute_composite_scores(factors, all_data, factor_cache, today)
            if scores and len(scores) >= BT_CONFIG["top_k"]:
                # 过滤不可交易的
                tradeable_scores = {}
                for sym, sc in scores.items():
                    if sym in all_data:
                        dt = all_data[sym][all_data[sym]["date"] <= today].tail(2)
                        if len(dt) >= 2 and not rules.is_suspended(sym, dt):
                            tradeable_scores[sym] = sc

                if len(tradeable_scores) >= BT_CONFIG["top_k"]:
                    holdings = list(bt.positions.keys())
                    decision = ranker.rank(tradeable_scores, holdings)
                    # 涨跌停过滤
                    decision["buy"] = [s for s in decision["buy"]
                                      if s in all_data and rules.can_buy(
                                          s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    decision["sell"] = [s for s in decision["sell"]
                                       if s in all_data and rules.can_sell(
                                           s, all_data[s][all_data[s]["date"] <= today].tail(2))]
                    pending_decision = decision
                    rebalance_count += 1

                    # 记录换手
                    n_sell = len(decision.get("sell", []))
                    n_buy = len(decision.get("buy", []))
                    turnover_history.append({
                        "date": str(today.date()),
                        "sells": n_sell,
                        "buys": n_buy,
                        "turnover": (n_sell + n_buy) / (2 * BT_CONFIG["top_k"]),
                    })

            if (rebalance_count % 6 == 0) and rebalance_count > 0:
                print(f"    调仓 #{rebalance_count}: {today.date()}, "
                      f"持仓={len(bt.positions)}, 交易={total_trades}笔", flush=True)

        # ── 每日mark-to-market ──
        close_prices = {}
        for sym in list(bt.positions.keys()):
            if sym in all_data:
                dt = all_data[sym][all_data[sym]["date"] <= today].tail(1)
                if len(dt) > 0:
                    close_prices[sym] = float(dt["close"].iloc[-1])

        equity = bt.mark_to_market(close_prices)
        equity_curve.append({"date": str(today.date()), "equity": equity})

        # 日收益率
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        daily_returns.append({"date": str(today.date()), "return": daily_ret})
        prev_equity = equity

        # 月收益率追踪
        month_key = today.strftime("%Y-%m")
        if prev_month is not None and month_key != prev_month:
            month_ret = (month_start_equity / equity - 1) if equity > 0 else 0
            # 修正: 用月初到月末
            monthly_returns.append({
                "month": prev_month,
                "return": (equity / month_start_equity - 1) if month_start_equity > 0 else 0,
            })
            month_start_equity = equity
        elif prev_month is None:
            month_start_equity = equity
        prev_month = month_key

        # 进度
        if (di + 1) % 200 == 0:
            print(f"    Day {di+1}/{len(bt_dates)}: equity={equity:,.0f}", flush=True)

    # 最后一个月
    if prev_month and monthly_returns and monthly_returns[-1]["month"] != prev_month:
        monthly_returns.append({
            "month": prev_month,
            "return": (prev_equity / month_start_equity - 1) if month_start_equity > 0 else 0,
        })

    # ── 基准: 全池等权 ──
    benchmark_returns = _calc_benchmark_returns(all_data, bt_dates)

    # ── 计算统计指标 ──
    equity_arr = np.array([e["equity"] for e in equity_curve])
    daily_ret_arr = np.array([r["return"] for r in daily_returns])

    total_return = (equity_arr[-1] / BT_CONFIG["initial_capital"] - 1)
    n_years = len(bt_dates) / 252.0
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # 基准
    bench_total = benchmark_returns.get("total_return", 0)
    bench_annual = benchmark_returns.get("annual_return", 0)
    excess_annual = annual_return - bench_annual

    # Sharpe (无风险利率=2.5%)
    rf_daily = 0.025 / 252
    excess_daily = daily_ret_arr - rf_daily
    sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(252) if np.std(excess_daily) > 0 else 0

    # IR (超额收益 / 跟踪误差)
    bench_daily = benchmark_returns.get("daily_returns", np.zeros(len(daily_ret_arr)))
    if len(bench_daily) == len(daily_ret_arr):
        active_returns = daily_ret_arr - bench_daily
        ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) > 0 else 0
    else:
        active_returns = daily_ret_arr
        ir = 0

    # 最大回撤
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Calmar
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    # 换手率
    avg_turnover = np.mean([t["turnover"] for t in turnover_history]) if turnover_history else 0
    # 成本拖累估算: 月换手 × 单边成本 × 12
    one_way_cost = (BT_CONFIG["slippage_bps"] / 10000 + BT_CONFIG["commission_buy"] + BT_CONFIG["commission_sell"]) / 2
    cost_drag_annual = avg_turnover * one_way_cost * 12 * 2  # 双边

    print(f"\n  回测结果:")
    print(f"    总收益: {total_return*100:+.1f}%")
    print(f"    年化收益: {annual_return*100:+.1f}%")
    print(f"    基准年化: {bench_annual*100:+.1f}%")
    print(f"    年化超额: {excess_annual*100:+.1f}%")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    IR: {ir:.2f}")
    print(f"    最大回撤: {max_drawdown*100:.1f}%")
    print(f"    Calmar: {calmar:.2f}")
    print(f"    月均换手: {avg_turnover*100:.1f}%")
    print(f"    年化成本拖累: {cost_drag_annual*100:.2f}%")
    print(f"    调仓次数: {rebalance_count}")
    print(f"    总交易: {total_trades}笔")

    return {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "benchmark_annual": round(bench_annual, 6),
        "excess_annual": round(excess_annual, 6),
        "sharpe": round(sharpe, 4),
        "ir": round(ir, 4),
        "max_drawdown": round(max_drawdown, 6),
        "calmar": round(calmar, 4),
        "monthly_turnover": round(avg_turnover, 4),
        "cost_drag": round(cost_drag_annual, 6),
        "rebalance_count": rebalance_count,
        "total_trades": total_trades,
        "n_days": len(bt_dates),
        "equity_curve": equity_curve[::20],  # 每20天采样, 减小文件
        "daily_returns": [r["return"] for r in daily_returns],
        "monthly_returns": monthly_returns,
        "turnover_history": turnover_history[::6],  # 每半年采样
        "_active_returns": active_returns.tolist() if len(active_returns) == len(daily_ret_arr) else [],
    }


def _calc_benchmark_returns(all_data: dict, bt_dates: list) -> dict:
    """计算全池等权基准收益。"""
    # 对每只股票计算期间收益, 取平均
    bt_start, bt_end = bt_dates[0], bt_dates[-1]
    stock_returns = []
    daily_bench = {}

    for sym, df in all_data.items():
        mask = (df["date"] >= bt_start) & (df["date"] <= bt_end)
        sub = df[mask]
        if len(sub) < 2:
            continue
        total_ret = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
        stock_returns.append(total_ret)

        # 日收益 (用于IR计算)
        sub_rets = sub["close"].pct_change().dropna()
        for d, r in zip(sub["date"].iloc[1:], sub_rets):
            d_str = str(d.date()) if hasattr(d, 'date') else str(d)
            if d_str not in daily_bench:
                daily_bench[d_str] = []
            daily_bench[d_str].append(r)

    # 等权日收益
    bench_daily_list = []
    for d_info in bt_dates:
        d_str = str(d_info.date()) if hasattr(d_info, 'date') else str(d_info)
        if d_str in daily_bench:
            bench_daily_list.append(np.mean(daily_bench[d_str]))
        else:
            bench_daily_list.append(0.0)

    bench_total = np.mean(stock_returns) if stock_returns else 0
    n_years = len(bt_dates) / 252.0
    bench_annual = (1 + bench_total) ** (1 / max(n_years, 0.1)) - 1 if bench_total > -1 else -1

    return {
        "total_return": bench_total,
        "annual_return": bench_annual,
        "daily_returns": np.array(bench_daily_list),
    }


# ════════════════════════════════════════════════════════════
#  Part 5: 统计验证
# ════════════════════════════════════════════════════════════

def statistical_validation(bt_result: dict) -> dict:
    """
    统计验证:
      1. Bootstrap: 重采样月收益1000次, 95% CI for IR
      2. 子样本: 按年分
      3. 市场阶段: bull/bear/recovery/recent
      4. 最大回撤 + Calmar
      5. 换手分析
    """
    print(f"\n{'='*70}")
    print(f"  Part 5: 统计验证")
    print(f"{'='*70}")

    validation = {}

    # ── 1. Bootstrap IR ──
    monthly_rets = [m["return"] for m in bt_result.get("monthly_returns", [])]
    if len(monthly_rets) >= 12:
        monthly_arr = np.array(monthly_rets)
        n_boot = 1000
        boot_irs = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            sample = rng.choice(monthly_arr, size=len(monthly_arr), replace=True)
            boot_ir = np.mean(sample) / np.std(sample) * np.sqrt(12) if np.std(sample) > 0 else 0
            boot_irs.append(boot_ir)

        boot_irs = np.array(boot_irs)
        ci_lower = float(np.percentile(boot_irs, 2.5))
        ci_upper = float(np.percentile(boot_irs, 97.5))
        ir_mean = float(np.mean(boot_irs))

        validation["bootstrap"] = {
            "n_samples": len(monthly_rets),
            "n_bootstrap": n_boot,
            "ir_mean": round(ir_mean, 4),
            "ir_ci_lower": round(ci_lower, 4),
            "ir_ci_upper": round(ci_upper, 4),
            "ci_excludes_zero": ci_lower > 0 or ci_upper < 0,
        }
        print(f"  Bootstrap IR: {ir_mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]", flush=True)
        print(f"  95% CI {'不含' if (ci_lower > 0 or ci_upper < 0) else '包含'} 0", flush=True)
    else:
        validation["bootstrap"] = {"error": "月收益样本不足"}
        print(f"  Bootstrap: 月收益样本不足 ({len(monthly_rets)}), 跳过", flush=True)

    # ── 2. 子样本 (按年) ──
    daily_rets = bt_result.get("daily_returns", [])
    equity_curve = bt_result.get("equity_curve", [])

    # 用equity_curve按年分段
    yearly_stats = []
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df["year"] = eq_df["date"].dt.year

        for year, group in eq_df.groupby("year"):
            if len(group) < 20:
                continue
            year_return = group["equity"].iloc[-1] / group["equity"].iloc[0] - 1
            yearly_stats.append({
                "period": str(year),
                "return": round(float(year_return), 6),
                "n_days": len(group),
            })

    validation["subsample"] = yearly_stats
    print(f"\n  子样本 (按年):")
    for ys in yearly_stats:
        marker = "+" if ys["return"] > 0 else ""
        print(f"    {ys['period']}: {marker}{ys['return']*100:.1f}%", flush=True)

    # ── 3. 市场阶段 ──
    regime_stats = []
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df["date"] = pd.to_datetime(eq_df["date"])

        for regime in REGIMES:
            r_start = pd.Timestamp(regime["start"])
            r_end = pd.Timestamp(regime["end"])
            mask = (eq_df["date"] >= r_start) & (eq_df["date"] <= r_end)
            sub = eq_df[mask]
            if len(sub) < 20:
                continue
            regime_return = sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1
            # 简化IR: 用日收益
            sub_rets = sub["equity"].pct_change().dropna()
            regime_ir = (np.mean(sub_rets) / np.std(sub_rets) * np.sqrt(252)
                        if np.std(sub_rets) > 0 else 0)
            regime_stats.append({
                "regime": regime["name"],
                "excess": round(float(regime_return), 6),
                "ir": round(float(regime_ir), 4),
                "n_days": len(sub),
            })

    validation["regime"] = regime_stats
    print(f"\n  市场阶段:")
    for rs in regime_stats:
        print(f"    {rs['regime']:<22}: return={rs['excess']*100:+.1f}%  IR={rs['ir']:+.2f}", flush=True)

    # ── 4. 回撤 + Calmar (已在Part4计算) ──
    validation["max_drawdown"] = bt_result.get("max_drawdown", 0)
    validation["calmar"] = bt_result.get("calmar", 0)

    # ── 5. 换手分析 ──
    validation["turnover"] = {
        "monthly_avg": bt_result.get("monthly_turnover", 0),
        "cost_drag_annual": bt_result.get("cost_drag", 0),
    }

    print(f"\n  最大回撤: {validation['max_drawdown']*100:.1f}%")
    print(f"  Calmar: {validation['calmar']:.2f}")
    print(f"  月均换手: {validation['turnover']['monthly_avg']*100:.1f}%")
    print(f"  年化成本拖累: {validation['turnover']['cost_drag_annual']*100:.2f}%")
    print(f"{'='*70}\n")

    return validation


# ════════════════════════════════════════════════════════════
#  Part 6: 判定 + 输出
# ════════════════════════════════════════════════════════════

def make_verdict(bt_result: dict, validation: dict) -> Tuple[str, List[str]]:
    """
    Go/No-Go 判定:
      - 年化超额 > 5%
      - IR > 0.5
      - MaxDD < 20%
      - Bootstrap CI 不含 0
    """
    reasons = []
    pass_count = 0
    total_checks = 4

    # 1. 年化超额
    excess = bt_result.get("excess_annual", 0)
    if excess > 0.05:
        pass_count += 1
        reasons.append(f"PASS: 年化超额 {excess*100:.1f}% > 5%")
    else:
        reasons.append(f"FAIL: 年化超额 {excess*100:.1f}% <= 5%")

    # 2. IR
    ir = bt_result.get("ir", 0)
    if ir > 0.5:
        pass_count += 1
        reasons.append(f"PASS: IR {ir:.2f} > 0.5")
    else:
        reasons.append(f"FAIL: IR {ir:.2f} <= 0.5")

    # 3. MaxDD
    mdd = abs(bt_result.get("max_drawdown", 1))
    if mdd < 0.20:
        pass_count += 1
        reasons.append(f"PASS: MaxDD {mdd*100:.1f}% < 20%")
    else:
        reasons.append(f"FAIL: MaxDD {mdd*100:.1f}% >= 20%")

    # 4. Bootstrap CI
    boot = validation.get("bootstrap", {})
    if boot.get("ci_excludes_zero", False):
        pass_count += 1
        reasons.append(f"PASS: Bootstrap 95% CI [{boot.get('ir_ci_lower', 0):.3f}, {boot.get('ir_ci_upper', 0):.3f}] 不含0")
    elif "error" in boot:
        reasons.append(f"SKIP: Bootstrap 样本不足")
    else:
        reasons.append(f"FAIL: Bootstrap 95% CI [{boot.get('ir_ci_lower', 0):.3f}, {boot.get('ir_ci_upper', 0):.3f}] 包含0")

    verdict = "PASS" if pass_count >= 3 else "FAIL"  # 至少3/4通过
    return verdict, reasons


def save_report(selected_factors: List[dict], corr_info: dict,
                bt_result: dict, validation: dict,
                verdict: str, verdict_reasons: List[str]):
    """保存完整报告到 JSON。"""
    report = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "P5 全因子组合验证报告",
            "bt_config": BT_CONFIG,
            "thresholds": {k: v for k, v in THRESHOLDS.items()},
        },
        "selected_factors": [
            {
                "name": f["name"],
                "icir": f["icir"],
                "category": f["category"],
                "weight_multiplier": f["weight_multiplier"],
                "n_days": f["n_days"],
            }
            for f in selected_factors
        ],
        "correlation_matrix": corr_info.get("correlation_matrix", {}),
        "pruned_factors": corr_info.get("pruned", {}),
        "backtest": {
            "total_return": bt_result.get("total_return", 0),
            "annual_return": bt_result.get("annual_return", 0),
            "benchmark_annual": bt_result.get("benchmark_annual", 0),
            "excess_annual": bt_result.get("excess_annual", 0),
            "sharpe": bt_result.get("sharpe", 0),
            "ir": bt_result.get("ir", 0),
            "max_drawdown": bt_result.get("max_drawdown", 0),
            "calmar": bt_result.get("calmar", 0),
            "monthly_turnover": bt_result.get("monthly_turnover", 0),
            "cost_drag": bt_result.get("cost_drag", 0),
            "rebalance_count": bt_result.get("rebalance_count", 0),
            "total_trades": bt_result.get("total_trades", 0),
        },
        "bootstrap": validation.get("bootstrap", {}),
        "subsample": validation.get("subsample", []),
        "regime": validation.get("regime", []),
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }

    # 移除内部字段
    for key in ["_active_returns", "daily_returns", "equity_curve", "monthly_returns", "turnover_history"]:
        bt_result.pop(key, None)

    os.makedirs(IC_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"  报告已保存: {REPORT_PATH}", flush=True)


def print_summary(selected_factors: List[dict], bt_result: dict,
                  validation: dict, verdict: str, verdict_reasons: List[str]):
    """打印格式化摘要。"""
    print(f"\n{'='*70}")
    print(f"  P5 全因子组合验证 — 最终报告")
    print(f"{'='*70}")
    print(f"  因子数: {len(selected_factors)}")
    print(f"  回测期间: {BT_CONFIG['start']} ~ {BT_CONFIG['end']}")
    print(f"  持仓: Top-{BT_CONFIG['top_k']} 等权, 月度调仓")
    print(f"{'='*70}")

    if bt_result:
        print(f"\n  {'指标':<20} {'值':>12} {'标准':>12} {'结果':>6}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*6}")
        print(f"  {'年化超额':<20} {bt_result.get('excess_annual',0)*100:>+11.1f}% {'>5%':>12} {'OK' if bt_result.get('excess_annual',0)>0.05 else 'NG':>6}")
        print(f"  {'IR':<20} {bt_result.get('ir',0):>12.2f} {'>0.5':>12} {'OK' if bt_result.get('ir',0)>0.5 else 'NG':>6}")
        print(f"  {'最大回撤':<20} {bt_result.get('max_drawdown',0)*100:>11.1f}% {'<20%':>12} {'OK' if abs(bt_result.get('max_drawdown',1))<0.2 else 'NG':>6}")
        print(f"  {'Sharpe':<20} {bt_result.get('sharpe',0):>12.2f}")
        print(f"  {'Calmar':<20} {bt_result.get('calmar',0):>12.2f}")
        print(f"  {'月均换手':<20} {bt_result.get('monthly_turnover',0)*100:>11.1f}%")
        print(f"  {'年化成本':<20} {bt_result.get('cost_drag',0)*100:>11.2f}%")

    print(f"\n  判定: {'PASS' if verdict == 'PASS' else 'FAIL'}")
    for r in verdict_reasons:
        print(f"    {r}")

    print(f"\n{'='*70}")
    print(f"  备注:")
    print(f"    - 资金流因子(P2)未纳入回测 (历史不足), 可作为实时叠加层")
    print(f"    - 基本面因子(P1)需构建fundamental_cache后纳入")
    print(f"    - 事件因子低功效(n_days<30)已给予0.5x权重折扣")
    print(f"{'='*70}\n")


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P5: 全因子组合验证")
    parser.add_argument("--skip-bt", action="store_true", help="跳过回测")
    parser.add_argument("--fast", action="store_true", help="快速模式")
    args = parser.parse_args()

    # ── gate 纪律: 只允许 research 期 (2015-01~2024-12), 禁止触碰 TEST/BLIND ──
    from gate import load_config, check_date_range, DateRangeGuard, GateViolation
    import os as _os
    _cfg = load_config(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))), "config.yaml"))
    _dp = _cfg["data_partition"]
    try:
        check_date_range(_dp["research"]["start"], _dp["research"]["end"],
                         _cfg, script_name="run_research_backtest")
    except GateViolation as ex:
        print(f"[GATE] {ex}", flush=True)
        return

    t0 = time.time()
    print("=" * 70, flush=True)
    print("  P5: 全因子组合验证 + Walk-Forward回测", flush=True)
    print("=" * 70, flush=True)

    # ── Part 1: 因子选择 ──
    print("\n[Part 1] 加载IC验证结果并筛选因子...", flush=True)
    all_factors = load_ic_results()
    selected = select_factors(all_factors)
    print_factor_selection(selected)

    if not selected:
        print("  无因子通过筛选, 终止。", flush=True)
        return

    # ── 加载数据 (Part 2-4 需要) ──
    if not args.skip_bt:
        print("[数据加载] 加载 data_cache/ 股票数据...", flush=True)
        from data_cache import get_cached_symbols, load_all
        syms = get_cached_symbols()
        print(f"  缓存股票: {len(syms)} 只", flush=True)
        all_data = load_all(syms)
        # 过滤: 至少100天数据
        all_data = {s: df for s, df in all_data.items() if df is not None and len(df) >= 100}
        print(f"  有效数据: {len(all_data)} 只", flush=True)

        # 预计算因子 (full_auto 包含所有价量因子: alpha158 + 早期因子)
        print("[因子预计算] 使用 full_auto preset (含alpha158+早期因子)...", flush=True)
        from factor_scorer import FactorScorer
        from factor_cache import FactorCache

        scorer = FactorScorer.from_preset("full_auto")
        factor_names = sorted(scorer.factor_weights.keys())
        factor_cache = FactorCache(scorer, factor_names)
        factor_cache.precompute(all_data)
        print(f"  预计算完成: {len(factor_names)} 个因子 × {len(all_data)} 只股票", flush=True)

        # 过滤: 只保留在factor_cache中可计算的因子
        computable = set(factor_names)
        selected_computable = [f for f in selected if f["name"] in computable]
        skipped = [f for f in selected if f["name"] not in computable]
        if skipped:
            print(f"  跳过不可计算因子: {[f['name'] for f in skipped]}", flush=True)
        selected = selected_computable
        print(f"  可计算因子: {len(selected)} 个", flush=True)

    # ── Part 2: 独立性检验 ──
    if not args.skip_bt and len(selected) > 1:
        sample_interval = 40 if args.fast else CORR_SAMPLE_INTERVAL
        selected, corr_info = check_independence(selected, all_data, factor_cache, sample_interval)
    else:
        corr_info = {"correlation_matrix": {}, "pruned": {}}

    if args.skip_bt:
        print("\n  --skip-bt: 跳过回测, 仅输出因子选择结果", flush=True)
        # 保存部分报告
        save_report(selected, corr_info, {}, {}, "SKIP", ["回测被跳过"])
        return

    # ── Part 3+4: 复合评分 + 回测 ──
    bt_result = run_walkforward_backtest(selected, all_data, factor_cache, fast=args.fast)

    if not bt_result:
        print("  回测失败, 终止。", flush=True)
        return

    # ── Part 5: 统计验证 ──
    validation = statistical_validation(bt_result)

    # ── Part 6: 判定 + 输出 ──
    verdict, verdict_reasons = make_verdict(bt_result, validation)
    save_report(selected, corr_info, bt_result, validation, verdict, verdict_reasons)
    print_summary(selected, bt_result, validation, verdict, verdict_reasons)

    elapsed = time.time() - t0
    print(f"  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
