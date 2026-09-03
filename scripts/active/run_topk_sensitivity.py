"""
scripts/active/run_topk_sensitivity.py — top_k 敏感性实验 (组合层复用权重)

动机 (2026-08-13): 用户资金 10-30 万, top_k 需要在整手+最低佣金约束下
  每笔订单过小 (v24e: 86% 订单 <5000 元, 中位 3024 元, 实际佣金率 0.17%,
  是账面 0.025% 的 7 倍)。目标: 用数据决定 5/10/15 哪个持仓数更优。

方法: top_k 不影响因子 ICIR 权重 (权重只由训练期 IC 决定) → 每个 fold
  训练期权重只算一次, 验证期组合层对 top_k ∈ {5,10,15} 各跑一遍
  (复用 run_walkforward_backtest 的 compute_icir_weights/run_backtest)。

用法:
  py scripts/active/run_topk_sensitivity.py                    # 全量 5/10/15
  py scripts/active/run_topk_sensitivity.py --topks 5,10,15    # 自定义
  py scripts/active/run_topk_sensitivity.py --sample 50        # 冒烟 (快)

输出: data/ic_validation/topk_sensitivity.json
"""

import os
import sys
import json
import copy
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from logger import get_logger
from gate import load_config, check_date_range, GateViolation

# 复用主回测模块的已验证函数 (不复制逻辑, 防止分叉)
import run_walkforward_backtest as wf

log = get_logger("topk_sensitivity")
IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
OUTPUT_PATH = os.path.join(IC_DIR, "topk_sensitivity.json")

DEFAULT_TOPKS = [5, 10, 15]


def load_config_sections(config: dict) -> dict:
    """读取与 v24e 一致的实验配置段 (保证敏感性实验同构, 只变 top_k)。"""
    _neut = config.get("neutralization", {}) or {}
    _min_cfg = config.get("minute_factors", {}) or {}
    ml_cfg = config.get("minute_layer", {})
    return {
        "neutralize_enabled": bool(_neut.get("enabled", False)),
        "neutralize_k": float(_neut.get("winsorize_k", 3.0)),
        "industry_neutral": bool(_neut.get("industry_neutral", False)),
        "minute_enabled": bool(_min_cfg.get("enabled", False)),
        "minute_lookback": int(_min_cfg.get("lookback", 20)),
        "portfolio_constraints": config.get("portfolio_constraints") or None,
        "minute_layer_cfg": ml_cfg,
        "weight_mode": str(config.get("portfolio_optimizer", "equal")),
        "pool_filter_cfg": config.get("pool_filter"),
        "vol_target_cfg": config.get("vol_target"),
        "trend_timing_cfg": config.get("trend_timing"),
        "max_factors": int(config.get("fold", {}).get("max_factors", 40)),
    }


def main():
    parser = argparse.ArgumentParser(description="top_k 敏感性实验 (复用fold权重)")
    parser.add_argument("--topks", type=str, default=",".join(map(str, DEFAULT_TOPKS)),
                        help="逗号分隔的 top_k 列表")
    parser.add_argument("--sample", type=int, default=None, help="抽样股票数 (冒烟)")
    parser.add_argument("--max-stocks", type=int, default=None, help="最多股票数")
    parser.add_argument("--liquid", action="store_true",
                        help="流动性 PIT universe (默认开启, 与 v24e 一致)")
    args = parser.parse_args()

    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    if not topks:
        log.error("无效 --topks: %s", args.topks)
        sys.exit(1)

    config = load_config(os.path.join(BASE_DIR, "config.yaml"))
    sec = load_config_sections(config)

    # ── 日期守卫 (fold 验证期 2020-2024 均在 research 分区内) ──
    try:
        check_date_range("2020-01-01", "2024-12-31",
                         config, script_name="run_topk_sensitivity")
    except GateViolation as ex:
        log.error(str(ex))
        sys.exit(1)

    # universe
    from data.pit_universe import get_liquid_universe
    universe_fn = get_liquid_universe if args.liquid else wf.get_universe

    log.info("=" * 60)
    log.info("  top_k 敏感性实验: %s (v24e 同构, 仅组合层 top_k 变化)",
             topks)
    log.info("  universe: %s", "流动性PIT(全市场+过滤)" if args.liquid else "指数成分")
    log.info("=" * 60)

    bt_config = wf.load_bt_config()
    log.info("  基线 bt_config: top_k=%d (将覆盖), lot=%d, 调仓=%dd",
             bt_config["top_k"], bt_config["lot_size"], bt_config["rebalance_days"])

    # ── 1. 加载数据 (与 main 一致) ──
    log.info("加载数据...")
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if args.sample and args.sample < len(syms):
        import random
        random.seed(42)
        syms = random.sample(syms, args.sample)
    elif args.max_stocks and args.max_stocks < len(syms):
        syms = syms[:args.max_stocks]
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))

    calendar = wf.build_calendar(all_data)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    log.info("  交易日历: %d 天 (%s ~ %s)", len(calendar),
             calendar[0].date(), calendar[-1].date())

    close_panel = wf.build_close_panel(all_data, calendar)

    # ── 2. 因子名单 (full_auto_v5 + 分钟因子, 与 v24e fold 模式一致) ──
    from factor_scorer import FactorScorer
    factor_names = sorted(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    if sec["minute_enabled"]:
        from minute_factors import get_minute_factor_names
        factor_names = sorted(set(factor_names) | set(get_minute_factor_names()))
    log.info("  因子数: %d (含分钟 %s)", len(factor_names),
             "开" if sec["minute_enabled"] else "关")

    # ── 3. 面板所需日期 (fold 训练期 + 验证期) ──
    needed = set()
    for fold in wf.FOLDS:
        ts, te = fold["train"]
        vs, ve = fold["val"]
        for d in calendar:
            d_ = d.date()
            if pd.Timestamp(ts).date() <= d_ <= pd.Timestamp(te).date():
                needed.add(d)
            elif pd.Timestamp(vs).date() <= d_ <= pd.Timestamp(ve).date():
                needed.add(d)
    needed_dates = sorted(needed)

    # ── 4. 预计算因子面板 (一次, 与 main 同参数) ──
    log.info("预计算因子面板...")
    t0 = time.time()
    factor_panels = wf.precompute_factor_panels(
        all_data, factor_names, needed_dates,
        include_fundamental=True,
        include_aux=True,
        include_minute=sec["minute_enabled"],
        minute_lookback=sec["minute_lookback"],
        neutralize_enabled=sec["neutralize_enabled"],
        neutralize_k=sec["neutralize_k"],
        industry_map=wf._load_industry_map() if sec["industry_neutral"] else None)
    log.info("  面板就绪: %d 因子 (%ds)", len(factor_panels),
             int(time.time() - t0))

    # ── 5. 分钟叠加层 (fold 4-5 训练期, 与 main 一致) ──
    ml_weights = None
    ml_lambda = 0.3
    ml_cfg = sec["minute_layer_cfg"]
    if sec["minute_enabled"] and ml_cfg.get("enabled"):
        train_folds = [("2022-01-01", "2023-12-31"),
                       ("2022-01-01", "2024-12-31")]
        ml_weights = wf.validate_minute_factors(
            factor_panels, close_panel, calendar, cal_idx,
            factor_names, train_folds,
            min_icir=float(ml_cfg.get("min_icir", 0.3)))
        ml_lambda = float(ml_cfg.get("lambda", 0.3))
        if ml_weights:
            log.info("  分钟叠加层: %d 个因子, λ=%.2f (fold 4-5 验证期)",
                     len(ml_weights), ml_lambda)

    # ── 6. 逐 fold: 权重算一次 × 每个 top_k 跑验证期回测 ──
    log.info("=" * 60)
    log.info("  逐 fold 回测 (权重复用, 组合层 × %d top_k)", len(topks))
    log.info("=" * 60)

    summary = {tk: {"fold_results": {}, "trades_stats": {}} for tk in topks}
    t_start = time.time()

    for fi, fold in enumerate(wf.FOLDS):
        ts, te = fold["train"]
        vs, ve = fold["val"]
        log.info("")
        log.info("── Fold %d: Train %s~%s → Val %s~%s ──",
                 fi + 1, ts, te, vs, ve)

        # 验证期首日
        val_first = None
        for d in calendar:
            if pd.Timestamp(vs).date() <= d.date() <= pd.Timestamp(ve).date():
                val_first = d
                break
        if val_first is None:
            log.warning("  Fold %d: 验证期无交易日, 跳过", fi + 1)
            continue

        # 训练期权重 — 只算一次 (top_k 无关)
        weights, ic_stats = wf.compute_icir_weights(
            factor_panels, close_panel, calendar, cal_idx,
            val_first, factor_names, train_start=ts, train_end=te,
            universe_fn=universe_fn)
        n_sel = len(weights)
        log.info("  训练期因子: %d/%d 入选 |ICIR|>=%.2f",
                 n_sel, len(factor_names), wf.FOLD_ICIR_MIN)
        if not weights:
            log.warning("  Fold %d: 无因子达标, 验证期跳过", fi + 1)
            continue

        # 验证期: 每个 top_k 跑一遍 (同一权重)
        for tk in topks:
            bt_cfg = copy.deepcopy(bt_config)
            bt_cfg["top_k"] = tk
            log.info("  [K%d] 验证期 %s ~ %s 回测...", tk, vs, ve)
            r = wf.run_backtest(
                all_data, factor_panels, close_panel, calendar, cal_idx,
                factor_names, bt_cfg, vs, ve, label=f"K{tk}V{fi+1}",
                fixed_weights=weights, universe_fn=universe_fn,
                use_regime=True,
                portfolio_constraints=sec["portfolio_constraints"],
                minute_weights=ml_weights if fi >= 3 else None,
                minute_lambda=ml_lambda,
                weight_mode=sec["weight_mode"],
                pool_filter_cfg=sec["pool_filter_cfg"],
                vol_target_cfg=sec["vol_target_cfg"],
                trend_timing_cfg=sec["trend_timing_cfg"])
            if not r:
                log.warning("  [K%d] fold %d 回测无结果", tk, fi + 1)
                continue
            summary[tk]["fold_results"][f"fold_{fi+1}"] = {
                "train": f"{ts}~{te}", "val": f"{vs}~{ve}",
                "excess_annual": r["excess_annual"],
                "annual_return": r["annual_return"],
                "sharpe": r["sharpe"], "ir": r["ir"],
                "max_drawdown": r["max_drawdown"],
                "avg_turnover": r["avg_turnover"],
                "n_rebalances": r["n_rebalances"],
            }
            # 订单金额统计 (整手+佣金约束下的真实可执行性)
            trades = r.get("trades", [])
            if trades:
                amts = [t.get("qty", 0) * t.get("price", 0) for t in trades]
                amts = [a for a in amts if a > 0]
                if amts:
                    arr = np.array(amts)
                    comms = np.array([t.get("commission", 0) for t in trades])
                    summary[tk]["trades_stats"][f"fold_{fi+1}"] = {
                        "n_trades": len(trades),
                        "median_amt": round(float(np.median(arr)), 0),
                        "pct_lt_5000": round(float((arr < 5000).mean() * 100), 1),
                        "pct_lt_10000": round(float((arr < 10000).mean() * 100), 1),
                        "mean_comm_ratio": round(
                            float((comms / arr).mean() * 100), 3),
                    }

    # ── 7. 汇总对比表 ──
    log.info("")
    log.info("=" * 68)
    log.info("  top_k 敏感性汇总 (5-fold 验证期均值)")
    log.info("  " + "-" * 66)
    log.info("  %-6s %9s %8s %7s %6s %6s %8s %10s",
             "top_k", "超额年化", "Sharpe", "MaxDD", "IR", "换手",
             "中位订单", "佣金占比")
    log.info("  " + "-" * 66)
    rows = []
    for tk in topks:
        fr = summary[tk]["fold_results"]
        if not fr:
            continue
        excess = np.mean([v["excess_annual"] for v in fr.values()])
        sharpe = np.mean([v["sharpe"] for v in fr.values()])
        mdd = np.mean([v["max_drawdown"] for v in fr.values()])
        ir = np.mean([v["ir"] for v in fr.values()])
        to = np.mean([v["avg_turnover"] for v in fr.values()])
        ts_stats = summary[tk]["trades_stats"]
        meds = [v["median_amt"] for v in ts_stats.values()]
        crs = [v["mean_comm_ratio"] for v in ts_stats.values()]
        med_amt = float(np.median(meds)) if meds else 0
        comm_ratio = float(np.mean(crs)) if crs else 0
        rows.append({"top_k": tk, "excess": excess, "sharpe": sharpe,
                     "mdd": mdd, "ir": ir, "turnover": to,
                     "median_amt": med_amt, "comm_ratio": comm_ratio})
        log.info("  %-6d %+8.1f%% %8.2f %6.1f%% %6.2f %6.1f%% %9.0f %9.3f%%",
                 tk, excess, sharpe, mdd, ir, to, med_amt, comm_ratio)
    log.info("  " + "-" * 66)
    log.info("  (佣金占比=实付佣金/订单金额均值, 含最低5元约束)")

    # 存档
    out = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topks": topks,
            "n_stocks": len(all_data),
            "n_factors": len(factor_names),
            "universe": "liquid" if args.liquid else "index",
            "baseline_top_k": bt_config["top_k"],
            "baseline_ref": "walkforward_results_v24e_pov.json",
            "method": "fold 权重复用 (训练期ICIR一次, 组合层top_k各一遍)",
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        },
        "summary_rows": rows,
        "details": summary,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("")
    log.info("  结果: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
