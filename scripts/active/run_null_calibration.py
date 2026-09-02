"""
scripts/active/run_null_calibration.py — 空假设置换检验诊断 (多重检验校正)

目的 (2026-09-02):
  run_walkforward_backtest.py 的稳定因子筛选对 ~205 个候选因子独立施加
  "核心折 ≥3/5 显著" 条件。本脚本量化: 在完全没有 alpha 的空假设下,
  这套流程平均会"选出"多少个假稳定因子 (假发现数的蒙特卡洛估计)。

方法:
  - 复用主流程的 fold 划分 (FOLDS/FOLD_CORE_N) 与选择逻辑
    (compute_icir_weights / select_stable_factors / 逐折 BH), 不重新实现。
  - 对前瞻收益做截面内随机重排 (每个观测日把股票的前瞻收益打乱,
    因子值不变) → 因子与收益的关联被破坏, 其余数据结构 (截面相关性、
    时序波动、universe 过滤) 全部保留。
  - 重复 N 次, 统计每次选出的"稳定因子"数分布 (原门槛口径与双门槛口径)。
  - 反推"期望假发现数 < 1"对应的 FOLD_ICIR_MIN 参考基线 (二分搜索,
    不改动生产参数, 由人工决定是否采纳)。

注意:
  - 置换检验只使用训练窗数据做选择统计, 不跑验证期回测, 不触碰
    TEST/BLIND 分区 (无需 DateRangeGuard)。
  - 因子相关矩阵与有效检验数不依赖收益标签, 每个 fold 只算一次。

用法:
  py scripts/active/run_null_calibration.py --sample 300 --n-perm 50    # 冒烟
  py scripts/active/run_null_calibration.py --n-perm 200                # 正式

输出:
  data/ic_validation/null_calibration_report.json
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import get_logger
import run_walkforward_backtest as wfb
from stats_correction import (effective_num_tests, factor_correlation_matrix,
                              fdr_correction, t_summary_to_pvalue)

log = get_logger("null_calibration")

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
REPORT_PATH = os.path.join(IC_DIR, "null_calibration_report.json")


def _fold_offsets(calendar, cal_idx, train_start, train_end, val_first):
    """训练窗观测日索引列表 (与 compute_icir_weights 固定窗口口径一致)。"""
    s = wfb._nearest_idx(cal_idx, train_start)
    e = wfb._nearest_idx(cal_idx, train_end)
    t = cal_idx.get(val_first)
    if s is None or e is None or t is None:
        return []
    end = min(e - wfb.LABEL_HORIZON, t - wfb.LABEL_HORIZON)
    return [oi for oi in range(s, end + 1, wfb.IC_STEP)
            if 0 <= oi + wfb.LABEL_HORIZON < len(calendar)]


def _select_from_fold_stats(core_names, fold_stats_list, m_effs,
                            fdr_alpha, max_factors, null_override=None):
    """对一组 fold 的 ic_stats 应用主流程同款选择逻辑 (真实+置换通用)。

    fold_stats_list: [{factor: ic_stats}, ...] 每核心折一个
    m_effs: 每核心折的有效检验数 (标签无关, 预先算好)
    返回: (stable, stable_pre_fdr) — 双门槛与原门槛口径的稳定因子列表
    """
    factor_hits = {fn: 0 for fn in core_names}
    factor_icirs = {fn: [] for fn in core_names}
    factor_bh_hits = {fn: 0 for fn in core_names}

    for fi, ic_stats in enumerate(fold_stats_list):
        pvals = []
        for fn in core_names:
            st = ic_stats.get(fn)
            pvals.append(1.0 if st is None else t_summary_to_pvalue(
                st.get("ic_mean", 0.0), st.get("ic_std", 0.0),
                st.get("n_obs", 0)))
        for fn in core_names:
            st = ic_stats.get(fn)
            if st is not None:
                n_obs = st.get("n_obs", 0)
                t_stat = abs(st["icir"]) * np.sqrt(n_obs) if n_obs > 0 else 0.0
                factor_icirs[fn].append(st["icir"])
                if t_stat >= wfb.FOLD_T_STAT_MIN \
                        and abs(st["icir"]) >= wfb.FOLD_ICIR_MIN:
                    factor_hits[fn] += 1
            else:
                factor_icirs[fn].append(0.0)
        bh_pass = fdr_correction(pvals, alpha=fdr_alpha,
                                 m=max(1, m_effs[fi] or 1))
        for fn, ok in zip(core_names, bh_pass):
            if ok:
                factor_bh_hits[fn] += 1

    stable_pre, _, _ = wfb.select_stable_factors(
        core_names, factor_hits, factor_icirs, factor_bh_hits=None,
        min_hits=wfb.FOLD_MIN_HITS, icir_min=wfb.FOLD_ICIR_MIN,
        max_factors=max_factors, null_override=null_override)
    stable_post, _, _ = wfb.select_stable_factors(
        core_names, factor_hits, factor_icirs, factor_bh_hits=factor_bh_hits,
        min_hits=wfb.FOLD_MIN_HITS, icir_min=wfb.FOLD_ICIR_MIN,
        max_factors=max_factors, null_override=null_override)
    return stable_post, stable_pre


def main():
    parser = argparse.ArgumentParser(
        description="空假设置换检验: 稳定因子筛选的假发现率校准")
    parser.add_argument("--n-perm", type=int, default=200,
                        help="置换次数 (建议 >=200)")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样股票数 (加速诊断; 空假设下 t 统计量分布"
                             "与截面规模无关, 300-500 只足够)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fdr-alpha", type=float, default=None,
                        help="BH 目标 FDR (默认读 config fold.fdr_alpha)")
    parser.add_argument("--sweep-step", type=float, default=0.01,
                        help="期望假发现数<1 反推 ICIR 阈值的二分步长")
    args = parser.parse_args()

    from gate import load_config
    config = load_config(os.path.join(BASE_DIR, "config.yaml"))
    _fold_cfg = config.get("fold", {}) or {}
    fdr_alpha = args.fdr_alpha if args.fdr_alpha is not None \
        else float(_fold_cfg.get("fdr_alpha", wfb.FDR_ALPHA_DEFAULT))
    fdr_ve = float(_fold_cfg.get("fdr_var_explained", wfb.FDR_VAR_EXPLAINED))
    max_factors = int(_fold_cfg.get("max_factors", wfb.FOLD_MAX_FACTORS))

    log.info("=" * 60)
    log.info("  空假设置换检验 (n_perm=%d, fdr_alpha=%.2f)", args.n_perm, fdr_alpha)
    log.info("=" * 60)

    # ── 1. 数据加载 (与主流程同源: data_store 日线) ──
    log.info("加载数据...")
    from data_cache import get_cached_symbols, load
    syms = get_cached_symbols()
    if args.sample and args.sample < len(syms):
        import random
        random.seed(args.seed)
        syms = random.sample(syms, args.sample)
    all_data = {}
    for sym in syms:
        df = load(sym)
        if df is not None and len(df) >= 250:
            all_data[sym] = df
    log.info("  有效: %d 只", len(all_data))
    if len(all_data) < wfb.MIN_CROSS_SECTION + 20:
        log.error("股票数过少 (<%d), 无法做截面 IC", wfb.MIN_CROSS_SECTION + 20)
        sys.exit(1)

    calendar = wfb.build_calendar(all_data)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    log.info("  交易日历: %d 天 (%s ~ %s)", len(calendar),
             calendar[0].date(), calendar[-1].date())
    close_panel = wfb.build_close_panel(all_data, calendar)

    # ── 2. 因子面板 (主流程 v5 预设; 分钟/预期差因子数据依赖重, 诊断用价量+基本面) ──
    from factor_scorer import FactorScorer
    factor_names = sorted(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    needed_dates = set()
    for fold in wfb.FOLDS[:wfb.FOLD_CORE_N]:  # 只用核心折训练窗
        ts, te = fold["train"]
        vs, ve = fold["val"]
        for d in calendar:
            d_ = d.date()
            if pd.Timestamp(ts).date() <= d_ <= pd.Timestamp(ve).date():
                needed_dates.add(d)
    needed_dates = sorted(needed_dates)
    log.info("预计算因子面板: %d 因子 × %d 天...", len(factor_names), len(needed_dates))
    t0 = time.time()
    factor_panels = wfb.precompute_factor_panels(
        all_data, factor_names, needed_dates,
        include_fundamental=True, include_aux=False, include_minute=False)
    core_names = [fn for fn in factor_names if fn in factor_panels]
    log.info("  面板就绪: %d/%d 因子 (%.0fs)", len(core_names),
             len(factor_names), time.time() - t0)

    from data.pit_universe import get_universe

    # ── 3. 每核心折一次性预备: val_first / 观测日 / 相关矩阵 / 有效检验数 ──
    fold_prep = []
    for fi, fold in enumerate(wfb.FOLDS[:wfb.FOLD_CORE_N]):
        ts, te = fold["train"]
        vs, ve = fold["val"]
        val_first = None
        for d in calendar:
            if pd.Timestamp(vs).date() <= d.date() <= pd.Timestamp(ve).date():
                val_first = d
                break
        offsets = _fold_offsets(calendar, cal_idx, ts, te, val_first)
        corr_dates = [calendar[oi] for oi in offsets]
        corr_mat, corr_names = factor_correlation_matrix(
            factor_panels, corr_dates, core_names)
        m_eff = effective_num_tests(corr_mat, fdr_ve)
        fold_prep.append({"fold": fi + 1, "train": (ts, te),
                          "val_first": val_first, "n_obs": len(offsets),
                          "m_eff": m_eff, "n_corr_factors": len(corr_names)})
        log.info("  Fold %d: %d 观测日, 有效检验数 m_eff=%d/%d",
                 fi + 1, len(offsets), m_eff, len(corr_names))

    m_effs = [fp["m_eff"] for fp in fold_prep]

    def _run_once(perm_seed: int | None, null_override=None):
        """对全部核心折做一轮选择统计; perm_seed=None 为真实数据基线。"""
        fold_stats = []
        for fi, fp in enumerate(fold_prep):
            ts, te = fp["train"]
            _weights, ic_stats = wfb.compute_icir_weights(
                factor_panels, close_panel, calendar, cal_idx,
                fp["val_first"], core_names, train_start=ts, train_end=te,
                universe_fn=get_universe,
                permute_seed=(None if perm_seed is None
                              else perm_seed * 1000 + fi))
            fold_stats.append(ic_stats)
        return _select_from_fold_stats(core_names, fold_stats, m_effs,
                                       fdr_alpha, max_factors, null_override)

    # ── 4. 真实数据基线 (不置换) ──
    log.info("")
    log.info("真实数据基线 (无置换)...")
    t0 = time.time()
    real_post, real_pre = _run_once(None)
    log.info("  稳定因子: 双门槛 %d / 原门槛 %d (%.0fs)",
             len(real_post), len(real_pre), time.time() - t0)

    # ── 5. 置换循环 ──
    counts_post, counts_pre = [], []
    factor_freq = {}       # 空假设下每个因子被误选的频率 (双门槛口径)
    log.info("")
    log.info("开始 %d 次置换...", args.n_perm)
    t_start = time.time()
    for i in range(args.n_perm):
        t1 = time.time()
        stable_post, stable_pre = _run_once(args.seed + i + 1)
        counts_post.append(len(stable_post))
        counts_pre.append(len(stable_pre))
        for fn in stable_post:
            factor_freq[fn] = factor_freq.get(fn, 0) + 1
        if (i + 1) % 10 == 0 or i == 0:
            log.info("  perm %d/%d: 双门槛 %d, 原门槛 %d (%.0fs/轮, 剩余 ~%.0f min)",
                     i + 1, args.n_perm, len(stable_post), len(stable_pre),
                     time.time() - t1,
                     (time.time() - t1) * (args.n_perm - i - 1) / 60)

    arr_post = np.array(counts_post)
    arr_pre = np.array(counts_pre)

    # ── 6. 反推"期望假发现数 < 1"的 ICIR 阈值 (双门槛口径, 二分) ──
    # 用最后一轮置换缓存重跑代价高, 直接对每轮子样本不现实;
    # 采用逐阈值近似: 在真实+置换数据上复用 _run_once(null_override)
    # 太贵 → 只对双门槛口径做少量阈值的蒙特卡洛 (max_sweep 轮)。
    max_sweep = min(args.n_perm, 30)
    lo, hi = wfb.FOLD_ICIR_MIN, 0.5
    best_thr = hi
    log.info("")
    log.info("反推期望假发现数<1 的 ICIR 阈值 (二分, 每阈值 %d 轮置换)...",
             max_sweep)
    while hi - lo > args.sweep_step:
        mid = (lo + hi) / 2
        cnts = []
        for i in range(max_sweep):
            sp, _ = _run_once(args.seed + 10_000 + i,
                              null_override={"icir_min": mid})
            cnts.append(len(sp))
        mean_fd = float(np.mean(cnts))
        log.info("  ICIR>=%.3f: 平均假发现 %.2f", mid, mean_fd)
        if mean_fd < 1.0:
            best_thr = mid
            hi = mid
        else:
            lo = mid

    # ── 7. 报告 ──
    top_noise = sorted(factor_freq.items(), key=lambda kv: -kv[1])[:20]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": ("空假设置换检验: 每个观测日截面内随机重排前瞻收益, "
                        "统计稳定因子筛选的假发现数分布"),
        "config": {
            "n_permutations": args.n_perm,
            "seed": args.seed,
            "n_stocks": len(all_data),
            "n_candidates": len(core_names),
            "fdr_alpha": fdr_alpha,
            "fdr_var_explained": fdr_ve,
            "fold_min_hits": wfb.FOLD_MIN_HITS,
            "fold_icir_min": wfb.FOLD_ICIR_MIN,
            "fold_t_stat_min": wfb.FOLD_T_STAT_MIN,
            "fold_core_n": wfb.FOLD_CORE_N,
            "max_factors": max_factors,
            "folds": [{"fold": fp["fold"], "train": fp["train"],
                       "n_obs": fp["n_obs"], "m_eff": fp["m_eff"],
                       "n_corr_factors": fp["n_corr_factors"]}
                      for fp in fold_prep],
        },
        "real_data_baseline": {
            "stable_factors_dual_gate": len(real_post),
            "stable_factors_pre_fdr": len(real_pre),
            "stable_factor_names": sorted(real_post),
        },
        "null_distribution": {
            "dual_gate": {
                "mean": float(arr_post.mean()),
                "median": float(np.median(arr_post)),
                "p95": float(np.percentile(arr_post, 95)),
                "max": int(arr_post.max()),
                "pct_zero": float((arr_post == 0).mean() * 100),
            },
            "pre_fdr": {
                "mean": float(arr_pre.mean()),
                "median": float(np.median(arr_pre)),
                "p95": float(np.percentile(arr_pre, 95)),
                "max": int(arr_pre.max()),
                "pct_zero": float((arr_pre == 0).mean() * 100),
            },
            "raw_counts": {"dual_gate": counts_post, "pre_fdr": counts_pre},
        },
        "top_null_factors": [
            {"factor": fn, "selected_pct": round(100.0 * c / args.n_perm, 1)}
            for fn, c in top_noise],
        "icir_threshold_for_efd_lt_1": {
            "value": round(best_thr, 3),
            "current_fold_icir_min": wfb.FOLD_ICIR_MIN,
            "method": ("二分搜索 (步长 %.3f, 每阈值 %d 轮置换): 最小使"
                       "平均假发现数<1 的 |中位数ICIR| 入选阈值。近似说明: "
                       "逐折 hits 未随阈值重算, 仅叠加中位数门槛 → 估计偏"
                       "严格 (阈值偏高)。是否采纳由人工决定"
                       % (args.sweep_step, max_sweep)),
        },
        "interpretation": {
            "note": ("real_data_baseline.stable_factors_dual_gate 显著高于 "
                     "null_distribution.dual_gate.p95 → 真实信号超出噪音基线; "
                     "接近或低于 → 筛选结果可能主要由多重比较假阳性驱动"),
        },
        "runtime_s": round(time.time() - t_start, 1),
    }
    os.makedirs(IC_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 8. 实验登记 ──
    try:
        from experiment_tracker import log_experiment
        log_experiment(
            script_name="run_null_calibration",
            partition="research",
            config={"n_perm": args.n_perm, "n_stocks": len(all_data),
                    "fdr_alpha": fdr_alpha, "sample": args.sample},
            results={
                "null_mean_dual_gate": report["null_distribution"]["dual_gate"]["mean"],
                "null_p95_dual_gate": report["null_distribution"]["dual_gate"]["p95"],
                "real_stable_dual_gate": len(real_post),
                "icir_threshold_efd_lt_1": best_thr,
            },
            notes="空假设置换检验诊断 (不改变生产参数)",
            experiments_dir=os.path.join(BASE_DIR, "experiments"),
        )
    except Exception as e:
        log.warning("experiment_tracker 登记失败: %s", e)

    # ── 9. 汇总 ──
    log.info("")
    log.info("=" * 60)
    log.info("  诊断结果")
    log.info("=" * 60)
    log.info("  真实数据: 稳定因子 %d (双门槛) / %d (原门槛)",
             len(real_post), len(real_pre))
    log.info("  空假设分布 (双门槛): 均值 %.2f, 中位 %.0f, P95 %.0f, 最大 %d",
             arr_post.mean(), np.median(arr_post),
             np.percentile(arr_post, 95), arr_post.max())
    log.info("  空假设分布 (原门槛): 均值 %.2f, P95 %.0f",
             arr_pre.mean(), np.percentile(arr_pre, 95))
    log.info("  期望假发现数<1 的 |ICIR| 阈值: %.3f (现行 %.3f)",
             best_thr, wfb.FOLD_ICIR_MIN)
    verdict = "真实信号显著超出噪音基线" \
        if len(real_post) > np.percentile(arr_post, 95) \
        else "⚠️ 真实结果未超出噪音基线, 需谨慎解读"
    log.info("  结论: %s", verdict)
    log.info("  报告: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
