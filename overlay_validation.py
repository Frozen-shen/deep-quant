"""
overlay_validation.py — Overlay 层参数的 5 折统计验证框架 (技术方案 3.3, P4)

问题: regime_detector (profile 乘数)、vol_target、trend_timing、pool_filter、
styles(sleeve) 等 overlay 模块的启用/停用/参数选择, 历史上靠"跑一次
extend-val 看数字变好/变差"决定, 构成核心因子 BH-FDR 校正覆盖不到的另一个
多重比较入口 (反复对着同一模拟考窗口调参)。本框架把这些 overlay 纳入与核心
因子同级的验证纪律: 核心折 (FOLD_CORE_N=5) 上逐折跑"基线 vs 候选参数",
再按下面规则判定:

判定规则 (与 DEVELOPMENT_DISCIPLINE 第十一条"禁止变体锦标赛"对齐):
  1. 方向一致性: 候选在某折的 excess 相对基线同方向改进 ≥3/5 折 (可调) —
     单折大涨不算数, 必须跨折一致;
  2. 统计显著性: 5 折配对差值 (excess_候选 − excess_基线) 的单样本 t 检验
     p 值; 多个候选并行比较时用 stats_correction.fdr_correction() (BH) 做
     多重比较校正 (沿用 2026-09-02 稳定因子筛选的校正工具);
  3. 判定: 同时满足一致性 + BH 后显著 (p ≤ alpha) + 平均差值 > 0 才 accept;
     否则 reject (当前生产参数未通过时只如实报告, 不要求立即更换)。

用法:
  from overlay_validation import validate_overlay_candidates
  res = validate_overlay_candidates(
      overlay_name="regime.profile",
      base_excess_by_fold={fold: float, ...},        # 基线 (如 disabled 无 overlay)
      cand_excess_by_fold={"conservative": {fold: float, ...}, ...},
      alpha=0.10, min_consistent_folds=3, n_folds=5)

  # 从各候选各折结果矩阵直接算 (runner 常用):
  from overlay_validation import validate_from_matrix
  res = validate_from_matrix("pool_filter", matrix, base_label="disabled")

被引用方 (模块准入): scripts/active/run_regime_robustness.py (--folds 模式)
与本文件单元测试。任何 overlay 的正式验证都通过本模块出结论, 结果写入
experiment_tracker.log_experiment() 留痕。
"""
import numpy as np
from scipy import stats

from stats_correction import fdr_correction

DEFAULT_ALPHA = 0.10          # 与 fold.fdr_alpha 一致
DEFAULT_MIN_CONSISTENT = 3    # ≥3/5 折同方向 (与 FOLD_MIN_HITS 同精神)
DEFAULT_N_FOLDS = 5           # 核心折数 (FOLD_CORE_N)


def paired_fold_deltas(base_by_fold: dict, cand_by_fold: dict) -> dict:
    """候选相对基线的逐折 excess 差值 (仅取两方都有的折)。

    Returns:
      {"deltas": {fold_label: delta_pp, ...}, "folds": [...], }
      空 folds → 判定 "insufficient"。
    """
    folds = sorted(set(base_by_fold) & set(cand_by_fold),
                   key=lambda f: str(f))
    deltas = {f: float(cand_by_fold[f]) - float(base_by_fold[f])
              for f in folds}
    return {"deltas": deltas, "folds": folds}


def _verdict(consistent: int, n: int, p: float, delta_mean: float,
             alpha: float, min_consistent: int, fdr_sig: bool) -> str:
    if n == 0:
        return "insufficient"
    # fdr_sig: BH 校正后仍显著 (fdr_correction True = 拒绝 H0 = 显著)
    if consistent >= min_consistent and delta_mean > 0 and fdr_sig:
        return "accept"
    if consistent >= min_consistent and delta_mean > 0:
        return "reject_not_significant"   # 方向一致但样本上不显著
    return "reject_not_consistent"        # 跨折方向不一致


def _candidate_pvalue(cand_by_fold: dict, base_by_fold: dict):
    """5 折配对差值的单样本 t 检验 (H0: 平均差值=0)。"""
    pd_ = paired_fold_deltas(base_by_fold, cand_by_fold)
    vals = list(pd_["deltas"].values())
    if len(vals) < 2:
        return 1.0, float(np.mean(vals)) if vals else 0.0, pd_
    t, p = stats.ttest_1samp(vals, 0.0)
    if not np.isfinite(p):
        p = 1.0
    return float(p), float(np.mean(vals)), pd_


def validate_overlay_candidates(
        overlay_name: str,
        base_excess_by_fold: dict,
        cand_excess_by_fold: dict,
        alpha: float = DEFAULT_ALPHA,
        min_consistent_folds: int = DEFAULT_MIN_CONSISTENT,
        n_folds: int = DEFAULT_N_FOLDS) -> dict:
    """对一组 overlay 候选参数出统一判定。

    Args:
      overlay_name: 如 "regime.profile" / "vol_target.enabled"
      base_excess_by_fold: 基线 (无该 overlay 或生产对照) 每折 excess_annual
      cand_excess_by_fold: {候选名: {fold: excess_annual}}
      alpha: BH 目标水平
      min_consistent_folds: 同方向改进的最少折数
      n_folds: 核心折总数 (用于记录)
    Returns:
      {"overlay", "alpha", "min_consistent_folds", "n_folds",
       "base": {fold: excess}, "candidates": {候选: {...判定明细...}},
       "verdicts": {候选: accept/reject_*}}
    """
    out = {
        "overlay": overlay_name,
        "method": ("5折配对检验: 方向一致性≥%d/%d + 单样本t检验 + BH-FDR"
                   % (min_consistent_folds, n_folds)),
        "alpha": float(alpha),
        "min_consistent_folds": int(min_consistent_folds),
        "n_folds": int(n_folds),
        "base": {str(k): float(v) for k, v in base_excess_by_fold.items()},
        "candidates": {},
        "verdicts": {},
    }
    cands = {k: v for k, v in cand_excess_by_fold.items()
             if isinstance(v, dict) and v}
    pvals = {}
    for name, c in cands.items():
        p, mean_delta, pdetail = _candidate_pvalue(c, base_excess_by_fold)
        deltas = pdetail["deltas"]
        n = len(deltas)
        consistent = sum(1 for d in deltas.values() if d > 0) if n else 0
        pvals[name] = p
        out["candidates"][name] = {
            "fold_deltas_pp": {str(k): round(float(v), 3)
                               for k, v in deltas.items()},
            "consistent_folds": int(consistent),
            "n_folds_compared": int(n),
            "mean_delta_pp": round(float(mean_delta), 3),
            "p_value": round(float(p), 5),
        }
    # BH 校正 (跨候选; True = 拒绝 H0 = 校正后仍显著; 单候选时与原始 p 等价)
    names = list(pvals.keys())
    if names:
        sig = fdr_correction(list(pvals.values()), alpha=alpha)
        out["fdr_significant"] = {n: bool(r) for n, r in zip(names, sig)}
    else:
        out["fdr_significant"] = {}
    # 判定
    for name, c in out["candidates"].items():
        out["verdicts"][name] = _verdict(
            c["consistent_folds"], c["n_folds_compared"], pvals[name],
            c["mean_delta_pp"], alpha, min_consistent_folds,
            out["fdr_significant"].get(name, False))
    return out


def validate_from_matrix(overlay_name: str, excess_matrix: dict,
                         base_label: str = "baseline", **kwargs) -> dict:
    """从 runner 产出的 {候选/基线名: {fold: excess_annual}} 矩阵直接判定。

    base_label 对应矩阵中的基线行; 其余行均为候选。
    """
    base = excess_matrix.get(base_label)
    if not base:
        raise ValueError(f"矩阵缺基线行: {base_label}")
    cands = {k: v for k, v in excess_matrix.items() if k != base_label and v}
    return validate_overlay_candidates(
        overlay_name, base, cands, **kwargs)
