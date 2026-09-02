"""
stats_correction.py — 多重检验校正工具 (因子筛选统计层)

来源: 自 archive/legacy/research_rigor.py 迁移 (2026-09-02)。原文件保留在
archive 不动 (历史引用), 本模块是主流程唯一允许 import 的统计校正入口
(模块准入: 被 scripts/active/run_walkforward_backtest.py、
run_null_calibration.py、run_corrected_backtest.py、run_regime_robustness.py
及 tests/ 引用)。

功能:
  1. 多重检验校正: bonferroni_correction / fdr_correction (BH, 支持有效检验数)
  2. IC → p 值: ic_to_pvalue (原始序列) / t_summary_to_pvalue (均值/标准差摘要)
  3. 有效独立检验数: factor_correlation_matrix + effective_num_tests
     (相关矩阵特征值分解, 保留解释 95% 方差的主成分数; 避免因子池里大量
     高度相关变体——如多周期 return_Nd/均线价差——把多重比较校正推向
     过度保守)
  4. 试验计数: count_trials (DSR n_trials 口径, 见函数说明)
"""

import json
import os
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. 多重检验校正 (自 archive/legacy/research_rigor.py 迁移)
# =============================================================================


def bonferroni_correction(p_values: List[float], alpha: float = 0.05,
                          m: int | None = None) -> List[bool]:
    """
    Bonferroni 多重检验校正。

    将显著性水平除以检验次数: adjusted_alpha = alpha / m
    保守但简单, 控制族错误率 (FWER)。

    Args:
        p_values: 各检验的原始p值列表
        alpha: 族显著性水平, 默认0.05
        m: 检验总数。默认 = len(p_values); 可传"有效独立检验数"
           (<= len(p_values)) 以缓解因子间高度相关导致的过度保守。

    Returns:
        布尔列表, True表示该检验通过校正 (拒绝H0)
    """
    p_values = list(p_values)
    if m is None:
        m = len(p_values)
    if m <= 0 or len(p_values) == 0:
        return [False] * len(p_values)
    adjusted_alpha = alpha / m
    return [p <= adjusted_alpha for p in p_values]


def fdr_correction(p_values: List[float], alpha: float = 0.05,
                   m: int | None = None) -> List[bool]:
    """
    Benjamini-Hochberg FDR (False Discovery Rate) 校正。

    步骤:
      1. 将p值从小到大排序, 记录原始索引
      2. 对第k个(排序后) p值, 阈值为 alpha * k / m
      3. 找到最大的k使得 p_(k) <= alpha * k / m
      4. 所有排名 <= k 的检验均拒绝H0

    相比Bonferroni更宽松, 控制的是错误发现率而非族错误率。
    适合因子筛选场景 — 允许少量假阳性, 但控制比例。

    m=None (标准形式) 时, BH 在独立或正相关 (PRDS) 检验下控制 FDR——
    因子 IC 之间通常正相关, 属标准做法。传 m=有效独立检验数 (< 实际
    检验数) 是相关性调整的常用启发式: 把冗余变体折算成一次检验,
    避免过度保守; 此时保证的是"有效检验族"意义上的 FDR。

    Args:
        p_values: 各检验的原始p值列表
        alpha: 目标FDR水平, 默认0.05
        m: 检验总数。默认 = len(p_values); 可传有效独立检验数。

    Returns:
        布尔列表 (与输入顺序对应), True表示该检验通过FDR校正
    """
    p_values = list(p_values)
    n = len(p_values)
    if n == 0:
        return []
    if m is None:
        m = n
    m = max(1, int(m))

    p_arr = np.asarray(p_values, dtype=float)
    # 排序并记录原始索引
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # BH阈值: alpha * k / m, k从1开始
    ranks = np.arange(1, n + 1)
    thresholds = alpha * ranks / m

    # 找最大的k使得 sorted_p[k-1] <= threshold[k-1]
    passed_sorted = sorted_p <= thresholds
    # 从后往前找第一个满足条件的
    max_k = 0
    for i in range(n - 1, -1, -1):
        if passed_sorted[i]:
            max_k = i + 1  # 转为1-based
            break

    # 所有排名 <= max_k 的都拒绝
    result = np.zeros(n, dtype=bool)
    if max_k > 0:
        result[sorted_indices[:max_k]] = True

    return result.tolist()


def ic_to_pvalue(ic_values: np.ndarray) -> float:
    """
    将IC序列转换为p值 (单样本t检验)。

    H0: mean(IC) = 0
    H1: mean(IC) != 0 (双侧)

    t统计量 = mean(IC) / (std(IC) / sqrt(n))
    自由度 = n - 1

    Args:
        ic_values: 日度/周度IC值数组 (一维)

    Returns:
        双侧p值。若IC序列为空或方差为0, 返回1.0 (不显著)
    """
    ic_values = np.asarray(ic_values, dtype=float)
    # 去除NaN
    ic_values = ic_values[~np.isnan(ic_values)]

    n = len(ic_values)
    if n < 2:
        return 1.0

    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values, ddof=1)

    if std_ic < 1e-12:
        return 1.0

    return t_summary_to_pvalue(mean_ic, std_ic, n)


def t_summary_to_pvalue(ic_mean: float, ic_std: float, n_obs: int) -> float:
    """
    从 IC 摘要统计量 (均值/标准差/观测数) 计算双侧 t 检验 p 值。

    与 ic_to_pvalue 等价, 用于只拿得到 ic_stats 摘要 (compute_icir_weights
    输出: ic_mean, ic_std, n_obs)、拿不到原始 IC 序列的场景。

    注意: 主流程的 IC 标准差用 ddof=0 (总体标准差), ic_to_pvalue 用
    ddof=1, 两者差异为 sqrt((n-1)/n), n>=40 时 <1.3%, 对 p 值影响可忽略。

    Returns:
        双侧p值。n_obs<2 或 ic_std<=0 时返回 1.0。
    """
    if n_obs is None or n_obs < 2:
        return 1.0
    if ic_std is None or ic_std < 1e-12:
        return 1.0
    t_stat = float(ic_mean) / (float(ic_std) / np.sqrt(n_obs))
    return float(2.0 * stats.t.sf(abs(t_stat), df=n_obs - 1))


# =============================================================================
# 2. 有效独立检验数 (因子相关性校正)
# =============================================================================


def factor_correlation_matrix(factor_panels: dict, dates: list,
                              factor_names: list) -> tuple[np.ndarray, list]:
    """
    因子池截面相关矩阵 (多日期平均)。

    每个日期取因子横截面 (n_stocks × n_factors), 逐列 z-score (NaN 补 0,
    与主流程 score_stocks 的 z-score 口径一致), 计算 Pearson 相关;
    各日期矩阵取均值。

    Args:
        factor_panels: {因子名: DataFrame(index=日期, columns=股票)}
        dates: 参与统计的日期列表 (建议取训练窗内按 IC_STEP 抽样的观测日)
        factor_names: 因子名列表 (决定输出矩阵的列序)

    Returns:
        (corr, used_names): corr 为 (k×k) 相关矩阵 (NaN→0, 对角线=1),
        used_names 为实际有面板的因子名 (保持输入顺序)。
    """
    names = [fn for fn in factor_names if fn in factor_panels]
    k = len(names)
    if k == 0:
        return np.zeros((0, 0)), []

    acc = np.zeros((k, k))
    cnt = np.zeros((k, k))
    for d in dates:
        cols = []
        for fn in names:
            panel = factor_panels[fn]
            if d in panel.index:
                cols.append(panel.loc[d].to_numpy(dtype=np.float64))
            else:
                cols.append(None)
        # 有效列: 有面板且有效值 >=30 (截面太小相关不可靠)
        valid_idx = [i for i, c in enumerate(cols)
                     if c is not None and int(np.isfinite(c).sum()) >= 30]
        if len(valid_idx) < 2:
            continue
        x = np.column_stack([cols[i] for i in valid_idx])
        # z-score (NaN→0, 与主流程 score_stocks 口径一致)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = np.where(np.isfinite(x), (x - mu) / sd, 0.0)
        c_mat = (z.T @ z) / z.shape[0]
        np.clip(c_mat, -1.0, 1.0, out=c_mat)
        blk = np.ix_(valid_idx, valid_idx)
        acc[blk] += c_mat
        cnt[blk] += 1

    if cnt.sum() == 0:
        return np.eye(k), names
    corr = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, names


def effective_num_tests(corr: np.ndarray,
                        variance_explained: float = 0.95) -> int:
    """
    由相关矩阵估计"有效独立检验数"。

    方法: 对相关矩阵做特征值分解, 按特征值降序累计方差解释率,
    取达到 variance_explained (默认95%) 所需的主成分数。
    完全相关的一组因子折算为 1 次检验, 完全不相关的 m 个因子仍为 m。

    Args:
        corr: (k×k) 对称相关矩阵
        variance_explained: 累计方差解释率阈值, 默认 0.95

    Returns:
        有效检验数, 钳制在 [1, k]。空矩阵返回 0。
    """
    k = corr.shape[0] if corr.ndim == 2 else 0
    if k == 0:
        return 0
    if k == 1:
        return 1
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    # 防御: 非有限值 (NaN/Inf) 会让特征分解静默塌缩到错误方向
    # (实测 NaN → m_eff=1, 最不保守) → 直接按完整检验数 k 保守处理
    if not np.isfinite(corr).all():
        return k
    try:
        eigvals = np.linalg.eigvalsh(corr)
    except np.linalg.LinAlgError:
        return k
    eigvals = np.clip(eigvals, 0.0, None)
    if not np.isfinite(eigvals).all():  # 防御: 特征值非有限同样回退 k
        return k
    total = eigvals.sum()
    if total <= 1e-12:
        return k
    order = np.argsort(eigvals)[::-1]
    cum = np.cumsum(eigvals[order]) / total
    n_eff = int(np.searchsorted(cum, variance_explained) + 1)
    return int(min(max(n_eff, 1), k))


# =============================================================================
# 3. 试验计数 (DSR n_trials 口径)
# =============================================================================


def count_trials(experiments_dir: str = "experiments") -> int:
    """
    累计试验次数 = experiments/ 目录中实验记录条数。

    口径 (写入 DEVELOPMENT_DISCIPLINE.md 与 evaluator._deflated_sharpe):
      每次经 experiment_tracker.log_experiment() 登记的运行算一次试验
      (含 walk-forward fold 分析、sleeve/参数实验、诊断脚本登记)。
      同一配置的机械重跑也计数——选择压力来自"研究者看到结果后保留或
      放弃", 与配置是否变化无关, 且高估试验数只会让 DSR 更保守,
      方向安全。

    Returns:
        实验记录条数; 目录不存在时为 0 (调用方应设 >=1 的下限)。
    """
    dir_path = experiments_dir if isinstance(experiments_dir, os.PathLike) \
        else os.fspath(experiments_dir)
    if not os.path.isdir(dir_path):
        return 0
    n = 0
    for fn in os.listdir(dir_path):
        if fn.startswith("exp_") and fn.endswith(".json"):
            n += 1
    return n


# =============================================================================
# 4. 校正结果产物读写 (动态替代旧静态 FDR_ELIMINATED 名单)
# =============================================================================


def load_fdr_correction(report_path: str) -> dict | None:
    """
    读取主选择流程产出的 FDR 校正报告 (fdr_correction_report.json)。

    Returns:
        {"stable_factors": [...], "rejected_factors": [...],
         "generated_at": ..., "meta": {...}};
        文件不存在/损坏返回 None (调用方决定降级策略)。
    """
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "stable_factors" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def apply_fdr_correction(factors: list, correction: dict | None,
                         log=None) -> list:
    """
    用动态 FDR 校正结果过滤因子列表 (替代旧静态 FDR_ELIMINATED 名单)。

    correction=None (报告缺失) 时返回原列表并打警告——降级为不剔除,
    而不是沿用任何静态名单。

    Args:
        factors: [{"name": ..., ...}, ...] 因子配置列表
        correction: load_fdr_correction() 的输出
        log: 可选 logger
    """
    if correction is None:
        if log:
            log.warning("FDR 校正报告缺失: 本次不剔除任何因子 "
                        "(请先跑 run_walkforward_backtest.py --folds-only 生成)")
        return list(factors)
    stable = set(correction.get("stable_factors") or [])
    rejected = set(correction.get("rejected_factors") or [])
    kept = [f for f in factors
            if f.get("name") in stable or f.get("name") not in rejected]
    n_drop = len(factors) - len(kept)
    if log:
        log.info("FDR 动态校正 (生成于 %s): 剔除 %d 因子, 保留 %d",
                 correction.get("generated_at", "?"), n_drop, len(kept))
    return kept
