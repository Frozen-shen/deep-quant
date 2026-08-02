"""
研究严谨性工具 — 多重检验校正、因子正交化、经济逻辑过滤

解决量化研究中的三类常见方法论缺陷:
  1. 多重检验问题: 177个因子做IC检验, 按5%水平预期~9个假阳性
     → Bonferroni / FDR (Benjamini-Hochberg) 校正
  2. 因子冗余: 相关性剪枝不够, 需要真正正交化
     → Modified Gram-Schmidt 正交化 + 增量IC
  3. 数据挖掘偏差: 纯统计显著但无经济逻辑的因子
     → 基于因子类别的方向性合理性检查
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# =============================================================================
# 1. 多重检验校正
# =============================================================================


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Bonferroni 多重检验校正。

    将显著性水平除以检验次数: adjusted_alpha = alpha / m
    保守但简单, 控制族错误率 (FWER)。

    Args:
        p_values: 各检验的原始p值列表
        alpha: 族显著性水平, 默认0.05

    Returns:
        布尔列表, True表示该检验通过校正 (拒绝H0)
    """
    m = len(p_values)
    if m == 0:
        return []
    adjusted_alpha = alpha / m
    return [p <= adjusted_alpha for p in p_values]


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg FDR (False Discovery Rate) 校正。

    步骤:
      1. 将p值从小到大排序, 记录原始索引
      2. 对第k个(排序后) p值, 阈值为 alpha * k / m
      3. 找到最大的k使得 p_(k) <= alpha * k / m
      4. 所有排名 <= k 的检验均拒绝H0

    相比Bonferroni更宽松, 控制的是错误发现率而非族错误率。
    适合因子筛选场景 — 允许少量假阳性, 但控制比例。

    Args:
        p_values: 各检验的原始p值列表
        alpha: 目标FDR水平, 默认0.05

    Returns:
        布尔列表 (与输入顺序对应), True表示该检验通过FDR校正
    """
    m = len(p_values)
    if m == 0:
        return []

    p_arr = np.array(p_values)
    # 排序并记录原始索引
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # BH阈值: alpha * k / m, k从1开始
    ranks = np.arange(1, m + 1)
    thresholds = alpha * ranks / m

    # 找最大的k使得 sorted_p[k-1] <= threshold[k-1]
    passed_sorted = sorted_p <= thresholds
    # 从后往前找第一个满足条件的
    max_k = 0
    for i in range(m - 1, -1, -1):
        if passed_sorted[i]:
            max_k = i + 1  # 转为1-based
            break

    # 所有排名 <= max_k 的都拒绝
    result = np.zeros(m, dtype=bool)
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
        ic_values: 日度IC值数组 (一维)

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

    t_stat = mean_ic / (std_ic / np.sqrt(n))
    # 双侧p值: 2 * P(T > |t|) = 2 * sf(|t|, df)
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)

    return float(p_value)


def validate_with_correction(
    ic_results: Dict[str, List[float]],
    method: str = "fdr",
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """
    对多因子IC结果进行多重检验校正。

    流程:
      1. 对每个因子的IC序列计算p值 (t检验)
      2. 对所有p值进行多重检验校正 (Bonferroni 或 FDR)
      3. 返回每个因子的统计摘要和显著性判定

    Args:
        ic_results: {因子名: [日度IC值列表]}
        method: 校正方法, "fdr" (默认) 或 "bonferroni"
        alpha: 显著性水平, 默认0.05

    Returns:
        {因子名: {
            "p_value": float,       # 原始p值
            "significant": bool,    # 校正后是否显著
            "ic_mean": float,       # IC均值
            "icir": float,          # ICIR = mean(IC) / std(IC)
        }}
    """
    if not ic_results:
        return {}

    factor_names = list(ic_results.keys())
    p_values = []
    summaries = {}

    for name in factor_names:
        ic_arr = np.array(ic_results[name], dtype=float)
        ic_arr = ic_arr[~np.isnan(ic_arr)]

        p_val = ic_to_pvalue(ic_arr)
        p_values.append(p_val)

        ic_mean = float(np.mean(ic_arr)) if len(ic_arr) > 0 else 0.0
        ic_std = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
        icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

        summaries[name] = {
            "p_value": p_val,
            "ic_mean": round(ic_mean, 6),
            "icir": round(icir, 4),
        }

    # 多重检验校正
    if method == "bonferroni":
        significant = bonferroni_correction(p_values, alpha=alpha)
    elif method == "fdr":
        significant = fdr_correction(p_values, alpha=alpha)
    else:
        raise ValueError(f"未知校正方法: {method}, 支持 'fdr' 或 'bonferroni'")

    # 组装结果
    results = {}
    for i, name in enumerate(factor_names):
        results[name] = {
            "p_value": summaries[name]["p_value"],
            "significant": significant[i],
            "ic_mean": summaries[name]["ic_mean"],
            "icir": summaries[name]["icir"],
        }

    return results


# =============================================================================
# 2. 因子正交化 (Gram-Schmidt)
# =============================================================================


def orthogonalize_factors(
    factor_matrix: np.ndarray,
    priority_order: List[int] = None,
) -> np.ndarray:
    """
    Modified Gram-Schmidt 正交化。

    将因子矩阵的列向量正交化, 使得每个因子只贡献独立于前面因子的信息。
    正交化顺序由 priority_order 决定 (默认按列索引顺序)。

    典型用法: 按|ICIR|降序排列因子, 最重要的因子保留完整信息,
    后续因子只保留与前面因子正交的残差部分。

    Args:
        factor_matrix: (n_samples, n_factors) 因子矩阵
        priority_order: 正交化优先级顺序 (列索引列表)。
                        若为None, 按 0, 1, 2, ... 顺序。

    Returns:
        正交化后的矩阵, shape与输入相同。
        列向量两两正交 (数值精度内)。
        若某列与前面所有列线性相关, 则该列变为零向量。
    """
    X = np.array(factor_matrix, dtype=float, copy=True)
    n_samples, n_factors = X.shape

    if priority_order is None:
        priority_order = list(range(n_factors))

    # 结果矩阵
    Q = np.zeros_like(X)
    # 工作副本
    V = X.copy()

    for idx in priority_order:
        v = V[:, idx].copy()
        # 减去在所有已正交化方向上的投影
        for prev_idx in priority_order:
            if prev_idx == idx:
                break
            q = Q[:, prev_idx]
            norm_sq = np.dot(q, q)
            if norm_sq > 1e-12:
                proj = np.dot(v, q) / norm_sq
                v = v - proj * q

        Q[:, idx] = v

    return Q


def compute_incremental_ic(
    factor_values: np.ndarray,
    target: np.ndarray,
    existing_factors: np.ndarray,
) -> float:
    """
    计算因子在控制已有因子后的增量IC。

    方法:
      1. 对 existing_factors 做OLS回归, 取 factor_values 的残差
      2. 计算残差与 target 的 Spearman 秩相关

    这衡量的是: 在已知 existing_factors 信息的条件下,
    该因子还能提供多少额外的预测能力。

    Args:
        factor_values: (n,) 待评估因子值
        target: (n,) 目标变量 (如前瞻收益)
        existing_factors: (n, k) 已有因子矩阵

    Returns:
        增量IC (Spearman相关系数)。
        若残差方差为0 (因子完全被已有因子解释), 返回0.0。
    """
    factor_values = np.asarray(factor_values, dtype=float)
    target = np.asarray(target, dtype=float)
    existing_factors = np.asarray(existing_factors, dtype=float)

    # 去除含NaN的行
    if existing_factors.ndim == 1:
        existing_factors = existing_factors.reshape(-1, 1)

    valid_mask = (
        ~np.isnan(factor_values)
        & ~np.isnan(target)
        & ~np.any(np.isnan(existing_factors), axis=1)
    )
    factor_values = factor_values[valid_mask]
    target = target[valid_mask]
    existing_factors = existing_factors[valid_mask]

    n = len(factor_values)
    if n < 10:
        return 0.0

    # OLS回归: factor_values ~ existing_factors + intercept
    # 添加截距项
    X = np.column_stack([np.ones(n), existing_factors])

    # 用最小二乘法求残差
    # beta = (X'X)^-1 X'y
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, factor_values, rcond=None)
        fitted = X @ beta
        resid = factor_values - fitted
    except np.linalg.LinAlgError:
        return 0.0

    # 残差方差检查
    if np.std(resid) < 1e-12:
        return 0.0

    # Spearman相关
    corr, _ = stats.spearmanr(resid, target)
    if np.isnan(corr):
        return 0.0

    return float(corr)


# =============================================================================
# 3. 经济逻辑过滤
# =============================================================================

# 因子类别 → 合理IC方向的映射
_ECONOMIC_LOGIC_RULES = {
    "momentum": {
        "expected_sign": "positive",
        "explanation": "动量效应: 过去表现好的股票倾向于继续表现好 (Jegadeesh & Titman 1993)",
        "contrarian_explanation": "短期反转效应: 过去1-4周的输家可能反弹 (Jegadeesh 1990)",
    },
    "value": {
        "expected_sign": "positive",
        "explanation": "价值溢价: 高EP/BP股票长期跑赢 (Fama & French 1992)",
        "contrarian_explanation": None,
    },
    "volatility": {
        "expected_sign": "negative",
        "explanation": "低波动异象: 低波动股票风险调整后收益更高 (Ang et al. 2006)",
        "contrarian_explanation": None,
    },
    "liquidity": {
        "expected_sign": "negative",
        "explanation": "流动性溢价: 低流动性资产要求更高收益补偿 (Amihud 2002)",
        "contrarian_explanation": None,
    },
    "quality": {
        "expected_sign": "positive",
        "explanation": "质量溢价: 高盈利/低杠杆公司长期表现更好 (Novy-Marx 2013)",
        "contrarian_explanation": None,
    },
    "size": {
        "expected_sign": "negative",
        "explanation": "规模效应: 小市值股票长期有超额收益 (Fama & French 1992)",
        "contrarian_explanation": None,
    },
}


def check_economic_logic(
    factor_name: str,
    category: str,
    ic_sign: float,
) -> dict:
    """
    根据因子名和类别判断IC方向是否符合经济逻辑。

    经济逻辑是区分真实alpha与数据挖掘的关键过滤器:
    - 如果一个因子的IC方向与已知经济理论一致, 更可能是真实信号
    - 如果方向相反, 需要额外解释 (可能是市场制度差异或新发现)

    规则:
      - momentum类: 正IC合理 (动量效应), 负IC可能是短期反转
      - value类 (EP/BP): 正IC合理 (价值溢价)
      - volatility类: 负IC合理 (低波动异象)
      - liquidity类: 负IC合理 (流动性溢价)
      - quality类: 正IC合理 (质量溢价)
      - size类: 负IC合理 (规模效应)

    Args:
        factor_name: 因子名称 (用于辅助判断)
        category: 因子类别, 如 "momentum", "value", "volatility", "liquidity"
        ic_sign: IC均值的符号 (正数或负数)

    Returns:
        {
            "plausible": bool,        # IC方向是否符合经济逻辑
            "explanation": str,       # 经济逻辑解释
            "warning": str or None,   # 若不符合, 给出警告信息
        }
    """
    category_lower = category.lower().strip()
    rule = _ECONOMIC_LOGIC_RULES.get(category_lower)

    if rule is None:
        # 未知类别, 无法判断
        return {
            "plausible": True,  # 不阻止, 但标记为未知
            "explanation": f"类别 '{category}' 无预设经济逻辑规则, 需人工判断",
            "warning": f"未识别的因子类别 '{category}', 建议补充经济逻辑说明",
        }

    expected_sign = rule["expected_sign"]
    is_positive = ic_sign > 0

    if expected_sign == "positive":
        direction_match = is_positive
    else:  # "negative"
        direction_match = not is_positive

    if direction_match:
        return {
            "plausible": True,
            "explanation": rule["explanation"],
            "warning": None,
        }
    else:
        # 方向不符, 检查是否有合理的反向解释
        contrarian = rule.get("contrarian_explanation")
        if contrarian:
            return {
                "plausible": True,  # 有合理的反向解释
                "explanation": contrarian,
                "warning": (
                    f"IC方向与主流理论 ({rule['explanation']}) 不一致, "
                    f"但可能符合: {contrarian}"
                ),
            }
        else:
            return {
                "plausible": False,
                "explanation": rule["explanation"],
                "warning": (
                    f"IC符号 ({'正' if is_positive else '负'}) 与 "
                    f"{category_lower} 类因子的经济逻辑预期 "
                    f"({'正' if expected_sign == 'positive' else '负'}) 相反, "
                    f"可能是数据挖掘或市场制度差异, 需要额外解释"
                ),
            }
