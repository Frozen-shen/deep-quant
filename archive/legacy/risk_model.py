"""
Barra风格风险模型 & 均值-方差组合优化器

实现:
- Barra CNE5 简化风格因子 (7个: Size, Value, Momentum, Volatility, Quality, Growth, Liquidity)
- 风格因子协方差估计: Σ = B·F·B' + D
- 带约束的均值-方差优化 (SLSQP)
- 行业偏离约束

用法:
    from risk_model import BarraRiskModel, optimize_portfolio

    model = BarraRiskModel()
    exposures = model.compute_style_exposures(factor_values, symbols)
    cov = model.estimate_covariance(returns, exposures)
    weights = optimize_portfolio(expected_returns, cov, symbols, constraints)
"""

from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ================================================================
#  Barra CNE5 风格因子 (简化版)
# ================================================================

STYLE_FACTORS = [
    "Size",        # 市值
    "Value",       # 估值 (BP)
    "Momentum",    # 动量
    "Volatility",  # 波动率 (残差波动)
    "Quality",     # 质量 (盈利质量)
    "Growth",      # 成长
    "Liquidity",   # 流动性
]


class BarraRiskModel:
    """
    Barra CNE5 简化风格风险模型。

    7个风格因子: Size, Value, Momentum, Volatility, Quality, Growth, Liquidity

    协方差结构: Σ = B·F·B' + D
      - B: 风格暴露矩阵 (N x K)
      - F: 风格因子协方差矩阵 (K x K)
      - D: 特异性风险对角矩阵 (N x N)
    """

    def __init__(self, n_styles: int = 7):
        """
        初始化风险模型。

        参数
        ----
        n_styles : int
            风格因子数量，默认7个 (对应 STYLE_FACTORS)。
        """
        self.n_styles = n_styles
        self.style_names = STYLE_FACTORS[:n_styles]

    def compute_style_exposures(
        self,
        factor_values: Dict[str, Dict[str, float]],
        symbols: List[str],
    ) -> pd.DataFrame:
        """
        计算每只股票的风格暴露 (横截面标准化)。

        对每个风格因子，在截面上做 z-score 标准化:
            z_i = (x_i - mean) / std

        参数
        ----
        factor_values : Dict[str, Dict[str, float]]
            {因子名: {股票代码: 原始值}}
            例如 {"Size": {"600519": 12.5, "000858": 11.8}, ...}
        symbols : List[str]
            参与计算的股票列表。

        返回
        ----
        pd.DataFrame
            index=symbols, columns=style_names, 值为标准化后的暴露。
            缺失值填0 (中性暴露)。
        """
        n = len(symbols)
        exposures = np.zeros((n, self.n_styles))

        for j, factor_name in enumerate(self.style_names):
            raw_values = []
            for sym in symbols:
                val = factor_values.get(factor_name, {}).get(sym, np.nan)
                raw_values.append(val)

            arr = np.array(raw_values, dtype=float)

            # 横截面 z-score 标准化
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() > 1:
                mean = np.nanmean(arr)
                std = np.nanstd(arr, ddof=0)
                if std > 1e-10:
                    arr = (arr - mean) / std
                else:
                    arr = np.zeros(n)
            else:
                arr = np.zeros(n)

            # 缺失值 → 0 (中性)
            arr = np.nan_to_num(arr, nan=0.0)
            exposures[:, j] = arr

        return pd.DataFrame(exposures, index=symbols, columns=self.style_names)

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        style_exposures: pd.DataFrame,
    ) -> np.ndarray:
        """
        用风格因子模型估计协方差矩阵: Σ = B·F·B' + D

        步骤:
        1. 从收益率和风格暴露回归得到因子收益 (OLS)
        2. 因子协方差 F = cov(factor_returns) * 252 (年化)
        3. 特异性风险 D = diag(var(residuals)) * 252 (年化)

        参数
        ----
        returns : pd.DataFrame
            日收益率, index=日期, columns=股票代码。
        style_exposures : pd.DataFrame
            风格暴露矩阵, index=股票代码, columns=风格因子名。

        返回
        ----
        np.ndarray
            年化协方差矩阵 (N x N)，N = len(style_exposures)。
            保证对称正半定 (通过特征值截断)。
        """
        symbols = list(style_exposures.index)
        n = len(symbols)
        k = self.n_styles

        # 对齐: 只保留returns中存在的股票
        common_symbols = [s for s in symbols if s in returns.columns]
        if len(common_symbols) == 0:
            # 退化: 返回单位矩阵 (等波动率假设)
            return np.eye(n) * 0.04  # ~20% 年化波动

        # 构建暴露矩阵 B (n x k)
        B = style_exposures.loc[common_symbols].values

        # 收益率矩阵 (T x n_common)
        R = returns[common_symbols].values
        T = R.shape[0]

        if T < k + 2:
            # 样本太少，回退到样本协方差
            if T > 1:
                sample_cov = np.cov(R, rowvar=False) * 252
                return self._ensure_psd(sample_cov)
            return np.eye(n) * 0.04

        # OLS 回归: R_t = B · f_t + ε_t  →  f_t = (B'B)^{-1} B' R_t
        BtB = B.T @ B
        # 正则化防止奇异
        BtB += np.eye(k) * 1e-8
        BtB_inv = np.linalg.inv(BtB)
        # 因子收益 (T x k)
        factor_returns = R @ B @ BtB_inv  # 等价于 R @ B @ (B'B)^{-1}
        # 更准确: f_t = (B'B)^{-1} B' r_t → F = (T x k)
        factor_returns = R @ B @ BtB_inv  # (T, n) @ (n, k) @ (k, k) = (T, k)

        # 因子协方差 (年化)
        F = np.cov(factor_returns, rowvar=False) * 252

        # 残差 → 特异性风险
        fitted = factor_returns @ B.T  # (T, n)
        residuals = R - fitted
        specific_var = np.var(residuals, axis=0, ddof=1) * 252  # (n_common,)

        # 构建完整协方差: Σ = B·F·B' + D
        n_common = len(common_symbols)
        cov_common = B @ F @ B.T + np.diag(specific_var)

        # 映射回完整 symbols 顺序
        cov_full = np.eye(n) * 0.04  # 默认对角
        sym_to_idx = {s: i for i, s in enumerate(common_symbols)}
        for i, s in enumerate(symbols):
            if s in sym_to_idx:
                ci = sym_to_idx[s]
                for j2, s2 in enumerate(symbols):
                    if s2 in sym_to_idx:
                        cj = sym_to_idx[s2]
                        cov_full[i, j2] = cov_common[ci, cj]

        return self._ensure_psd(cov_full)

    def compute_portfolio_risk(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """
        计算组合年化波动率。

        σ_p = sqrt(w' Σ w)

        参数
        ----
        weights : np.ndarray
            权重向量 (N,)
        cov_matrix : np.ndarray
            年化协方差矩阵 (N x N)

        返回
        ----
        float
            年化组合波动率 (标准差)。
        """
        w = np.asarray(weights, dtype=float)
        variance = w @ cov_matrix @ w
        # 防止数值误差导致负值
        return float(np.sqrt(max(variance, 0.0)))

    @staticmethod
    def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
        """
        确保矩阵对称正半定 (通过特征值截断)。

        将负特征值截断为一个小正数，保证数值稳定性。
        """
        matrix = (matrix + matrix.T) / 2  # 强制对称
        eigvals, eigvecs = np.linalg.eigh(matrix)
        # 截断负特征值
        eigvals = np.maximum(eigvals, 1e-10)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ================================================================
#  行业约束工具函数
# ================================================================

def get_industry_weights(
    symbols: List[str],
    industry_map: Dict[str, str],
) -> Dict[str, float]:
    """
    计算等权基准的行业权重。

    每只股票权重 = 1/N，然后按行业汇总。

    参数
    ----
    symbols : List[str]
        股票池。
    industry_map : Dict[str, str]
        {股票代码: 行业名称}

    返回
    ----
    Dict[str, float]
        {行业名: 权重}，权重之和 = 1。
    """
    if not symbols:
        return {}

    n = len(symbols)
    equal_weight = 1.0 / n
    industry_weights: Dict[str, float] = defaultdict(float)

    for sym in symbols:
        industry = industry_map.get(sym, "其他")
        industry_weights[industry] += equal_weight

    return dict(industry_weights)


def check_industry_constraint(
    weights: Dict[str, float],
    industry_map: Dict[str, str],
    benchmark_weights: Dict[str, float],
    max_dev: float,
) -> bool:
    """
    检查组合的行业偏离是否满足约束。

    对每个行业: |组合行业权重 - 基准行业权重| <= max_dev

    参数
    ----
    weights : Dict[str, float]
        {股票代码: 权重}
    industry_map : Dict[str, str]
        {股票代码: 行业名称}
    benchmark_weights : Dict[str, float]
        {行业名: 基准权重}
    max_dev : float
        最大允许偏离 (如 0.05 表示 5%)。

    返回
    ----
    bool
        True 表示所有行业偏离均在限制内。
    """
    # 计算组合的行业权重
    portfolio_industry: Dict[str, float] = defaultdict(float)
    for sym, w in weights.items():
        industry = industry_map.get(sym, "其他")
        portfolio_industry[industry] += w

    # 检查每个行业 (包括基准中有但组合中没有的)
    all_industries = set(portfolio_industry.keys()) | set(benchmark_weights.keys())
    for ind in all_industries:
        port_w = portfolio_industry.get(ind, 0.0)
        bench_w = benchmark_weights.get(ind, 0.0)
        if abs(port_w - bench_w) > max_dev + 1e-9:
            return False

    return True


# ================================================================
#  均值-方差优化器 (带约束)
# ================================================================

def optimize_portfolio(
    expected_returns: Dict[str, float],
    cov_matrix: np.ndarray,
    symbols: List[str],
    constraints: Optional[dict] = None,
    prev_weights: Optional[Dict[str, float]] = None,
    industry_map: Optional[Dict[str, str]] = None,
    style_exposures: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    均值-方差组合优化 (带约束)。

    目标函数: max  w'μ - (λ/2)·w'Σw - γ·turnover
    等价于:   min  -(w'μ) + (λ/2)·w'Σw + γ·turnover

    约束:
    - 权重和 = 1 (全额投资)
    - 0 <= w_i <= max_single_weight (个股上限)
    - 行业偏离 <= max_industry_dev (每个行业)
    - 风格暴露 <= max_style_exp σ (每个风格因子)

    参数
    ----
    expected_returns : Dict[str, float]
        {股票代码: 预期收益率}
    cov_matrix : np.ndarray
        年化协方差矩阵 (N x N)，顺序与 symbols 一致。
    symbols : List[str]
        股票池列表。
    constraints : dict, optional
        约束参数:
        - max_single: 个股最大权重 (默认 0.08)
        - max_industry_dev: 行业最大偏离 (默认 0.05)
        - max_style_exp: 风格暴露上限/标准差 (默认 0.5)
        - risk_aversion: 风险厌恶系数 λ (默认 1.0)
        - turnover_penalty: 换手惩罚系数 γ (默认 0.1)
    prev_weights : Dict[str, float], optional
        上期权重 (用于计算换手)。None 则不计换手惩罚。
    industry_map : Dict[str, str], optional
        {股票代码: 行业名称}。None 则跳过行业约束。
    style_exposures : pd.DataFrame, optional
        风格暴露矩阵 (index=symbols, columns=因子名)。None 则跳过风格约束。

    返回
    ----
    Dict[str, float]
        {股票代码: 最优权重}

    备注
    ----
    使用 scipy.optimize.minimize (SLSQP) 求解。
    对于退化情况 (协方差奇异、约束不可行等)，回退到等权组合。
    """
    n = len(symbols)
    if n == 0:
        return {}

    # 默认约束参数
    params = {
        "max_single": 0.08,
        "max_industry_dev": 0.05,
        "max_style_exp": 0.5,
        "risk_aversion": 1.0,
        "turnover_penalty": 0.1,
    }
    if constraints:
        params.update(constraints)

    lam = params["risk_aversion"]
    gamma = params["turnover_penalty"]
    max_single = params["max_single"]

    # 预期收益向量
    mu = np.array([expected_returns.get(s, 0.0) for s in symbols], dtype=float)

    # 确保协方差矩阵有效
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.shape != (n, n):
        # 退化: 使用对角矩阵
        cov = np.eye(n) * 0.04

    # 上期权重向量 (用于换手惩罚)
    if prev_weights is not None:
        w_prev = np.array([prev_weights.get(s, 0.0) for s in symbols], dtype=float)
    else:
        w_prev = None

    # 行业基准 (如果需要行业约束)
    benchmark_industry = None
    symbol_industries = None
    if industry_map is not None:
        benchmark_industry = get_industry_weights(symbols, industry_map)
        symbol_industries = [industry_map.get(s, "其他") for s in symbols]

    # 风格暴露矩阵
    B = None
    if style_exposures is not None:
        try:
            B = style_exposures.loc[symbols].values  # (n, k)
        except (KeyError, IndexError):
            B = None

    # ---- 目标函数 ----
    def objective(w):
        """负效用: -(收益) + (λ/2)·风险 + γ·换手"""
        ret = w @ mu
        risk = w @ cov @ w
        obj = -ret + (lam / 2.0) * risk

        if w_prev is not None and gamma > 0:
            turnover = np.sum(np.abs(w - w_prev))
            obj += gamma * turnover

        return obj

    def objective_jac(w):
        """目标函数梯度。"""
        grad = -mu + lam * (cov @ w)
        if w_prev is not None and gamma > 0:
            # |w - w_prev| 的次梯度: sign(w - w_prev)
            grad += gamma * np.sign(w - w_prev)
        return grad

    # ---- 约束 ----
    cons = []

    # 1. 权重和 = 1
    cons.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones(n),
    })

    # 2. 行业偏离约束
    if symbol_industries is not None and benchmark_industry is not None:
        max_dev = params["max_industry_dev"]
        industries = list(set(symbol_industries))

        for ind in industries:
            # 该行业对应的股票索引
            ind_mask = np.array([1.0 if si == ind else 0.0 for si in symbol_industries])
            bench_w = benchmark_industry.get(ind, 0.0)

            # 组合行业权重 - 基准 <= max_dev
            cons.append({
                "type": "ineq",
                "fun": lambda w, m=ind_mask, bw=bench_w, md=max_dev: md - (w @ m - bw),
            })
            # 基准 - 组合行业权重 <= max_dev
            cons.append({
                "type": "ineq",
                "fun": lambda w, m=ind_mask, bw=bench_w, md=max_dev: md - (bw - w @ m),
            })

    # 3. 风格暴露约束: |B'w| <= max_style_exp
    if B is not None:
        max_exp = params["max_style_exp"]
        k = B.shape[1]
        for j in range(k):
            bj = B[:, j]
            # B_j' w <= max_exp
            cons.append({
                "type": "ineq",
                "fun": lambda w, b=bj, me=max_exp: me - w @ b,
            })
            # -B_j' w <= max_exp
            cons.append({
                "type": "ineq",
                "fun": lambda w, b=bj, me=max_exp: me + w @ b,
            })

    # ---- 边界: 0 <= w_i <= max_single ----
    bounds = [(0.0, max_single) for _ in range(n)]

    # ---- 初始点: 等权 ----
    w0 = np.ones(n) / n
    # 如果等权超出个股上限，调整初始点
    if max_single < 1.0 / n:
        w0 = np.full(n, max_single)
        w0 /= w0.sum()

    # ---- 求解 ----
    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            weights = result.x
            # 清理数值噪声
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
        else:
            # SLSQP 失败 → 回退等权
            weights = w0.copy()
            weights = np.minimum(weights, max_single)
            weights /= weights.sum()

    except Exception:
        # 任何异常 → 回退等权
        weights = np.ones(n) / n

    return {sym: float(w) for sym, w in zip(symbols, weights)}
