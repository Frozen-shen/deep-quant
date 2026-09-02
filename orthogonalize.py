"""
orthogonalize.py — 因子正交化 (P2, 2026-08-09)

Gram-Schmidt 正交化 (按给定顺序): 每个因子减去其在已正交因子上的投影。
用于去除稳定因子间的冗余 (corr 0.6 剪枝之外的系统性去相关)。
不改变第一个因子的值 (方向保留)。

2026-09-02 前视偏差修复: 投影系数改为逐日横截面估计 (每日只用当天
股票样本回归), 替代旧实现"全日期展平后一次性回归"——后者让早期日期
的因子值被未来数据拟合出的系数修正, 是跨时间泄露, 且隐含"因子间
线性关系十年不变"的不成立假设。
"""
import numpy as np
import pandas as pd

MIN_SAMPLES = 10  # 单日横截面最小有效样本数, 不足则该日跳过投影


def _daily_projection(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐日横截面回归投影: 对每一行 (交易日) 用当天样本估计
    v = alpha + beta*b + eps, 返回残差; 有效样本 < MIN_SAMPLES 的行不动。

    Args:
        v: (n_dates, n_symbols) 待正交化因子
        b: (n_dates, n_symbols) basis 因子 (已正交)

    Returns:
        与 v 同形状的新数组 (投影行替换为残差, 其余行保持原值)
    """
    valid = ~np.isnan(v) & ~np.isnan(b)
    n_valid = valid.sum(axis=1)
    enough = n_valid >= MIN_SAMPLES

    v_m = np.where(valid, v, np.nan)
    b_m = np.where(valid, b, np.nan)
    with np.errstate(invalid="ignore"):
        b_mean = np.nanmean(b_m, axis=1)
        v_mean = np.nanmean(v_m, axis=1)
        bc = b_m - b_mean[:, None]
        vc = v_m - v_mean[:, None]
        cov = np.nansum(bc * vc, axis=1)
        var = np.nansum(bc * bc, axis=1)
    beta = cov / (var + 1e-12)
    alpha = v_mean - beta * b_mean

    out = v.copy()
    rows = np.where(enough)[0]
    if len(rows):
        proj = alpha[rows, None] + beta[rows, None] * b[rows]
        out[rows] = np.where(valid[rows], v[rows] - proj, v[rows])
    return out


def orthogonalize_panels(panels: dict, factor_names: list,
                         method: str = "gs") -> dict:
    """按 factor_names 顺序逐日横截面 Gram-Schmidt 正交化面板。

    每个 basis 因子的投影系数逐交易日独立估计, 只使用当天横截面样本,
    任意交易日的系数估计不使用其他交易日数据 (无前视偏差)。

    Args:
        panels: {factor: DataFrame(date × symbol)} (float32)
        factor_names: 正交化顺序 (先正交的优先保留)
        method: 仅支持 "gs" (Gram-Schmidt)

    Returns:
        新面板 dict (原 panels 不修改)
    """
    names = [fn for fn in factor_names if fn in panels]
    if not names:
        return panels
    out = {}
    basis = []  # 已正交因子 (2D: n_dates × n_symbols)
    for fn in names:
        arr = panels[fn].to_numpy(dtype=np.float64)
        vc = arr.copy()
        if (~np.isnan(vc)).sum() > MIN_SAMPLES:
            for b in basis:
                vc = _daily_projection(vc, b)
        out[fn] = pd.DataFrame(
            vc, index=panels[fn].index,
            columns=panels[fn].columns).astype(np.float32)
        basis.append(vc)
    return out
