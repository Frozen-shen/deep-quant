"""
orthogonalize.py — 因子正交化 (P2, 2026-08-09)

Gram-Schmidt 正交化 (按给定顺序): 每个因子减去其在已正交因子上的投影。
用于去除稳定因子间的冗余 (corr 0.6 剪枝之外的系统性去相关)。
不改变第一个因子的值 (方向保留)。
"""
import numpy as np
import pandas as pd


def orthogonalize_panels(panels: dict, factor_names: list,
                         method: str = "gs") -> dict:
    """按 factor_names 顺序 Gram-Schmidt 正交化面板。

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
    basis = []  # 已正交因子 (展平向量)
    for fn in names:
        v = panels[fn].to_numpy(dtype=np.float64).ravel()
        m = ~np.isnan(v)
        vc = v.copy()
        if m.sum() > 10:
            for b in basis:
                bm = m & ~np.isnan(b)
                if bm.sum() < 10:
                    continue
                # 回归投影: v = alpha + beta*b + eps → 残差
                b_ = b[bm]
                v_ = vc[bm]
                beta = np.cov(b_, v_)[0, 1] / (np.var(b_) + 1e-12)
                alpha = v_.mean() - beta * b_.mean()
                vc[bm] = v_ - (alpha + beta * b_)
        out[fn] = pd.DataFrame(
            vc.reshape(panels[fn].shape), index=panels[fn].index,
            columns=panels[fn].columns).astype(np.float32)
        basis.append(vc)
    return out
