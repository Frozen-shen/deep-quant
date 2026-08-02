"""
因子中性化 — 剥离行业/市值/风格暴露

三种中性化:
  1. 行业中性: 因子值减去行业均值 (或行业哑变量回归取残差)
  2. 市值中性: 对 log(market_cap) 回归取残差
  3. 联合中性: 同时剥离行业 + 市值

用法:
    from factor_factory.neutralize import neutralize_cross_section

    # 单日截面中性化
    neutralized = neutralize_cross_section(
        factor_values={"000001": 0.5, "600519": -0.3, ...},
        industry_map={"000001": "银行", "600519": "白酒", ...},
        market_cap={"000001": 3e10, "600519": 2e12, ...},
        method="industry_size"
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def neutralize_cross_section(
    factor_values: Dict[str, float],
    industry_map: Dict[str, str] = None,
    market_cap: Dict[str, float] = None,
    method: str = "industry_size",
) -> Dict[str, float]:
    """
    对单个截面的因子值做中性化。

    Args:
        factor_values: {symbol: raw_factor_value}
        industry_map: {symbol: industry_name}
        market_cap: {symbol: total_market_cap}
        method: "industry" / "size" / "industry_size"

    Returns:
        {symbol: neutralized_value}
    """
    symbols = list(factor_values.keys())
    if len(symbols) < 10:
        return factor_values  # 样本太少不做中性化

    y = np.array([factor_values[s] for s in symbols], dtype=float)
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 10:
        return factor_values

    # 构建回归矩阵 X
    X_parts = []

    if method in ("industry", "industry_size") and industry_map:
        # 行业哑变量 (drop_first 避免多重共线)
        industries = [industry_map.get(s, "未知") for s in symbols]
        ind_dummies = pd.get_dummies(industries, prefix="ind", drop_first=True)
        X_parts.append(ind_dummies.values.astype(float))

    if method in ("size", "industry_size") and market_cap:
        # log市值
        log_cap = np.array([
            np.log(market_cap[s]) if s in market_cap and market_cap[s] > 0 else np.nan
            for s in symbols
        ])
        # 缺失值用中位数填充
        median_cap = np.nanmedian(log_cap)
        log_cap = np.where(np.isnan(log_cap), median_cap, log_cap)
        X_parts.append(log_cap.reshape(-1, 1))

    if not X_parts:
        return factor_values

    X = np.hstack(X_parts)

    # 对有效行做 OLS 回归, 取残差
    result = np.full(len(symbols), np.nan)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < X.shape[1] + 5:
        return factor_values  # 样本不够做回归

    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    # 处理 X 中的 nan
    x_valid_mask = ~np.any(np.isnan(X_valid), axis=1)
    if x_valid_mask.sum() < X.shape[1] + 5:
        return factor_values

    X_clean = X_valid[x_valid_mask]
    y_clean = y_valid[x_valid_mask]

    # OLS: residual = y - X @ (X'X)^-1 X'y
    try:
        # 用 lstsq 更稳定
        beta, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
        residual = y_clean - X_clean @ beta

        # 填回
        clean_idx = valid_idx[x_valid_mask]
        for i, idx in enumerate(clean_idx):
            result[idx] = residual[i]

        # 对无法回归的行保持原值
        for i, idx in enumerate(valid_idx):
            if np.isnan(result[idx]):
                result[idx] = y[idx]
    except np.linalg.LinAlgError:
        return factor_values

    return {s: float(result[i]) if not np.isnan(result[i]) else factor_values[s]
            for i, s in enumerate(symbols)}


def neutralize_panel(
    factor_panel: pd.DataFrame,
    industry_col: str = "industry",
    mktcap_col: str = "total_mv",
    method: str = "industry_size",
) -> pd.DataFrame:
    """
    对面板数据 (date × symbol) 逐日做中性化。

    Args:
        factor_panel: MultiIndex DataFrame (date, symbol) → factor_value
                      或 columns=[date, symbol, factor, industry, mktcap]
        industry_col: 行业列名
        mktcap_col: 市值列名
        method: 中性化方法

    Returns:
        同结构 DataFrame, factor 列替换为中性化后的值
    """
    if isinstance(factor_panel.index, pd.MultiIndex):
        # MultiIndex 格式
        result = factor_panel.copy()
        for date in factor_panel.index.get_level_values(0).unique():
            day_data = factor_panel.loc[date]
            fv = day_data["factor"].to_dict()
            ind = day_data[industry_col].to_dict() if industry_col in day_data.columns else None
            cap = day_data[mktcap_col].to_dict() if mktcap_col in day_data.columns else None
            neutralized = neutralize_cross_section(fv, ind, cap, method)
            for sym, val in neutralized.items():
                result.loc[(date, sym), "factor"] = val
        return result
    else:
        # 长表格式
        result = factor_panel.copy()
        for date in factor_panel["date"].unique():
            mask = factor_panel["date"] == date
            day = factor_panel[mask]
            fv = dict(zip(day["symbol"], day["factor"]))
            ind = dict(zip(day["symbol"], day[industry_col])) if industry_col in day.columns else None
            cap = dict(zip(day["symbol"], day[mktcap_col])) if mktcap_col in day.columns else None
            neutralized = neutralize_cross_section(fv, ind, cap, method)
            for sym, val in neutralized.items():
                result.loc[mask & (result["symbol"] == sym), "factor"] = val
        return result


def zscore_cross_section(values: Dict[str, float]) -> Dict[str, float]:
    """截面 z-score 标准化 (中性化后通常需要)。"""
    arr = np.array(list(values.values()))
    valid = arr[~np.isnan(arr)]
    if len(valid) < 5:
        return values
    mean, std = valid.mean(), valid.std()
    if std < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std if not np.isnan(v) else 0.0
            for k, v in values.items()}
