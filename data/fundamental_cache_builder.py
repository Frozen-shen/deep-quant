"""
基本面因子预计算器 — 为每只股票生成 {date: {fund_roe, fund_roe_ttm, ...}} DataFrame
与 FactorCache 格式对齐, 无缝接入 pipeline
"""
import pandas as pd
import numpy as np
from data.fundamental import get_fundamental_factors, fetch_financials

FUND_FACTOR_NAMES = [
    'fund_roe', 'fund_roe_ttm', 'fund_profit_growth', 'fund_profit_growth_ttm',
    'fund_revenue_growth', 'fund_debt_ratio', 'fund_bvps', 'fund_eps_ttm',
]

FUND_FACTOR_DEFAULTS = {
    'fund_roe': np.nan, 'fund_roe_ttm': np.nan,
    'fund_profit_growth': np.nan, 'fund_profit_growth_ttm': np.nan,
    'fund_revenue_growth': np.nan, 'fund_debt_ratio': np.nan,
    'fund_bvps': np.nan, 'fund_eps_ttm': np.nan,
}

FUND_NAME_MAP = {
    'roe': 'fund_roe', 'roe_ttm': 'fund_roe_ttm',
    'profit_growth': 'fund_profit_growth', 'profit_growth_ttm': 'fund_profit_growth_ttm',
    'revenue_growth': 'fund_revenue_growth', 'debt_ratio': 'fund_debt_ratio',
    'bvps': 'fund_bvps', 'eps_ttm': 'fund_eps_ttm',
}


def precompute_fundamental_factors(all_data: dict) -> dict:
    """
    为所有股票预计算基本面因子。

    Args:
      all_data: {symbol: DataFrame(date, open, close, ...)}

    Returns:
      {symbol: DataFrame(date, fund_roe, fund_roe_ttm, ...)}
    """
    fund_cache = {}
    total = len(all_data)

    print(f"[Fundamental] 预计算 {total} 只...")
    for i, (sym, df) in enumerate(all_data.items()):
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{total}")

        # 取所有交易日
        dates = df['date'].tolist()

        rows = []
        for d in dates:
            factors = get_fundamental_factors(sym, d)
            row = {'date': d}
            if factors:
                for raw_name, value in factors.items():
                    col_name = FUND_NAME_MAP.get(raw_name, raw_name)
                    row[col_name] = value
            # 填充缺失列
            for name in FUND_FACTOR_NAMES:
                if name not in row:
                    row[name] = FUND_FACTOR_DEFAULTS[name]
            rows.append(row)

        fdf = pd.DataFrame(rows)
        # 前向填充: 基本面因子在下一次财报出来前保持不变
        for col in FUND_FACTOR_NAMES:
            if col in fdf.columns:
                fdf[col] = fdf[col].ffill()

        fund_cache[sym] = fdf

    print(f"  完成: {len(fund_cache)} 只")
    return fund_cache


def merge_fundamental_to_features(sym: str, today: pd.Timestamp, fund_cache: dict, price_features: list) -> list:
    """
    将基本面因子合并到价量因子向量中。

    Args:
      sym: 股票代码
      today: 日期
      fund_cache: 预计算的基本面因子缓存
      price_features: 已有的价量因子值列表 [f1, f2, ...]

    Returns:
      合并后的因子值列表 [price_f1, price_f2, ..., fund_f1, fund_f2, ...]
    """
    if sym not in fund_cache:
        return price_features

    fdf = fund_cache[sym]
    row = fdf[fdf['date'] == today]
    if len(row) == 0:
        return price_features

    fund_values = []
    for name in FUND_FACTOR_NAMES:
        if name in row.columns:
            v = row[name].iloc[0]
            fund_values.append(v if not (isinstance(v, float) and np.isnan(v)) else 0.0)
        else:
            fund_values.append(0.0)

    return list(price_features) + fund_values
