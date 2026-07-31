"""
基本面因子预计算器 — 为每只股票生成 {date: {fund_roe, fund_roe_ttm, ...}} DataFrame
与 FactorCache 格式对齐, 无缝接入 pipeline
"""
import pandas as pd
import numpy as np
from data.fundamental import fetch_financials, safe_float
from datetime import timedelta
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

        # 优化: 先用merge_asof在报告日期做对齐, 再ffill到交易日
        fin_df = fetch_financials(sym)
        if fin_df is None or len(fin_df) == 0:
            # 没有基本面数据 → 全填NaN → ffill后仍是NaN → merge时填0
            fdf = pd.DataFrame({'date': df['date'].tolist()})
            for name in FUND_FACTOR_NAMES:
                fdf[name] = np.nan
            fund_cache[sym] = fdf
            continue

        fin_df['日期'] = pd.to_datetime(fin_df['日期'])

        # 1. 在报告期日期上计算因子值
        from datetime import timedelta
        report_rows = []
        for _, row in fin_df.iterrows():
            rpt_date = row['日期']
            # 取该报告期及之前的所有财报 (PIT: 报告期45天后可用)
            avail_date = rpt_date + timedelta(days=45)
            historical = fin_df[fin_df['日期'] <= rpt_date]

            factors = {}
            roe = safe_float(row.get('净资产收益率(%)'))
            if roe is not None: factors['fund_roe'] = roe

            pg = safe_float(row.get('净利润增长率(%)'))
            if pg is not None: factors['fund_profit_growth'] = pg

            rg = safe_float(row.get('主营业务收入增长率(%)'))
            if rg is not None: factors['fund_revenue_growth'] = rg

            dr = safe_float(row.get('资产负债率(%)'))
            if dr is not None: factors['fund_debt_ratio'] = dr

            bv = safe_float(row.get('每股净资产_调整前(元)'))
            if bv is not None: factors['fund_bvps'] = bv

            # EPS TTM: 最近4个季度
            recent_4q = historical.tail(4)
            eps_sum = sum(safe_float(r.get('摊薄每股收益(元)')) or 0 for _, r in recent_4q.iterrows())
            factors['fund_eps_ttm'] = eps_sum if eps_sum != 0 else np.nan

            # ROE TTM
            roe_vals = [safe_float(r.get('净资产收益率(%)')) for _, r in recent_4q.iterrows()]
            roe_vals = [v for v in roe_vals if v is not None]
            factors['fund_roe_ttm'] = np.mean(roe_vals) if roe_vals else np.nan

            # Profit Growth TTM
            if len(historical) >= 8:
                r4 = [safe_float(r.get('净利润增长率(%)')) for _, r in historical.tail(4).iterrows()]
                p4 = [safe_float(r.get('净利润增长率(%)')) for _, r in historical.iloc[-8:-4].iterrows()]
                r4 = [v for v in r4 if v is not None]
                p4 = [v for v in p4 if v is not None]
                if r4 and p4:
                    factors['fund_profit_growth_ttm'] = np.mean(r4) - np.mean(p4)

            report_rows.append({'avail_date': avail_date, **factors})

        if not report_rows:
            fdf = pd.DataFrame({'date': df['date'].tolist()})
            for name in FUND_FACTOR_NAMES:
                fdf[name] = np.nan
            fund_cache[sym] = fdf
            continue

        # 2. 构建基本面时间序列: 在avail_date处有值, 然后ffill
        rdf = pd.DataFrame(report_rows)
        rdf = rdf.sort_values('avail_date')

        # 3. merge_asof: 每个交易日取最近可用的基本面数据
        tdf = pd.DataFrame({'date': pd.to_datetime(df['date'].tolist())})
        tdf = tdf.sort_values('date')
        # 统一 dtype 避免 merge_asof 报错
        tdf['date'] = tdf['date'].astype('datetime64[us]')
        rdf['avail_date'] = rdf['avail_date'].astype('datetime64[us]')
        merged = pd.merge_asof(tdf, rdf, left_on='date', right_on='avail_date', direction='backward')
        merged = merged.drop(columns=['avail_date'])

        # 补充缺失列
        for name in FUND_FACTOR_NAMES:
            if name not in merged.columns:
                merged[name] = np.nan

        fund_cache[sym] = merged[['date'] + FUND_FACTOR_NAMES]

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
