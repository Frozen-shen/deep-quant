"""
PIT 基本面因子 — 公告日对齐的财务数据

数据源: akshare stock_financial_analysis_indicator (季度)
PIT规则: 财报在报告期末+45天后才可用 (监管截止日保守近似)
"""
import os, json
import pandas as pd
import numpy as np
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "fundamental_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# PIT 延迟: 报告期结束到公告可用的天数
PIT_LAG_DAYS = 45


def fetch_financials(symbol: str) -> pd.DataFrame:
    """获取单只股票的季度财务指标"""
    import akshare as ak
    import warnings
    warnings.filterwarnings('ignore')

    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year='2017')
        if df is None or len(df) == 0:
            return None
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"  [fundamental] {symbol} fetch error: {e}")
        return None


def get_fundamental_factors(symbol: str, today: pd.Timestamp) -> dict:
    """
    获取某日可用的基本面因子 (PIT-safe)。

    返回: {factor_name: value} 或 None
    """
    df = fetch_financials(symbol)
    if df is None or len(df) == 0:
        return None

    # 确保日期格式
    if '日期' not in df.columns:
        return None
    df['日期'] = pd.to_datetime(df['日期'])

    # PIT 过滤: 只使用 today - PIT_LAG_DAYS 之前公布的财报
    cutoff = today - timedelta(days=PIT_LAG_DAYS)
    available = df[df['日期'] <= cutoff]

    if len(available) == 0:
        return None

    latest = available.iloc[-1]  # 最新可用的一份财报

    factors = {}

    # ROE
    roe = safe_float(latest.get('净资产收益率(%)'))
    if roe is not None:
        factors['roe'] = roe

    # 净利润增长率
    profit_g = safe_float(latest.get('净利润增长率(%)'))
    if profit_g is not None:
        factors['profit_growth'] = profit_g

    # 营收增长率
    rev_g = safe_float(latest.get('主营业务收入增长率(%)'))
    if rev_g is not None:
        factors['revenue_growth'] = rev_g

    # 资产负债率
    debt_r = safe_float(latest.get('资产负债率(%)'))
    if debt_r is not None:
        factors['debt_ratio'] = debt_r

    # 每股净资产
    bvps = safe_float(latest.get('每股净资产_调整前(元)'))
    if bvps is not None:
        factors['bvps'] = bvps

    # 每股收益 (TTM: 最近4个季度累计)
    recent_4q = available.tail(4)
    eps_sum = 0
    for _, row in recent_4q.iterrows():
        eps = safe_float(row.get('摊薄每股收益(元)'))
        if eps is not None:
            eps_sum += eps
    factors['eps_ttm'] = eps_sum if eps_sum != 0 else None

    # ROE TTM
    roe_vals = []
    for _, row in recent_4q.iterrows():
        r = safe_float(row.get('净资产收益率(%)'))
        if r is not None:
            roe_vals.append(r)
    if roe_vals:
        factors['roe_ttm'] = np.mean(roe_vals)

    # TTM 增长率 (最近4季度 vs 去年同期)
    if len(available) >= 8:
        recent_4q_profit = available.tail(4)['净利润增长率(%)'].apply(safe_float).dropna()
        prev_4q_profit = available.iloc[-8:-4]['净利润增长率(%)'].apply(safe_float).dropna()
        if len(recent_4q_profit) > 0 and len(prev_4q_profit) > 0:
            factors['profit_growth_ttm'] = float(recent_4q_profit.mean() - prev_4q_profit.mean())

    return factors if factors else None


def safe_float(val):
    """安全转换为float"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def build_fundamental_cache(symbols: list):
    """预拉取所有股票的基本面数据"""
    print(f"[Fundamental] 拉取 {len(symbols)} 只股票...")
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)}")
        fetch_financials(sym)
    print(f"  完成: {len(os.listdir(CACHE_DIR))} 个缓存文件")


if __name__ == "__main__":
    # 测试
    import sys
    from datetime import datetime
    sym = sys.argv[1] if len(sys.argv) > 1 else '000001'
    today = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp('2024-06-30')

    factors = get_fundamental_factors(sym, today)
    print(f"{sym} @ {today.date()}:")
    if factors:
        for k, v in sorted(factors.items()):
            print(f"  {k}: {v}")
    else:
        print("  无可用数据")
