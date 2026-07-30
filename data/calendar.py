"""
A股交易日历 — 替代 set().union() 土办法

用法:
  from data.calendar import get_trading_days
  days = get_trading_days("2018-01-01", "2026-07-10")  # 交易所真实交易日列表
"""
import pandas as pd
import os

# 缓存文件
_CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "cache", "trading_calendar.csv")


def _fetch_from_akshare():
    """从 akshare 获取交易日历。"""
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(_CALENDAR_PATH, index=False)
        return set(df["date"].tolist())
    except:
        return None


def get_trading_days(start: str, end: str) -> list:
    """获取指定区间的交易日列表 (按交易所真实日历)。"""
    days = set()
    if os.path.exists(_CALENDAR_PATH):
        df = pd.read_csv(_CALENDAR_PATH)
        days = set(pd.to_datetime(df["date"]).tolist())
    else:
        days = _fetch_from_akshare()
        if days is None:
            # 回退: 用全集日期
            return []

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return sorted([d for d in days if start_dt <= d <= end_dt])
