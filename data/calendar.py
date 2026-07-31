"""
A股交易日历 — 替代 set().union() 土办法

用法:
  from data.calendar import get_trading_days, is_trading_day, next_trading_day
  days = get_trading_days("2018-01-01", "2026-07-10")  # 交易所真实交易日列表
  is_td = is_trading_day(pd.Timestamp("2026-08-03"))     # 检查是否为交易日
"""
import pandas as pd
import os
from typing import Optional, List

# 缓存文件
_CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "cache", "trading_calendar.csv")

# 全局缓存
_trading_days_set: Optional[set] = None
_trading_days_list: Optional[List[pd.Timestamp]] = None


def _fetch_from_akshare():
    """从 akshare 获取交易日历。"""
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(_CALENDAR_PATH, index=False)
        return set(df["date"].tolist()), sorted(df["date"].tolist())
    except:
        return None, None


def _ensure_calendar():
    """确保交易日历已加载到全局缓存。"""
    global _trading_days_set, _trading_days_list
    if _trading_days_set is not None:
        return

    if os.path.exists(_CALENDAR_PATH):
        df = pd.read_csv(_CALENDAR_PATH)
        _trading_days_set = set(pd.to_datetime(df["date"]).tolist())
        _trading_days_list = sorted(_trading_days_set)
    else:
        s, l = _fetch_from_akshare()
        _trading_days_set = s if s else set()
        _trading_days_list = l if l else []


def get_trading_days(start: str, end: str) -> list:
    """获取指定区间的交易日列表 (按交易所真实日历)。"""
    _ensure_calendar()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return sorted([d for d in _trading_days_list if start_dt <= d <= end_dt])


def is_trading_day(date=None) -> bool:
    """检查是否为交易日。"""
    _ensure_calendar()
    if date is None:
        date = pd.Timestamp.now().normalize()
    else:
        date = pd.Timestamp(date)
    return date in _trading_days_set


def next_trading_day(date=None, n: int = 1) -> Optional[pd.Timestamp]:
    """获取下 N 个交易日。"""
    _ensure_calendar()
    if date is None:
        date = pd.Timestamp.now().normalize()
    else:
        date = pd.Timestamp(date)

    count = 0
    for d in _trading_days_list:
        if d > date:
            count += 1
            if count == n:
                return d
    return _trading_days_list[-1] if _trading_days_list else date


def prev_trading_day(date=None, n: int = 1) -> Optional[pd.Timestamp]:
    """获取前 N 个交易日。"""
    _ensure_calendar()
    if date is None:
        date = pd.Timestamp.now().normalize()
    else:
        date = pd.Timestamp(date)

    prev = [d for d in _trading_days_list if d < date]
    if len(prev) >= n:
        return prev[-n]
    return prev[0] if prev else date
