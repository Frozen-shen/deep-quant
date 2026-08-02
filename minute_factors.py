"""
minute_factors.py — 分钟频因子计算模块

从 5 分钟 K 线聚合为日频因子值，每只股票每天输出一组浮点数。

数据源: Sina (ak.stock_zh_a_minute)
  列: day, open, high, low, close, volume, amount
  频率: 5 分钟 (48 根/天)
  历史深度: ~40 个交易日 (滚动)

因子列表 (10个):
  min_realized_vol      — 已实现波动率 (年化)
  min_realized_skew     — 已实现偏度
  min_vwap_dev          — 收盘 vs VWAP 偏离
  min_tail_return       — 尾盘效应 (最后30min收益)
  min_open_gap          — 开盘跳空
  min_intraday_trend    — 日内趋势 (open→close)
  min_vol_concentration — 成交集中度 (Gini)
  min_large_order_flow  — 大单净流入代理
  min_am_pm_ratio       — 上下午量比
  min_close_strength    — 收盘强度

用法:
  from minute_factors import compute_minute_factors_batch, get_minute_factor_names

  # all_minute_data: {symbol: DataFrame} — 从 fetch_minute_data.py 缓存加载
  factors = compute_minute_factors_batch(all_minute_data, as_of_date="2026-07-31")
  # factors: {symbol: {factor_name: float_value}}
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "minute")

# 模块级缓存 (回测循环中避免重复IO)
_minute_cache: Optional[Dict[str, pd.DataFrame]] = None


# ============================================================
# 因子名称
# ============================================================

MINUTE_FACTOR_NAMES = [
    "min_realized_vol",
    "min_realized_skew",
    "min_vwap_dev",
    "min_tail_return",
    "min_open_gap",
    "min_intraday_trend",
    "min_vol_concentration",
    "min_large_order_flow",
    "min_am_pm_ratio",
    "min_close_strength",
]


def get_minute_factor_names() -> List[str]:
    """返回所有分钟频因子名称。"""
    return list(MINUTE_FACTOR_NAMES)


# ============================================================
# 数据加载
# ============================================================

def load_minute_data(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    加载所有已缓存的分钟数据。

    Args:
      use_cache: 是否使用模块级缓存

    Returns:
      {symbol: DataFrame} — 列: day, open, high, low, close, volume, amount
    """
    global _minute_cache
    if use_cache and _minute_cache is not None:
        return _minute_cache

    data = {}
    if not os.path.exists(MINUTE_CACHE_DIR):
        return data

    for fname in os.listdir(MINUTE_CACHE_DIR):
        if fname.endswith(".parquet"):
            sym = fname.replace(".parquet", "")
            try:
                df = pd.read_parquet(os.path.join(MINUTE_CACHE_DIR, fname))
                if len(df) > 0:
                    # 标准化列名
                    if "day" in df.columns:
                        df["day"] = pd.to_datetime(df["day"])
                    elif "时间" in df.columns:
                        df = df.rename(columns={"时间": "day", "开盘": "open",
                                                "收盘": "close", "最高": "high",
                                                "最低": "low", "成交量": "volume",
                                                "成交额": "amount"})
                        df["day"] = pd.to_datetime(df["day"])
                    data[sym] = df
            except Exception:
                pass

    if use_cache:
        _minute_cache = data
    return data


# ============================================================
# 单日因子计算
# ============================================================

def _gini_coefficient(arr: np.ndarray) -> float:
    """计算 Gini 系数 (0=完全均匀, 1=完全集中)。"""
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _compute_single_day(day_bars: pd.DataFrame, prev_close: Optional[float]) -> dict:
    """
    从单日的 5 分钟 K 线计算所有因子。

    Args:
      day_bars: 当日所有 5min bars (48根), 列: open, high, low, close, volume, amount
      prev_close: 前一交易日收盘价 (用于 open_gap)

    Returns:
      {factor_name: float_value}
    """
    if len(day_bars) < 10:
        return {}

    close = day_bars["close"].values.astype(float)
    open_ = day_bars["open"].values.astype(float)
    high = day_bars["high"].values.astype(float)
    low = day_bars["low"].values.astype(float)
    volume = day_bars["volume"].values.astype(float)

    n_bars = len(close)
    results = {}

    # ── 1. 已实现波动率 (年化) ──
    # 5min returns → std → 年化 (×√(48×252))
    returns_5min = np.diff(close) / close[:-1]
    returns_5min = returns_5min[np.isfinite(returns_5min)]
    if len(returns_5min) > 5:
        results["min_realized_vol"] = float(np.std(returns_5min) * np.sqrt(48 * 252))
    else:
        results["min_realized_vol"] = np.nan

    # ── 2. 已实现偏度 ──
    if len(returns_5min) > 10:
        mean_r = np.mean(returns_5min)
        std_r = np.std(returns_5min)
        if std_r > 1e-9:
            skew = np.mean(((returns_5min - mean_r) / std_r) ** 3)
            results["min_realized_skew"] = float(skew)
        else:
            results["min_realized_skew"] = 0.0
    else:
        results["min_realized_skew"] = np.nan

    # ── 3. VWAP 偏离 ──
    # VWAP = Σ(close × volume) / Σ(volume)
    total_vol = volume.sum()
    if total_vol > 0:
        vwap = np.sum(close * volume) / total_vol
        day_close = close[-1]
        results["min_vwap_dev"] = float((day_close - vwap) / vwap) if vwap > 0 else 0.0
    else:
        results["min_vwap_dev"] = np.nan

    # ── 4. 尾盘效应 (最后6根bar = 30min) ──
    tail_n = min(6, n_bars // 4)  # 至少用 1/4 的数据
    if tail_n >= 2 and n_bars > tail_n:
        tail_start_price = close[-(tail_n + 1)]
        tail_end_price = close[-1]
        results["min_tail_return"] = float((tail_end_price - tail_start_price) / tail_start_price) \
            if tail_start_price > 0 else 0.0
    else:
        results["min_tail_return"] = np.nan

    # ── 5. 开盘跳空 ──
    if prev_close is not None and prev_close > 0:
        results["min_open_gap"] = float((open_[0] - prev_close) / prev_close)
    else:
        results["min_open_gap"] = np.nan

    # ── 6. 日内趋势 (open→close) ──
    if open_[0] > 0:
        results["min_intraday_trend"] = float((close[-1] - open_[0]) / open_[0])
    else:
        results["min_intraday_trend"] = np.nan

    # ── 7. 成交集中度 (Gini) ──
    if volume.sum() > 0:
        results["min_vol_concentration"] = _gini_coefficient(volume)
    else:
        results["min_vol_concentration"] = np.nan

    # ── 8. 大单净流入代理 ──
    # 成交量 > 2×均值的bar视为"大单"，按涨跌方向加总
    mean_vol = volume.mean()
    if mean_vol > 0:
        large_mask = volume > (2 * mean_vol)
        if large_mask.sum() > 0:
            bar_returns = np.diff(close) / close[:-1]
            # 对齐: large_mask 有 n_bars 个, bar_returns 有 n_bars-1 个
            # 用 bar 的 close vs open 判断方向
            bar_direction = np.sign(close - open_)
            large_flow = np.sum(volume[large_mask] * bar_direction[large_mask])
            # 归一化: 除以总成交量
            results["min_large_order_flow"] = float(large_flow / total_vol) if total_vol > 0 else 0.0
        else:
            results["min_large_order_flow"] = 0.0
    else:
        results["min_large_order_flow"] = np.nan

    # ── 9. 上下午量比 ──
    # A股: 上午 9:30-11:30 (24根5min bar), 下午 13:00-15:00 (24根)
    # 近似: 前半 vs 后半
    mid = n_bars // 2
    am_vol = volume[:mid].sum()
    pm_vol = volume[mid:].sum()
    if pm_vol > 0:
        results["min_am_pm_ratio"] = float(am_vol / pm_vol)
    else:
        results["min_am_pm_ratio"] = np.nan

    # ── 10. 收盘强度 ──
    day_high = high.max()
    day_low = low.min()
    day_range = day_high - day_low
    if day_range > 0:
        results["min_close_strength"] = float((close[-1] - day_low) / day_range)
    else:
        results["min_close_strength"] = 0.5  # 无波动时居中

    return results


# ============================================================
# 多日聚合
# ============================================================

def compute_minute_factors(minute_df: pd.DataFrame, as_of_date,
                           lookback: int = 20) -> dict:
    """
    计算单只股票的分钟频因子 (多日聚合)。

    对最近 lookback 天的日频因子取均值，得到稳定的截面信号。

    Args:
      minute_df: 5分钟K线 DataFrame (列: day, open, high, low, close, volume, amount)
      as_of_date: 截止日期 (只用 <= 此日期的数据)
      lookback: 回看天数

    Returns:
      {factor_name: float_value} — 或 {} 如果数据不足
    """
    if minute_df is None or len(minute_df) == 0:
        return {}

    as_of = pd.Timestamp(as_of_date)
    df = minute_df[minute_df["day"] <= as_of].copy()

    if len(df) == 0:
        return {}

    # 按日分组
    df["trade_date"] = df["day"].dt.date
    trade_dates = sorted(df["trade_date"].unique())

    # 只取最近 lookback 天
    trade_dates = trade_dates[-lookback:]
    if len(trade_dates) < 5:  # 至少5天数据
        return {}

    # 逐日计算
    daily_factors = []
    prev_close = None

    for td in trade_dates:
        day_bars = df[df["trade_date"] == td].reset_index(drop=True)
        factors = _compute_single_day(day_bars, prev_close)
        if factors:
            daily_factors.append(factors)
            prev_close = float(day_bars["close"].iloc[-1])
        else:
            prev_close = float(day_bars["close"].iloc[-1]) if len(day_bars) > 0 else prev_close

    if len(daily_factors) < 5:
        return {}

    # 聚合: 取均值 (截面因子用均值更稳定)
    result = {}
    for name in MINUTE_FACTOR_NAMES:
        vals = [f[name] for f in daily_factors if name in f and not np.isnan(f.get(name, np.nan))]
        if len(vals) >= 3:
            result[name] = float(np.mean(vals))
        else:
            result[name] = np.nan

    return result


def compute_minute_factors_batch(all_minute_data: Dict[str, pd.DataFrame],
                                 as_of_date, lookback: int = 20) -> Dict[str, dict]:
    """
    批量计算所有股票的分钟频因子。

    Args:
      all_minute_data: {symbol: DataFrame}
      as_of_date: 截止日期
      lookback: 回看天数

    Returns:
      {symbol: {factor_name: float_value}}
    """
    results = {}
    for sym, df in all_minute_data.items():
        factors = compute_minute_factors(df, as_of_date, lookback)
        if factors:
            results[sym] = factors
    return results
