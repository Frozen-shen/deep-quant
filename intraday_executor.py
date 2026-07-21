"""
日内执行引擎 — VWAP成交 + 分时执行 + 日内止损

替代 "次日开盘价成交" 的简化假设:
  - VWAP执行: 在N分钟内分批成交,获得成交量加权均价
  - 时间切片: 前30分钟/前1小时/全天的VWAP
  - 日内止损: 每5分钟检查一次,触发即平仓

用法:
  from intraday_executor import execute_vwap
  fill_price = execute_vwap(symbol, trade_date, action="BUY", minutes=30)
"""

import pandas as pd
import numpy as np
from intraday_fetcher import IntradayFetcher


class IntradayExecutor:
    """
    日内执行器 — 用5分钟K线优化成交价格。
    """

    def __init__(self, scale: int = 5):
        self.fetcher = IntradayFetcher()
        self.scale = scale

    def execute_vwap(self, symbol: str, trade_date: str,
                     action: str = "BUY", minutes: int = 30) -> float:
        """
        VWAP执行: 在指定时间内按成交量加权成交。

        参数:
          symbol: 股票代码
          trade_date: 交易日期 YYYY-MM-DD
          action: BUY/SELL
          minutes: 执行时间段(前N分钟)

        返回: 加权成交价
        """
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < 3:
            return None  # 数据不足,回退到日线

        # 取前N分钟
        n_bars = max(2, minutes // self.scale)
        df_slice = df.head(n_bars)

        vol = df_slice["volume"].fillna(0)
        price = (df_slice["high"] + df_slice["low"] + df_slice["close"]) / 3  # 典型价

        if vol.sum() == 0:
            return float(df_slice["close"].mean())

        # VWAP = sum(price * vol) / sum(vol)
        vwap = (price * vol).sum() / vol.sum()

        # BUY加滑点, SELL减滑点
        slippage = 0.001 if action == "BUY" else -0.001
        return float(vwap * (1 + slippage))

    def execute_time_sliced(self, symbol: str, trade_date: str,
                            action: str = "BUY", slices: int = 3) -> list:
        """
        时间切片执行: 将订单分成N份,在不同时段成交。

        返回: [(时间, 成交价), ...]
        """
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < slices * 3:
            return [(df["datetime"].iloc[-1], float(df["close"].mean()))]

        n = len(df)
        slice_size = n // slices
        results = []
        for i in range(slices):
            start = i * slice_size
            end = start + slice_size
            chunk = df.iloc[start:end]
            avg_price = float(chunk["close"].mean())
            results.append((chunk["datetime"].iloc[-1], avg_price))
        return results

    def intraday_stop_check(self, symbol: str, trade_date: str,
                            entry_price: float, stop_pct: float = 0.03) -> bool:
        """
        日内止损检查: 遍历5分钟K线,检测是否触发止损。

        返回: True=触发止损
        """
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < 5:
            return False

        for _, bar in df.iterrows():
            low = bar["low"]
            if entry_price > 0 and (low / entry_price - 1) < -stop_pct:
                return True
        return False

    def get_best_execution_price(self, symbol: str, trade_date: str,
                                  action: str = "BUY") -> float:
        """
        智能执行: 优先用VWAP,数据不足回退到日线开盘价。

        返回: 最优成交价
        """
        vwap = self.execute_vwap(symbol, trade_date, action, minutes=30)
        if vwap is not None:
            return vwap

        # 回退: 用5分钟开盘价
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) > 0:
            return float(df["open"].iloc[0])

        return None  # 完全无数据,跳过
