"""
选股过滤器 — 只在策略擅长的股票上交易

规则:
  · 排除极端趋势型(60日收益>150%或<-40%): 追不上/空仓更好
  · 排除无趋势型(ADX<15): 震荡磨损
  · 保留温和趋势型: 策略的有效区间
  · 每60天重新评估,动态调整标的池
"""

import pandas as pd
import numpy as np


class StockFilter:
    """
    选股过滤器。

    用法:
        f = StockFilter()
        ok, reason = f.check(df, symbol)
        if not ok: 跳过这只股票
    """

    def __init__(self, max_60d_return: float = 2.0, min_60d_return: float = -0.5,
                 min_adx: float = 12, lookback: int = 60):
        self.max_60d_return = max_60d_return    # >150% → 太极端
        self.min_60d_return = min_60d_return    # <-40% → 崩溃中
        self.min_adx = min_adx                  # <15 → 无趋势
        self.lookback = lookback

    def check(self, df: pd.DataFrame, symbol: str = "") -> (bool, str):
        """
        检查股票是否适合交易。

        返回: (pass: bool, reason: str)
        """
        if len(df) < self.lookback:
            return False, f"数据不足({len(df)}<{self.lookback})"

        close = df["close"]
        ret_60d = close.iloc[-1] / close.iloc[-min(self.lookback, len(df))] - 1

        # 极端涨跌 — 追不上/不需要交易
        if ret_60d > self.max_60d_return:
            return False, f"60日涨幅{ret_60d*100:+.0f}%>150%,极端单边涨,追不上"
        if ret_60d < self.min_60d_return:
            return False, f"60日跌幅{ret_60d*100:+.0f}%<-40%,崩溃中,空仓更好"

        # ADX趋势强度
        adx_val = 0
        if all(c in df.columns for c in ["high", "low"]):
            from indicators import ADX
            adx_series = ADX(df["high"], df["low"], close, 14)
            adx_val = adx_series.iloc[-1] if not adx_series.isna().all() else 0

        if adx_val < self.min_adx:
            return False, f"ADX={adx_val:.1f}<{self.min_adx},无趋势,震荡磨损"

        return True, f"OK(ret_60d={ret_60d*100:+.0f}%, ADX={adx_val:.1f})"

    def filter_stocks(self, stock_data: dict) -> dict:
        """
        批量过滤。

        参数: {symbol: df_price}
        返回: {symbol: (pass, reason)}
        """
        results = {}
        for sym, df in stock_data.items():
            results[sym] = self.check(df, sym)
        return results
