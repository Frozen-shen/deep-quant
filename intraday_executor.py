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

    # ================================================================
    #  L1+L2+L3+L4: 智能执行 + 信号确认 + 风控 + Alpha
    # ================================================================

    def get_best_execution_price(self, symbol: str, trade_date: str,
                                  action: str = "BUY", signal_strength: float = 0.0
                                  ) -> float:
        """智能执行入口: 强信号抢筹,弱信号等回调。"""
        vwap = self.adaptive_execute(symbol, trade_date, action, signal_strength)
        if vwap is not None:
            return vwap
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) > 0:
            return float(df["open"].iloc[0])
        return None

    def adaptive_execute(self, symbol: str, trade_date: str, action: str,
                         signal_strength: float = 0.0) -> float:
        """信号>0.5抢筹, 0.15~0.5VWAP, <0.15等回调。"""
        if abs(signal_strength) > 0.5:
            return self.execute_vwap(symbol, trade_date, action, minutes=10)
        elif abs(signal_strength) > 0.15:
            return self.execute_vwap(symbol, trade_date, action, minutes=30)
        else:
            return self.execute_vwap(symbol, trade_date, action, minutes=240)

    def confirm_signal(self, symbol: str, trade_date: str,
                       action: str = "BUY") -> tuple:
        """开盘跳空+量比验证日线信号。返回(confirmed,adjustment,reason)。"""
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < 10:
            return True, 1.0, "数据不足"
        open_px, prev = df["open"].iloc[0], df["close"].iloc[0]
        gap = (open_px / prev - 1) if prev > 0 else 0
        vr = df["volume"].head(1).sum() / df["volume"].mean() if df["volume"].mean() > 0 else 1.0
        if action == "BUY":
            if gap > 0.03: return True, 0.5, f"高开{gap*100:.1f}%减半"
            if gap < -0.02: return False, 0.0, f"低开{gap*100:.1f}%观望"
            if vr < 0.5: return True, 0.5, f"缩量减半"
            if vr > 2.0: return True, 1.2, f"放量加仓"
            return True, 1.0, "确认"
        if action == "SELL" and gap < -0.03:
            return True, 1.2, "低开加速卖"
        return True, 1.0, "通过"

    def intraday_risk_check(self, symbol: str, trade_date: str,
                            entry_price: float, time_stop_hour: int = 11) -> dict:
        """时间止损+尾盘清仓。返回{stop,reason,action}。"""
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < 5:
            return {"stop": False, "reason": "", "action": "hold"}
        morning = df[df["datetime"].dt.hour < time_stop_hour]
        if len(morning) > 0 and entry_price > 0:
            loss = float(morning["close"].iloc[-1] / entry_price - 1)
            if loss < -0.01:
                return {"stop": True, "reason": f"11:00跌{loss*100:.1f}%", "action": "reduce"}
        afternoon = df[df["datetime"].dt.hour >= 14]
        if len(afternoon) > 0 and entry_price > 0:
            loss = float(afternoon["close"].iloc[-1] / entry_price - 1)
            if loss < -0.005:
                return {"stop": True, "reason": f"尾盘清{loss*100:.1f}%", "action": "close"}
        return {"stop": False, "reason": "", "action": "hold"}

    def compute_alpha_factors(self, symbol: str, trade_date: str) -> dict:
        """5个日内Alpha因子。"""
        df = self.fetcher.fetch(symbol, trade_date, scale=self.scale)
        if len(df) < 20: return {}
        c, v, n = df["close"], df["volume"], len(df)
        f = {}
        f["opening_gap"] = float(df["open"].iloc[0] / c.iloc[0] - 1)
        f["morning_vol_ratio"] = float(v.head(min(6,n)).mean() / v.mean())
        h = n // 2
        if h > 0:
            f["afternoon_reversal"] = float(c.iloc[-1]/c.iloc[h]-1) - float(c.iloc[h-1]/c.iloc[0]-1)
        if v.sum() > 0:
            f["vwap_position"] = float(c.iloc[-1] / ((c*v).sum()/v.sum()) - 1)
        vr = v / v.rolling(5).mean().fillna(1)
        f["large_order_bars"] = float((vr > 3).sum() / n)
        return f

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
