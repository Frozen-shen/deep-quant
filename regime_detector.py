"""
市场状态检测 (Regime Detection)

基于指数均线趋势 + ADX 识别三种市场状态:
  - trend_up: 指数在MA60之上, ADX>20 → 趋势上涨 (动量策略, 持有期延长)
  - trend_down: 指数在MA60之下, ADX>20 → 趋势下跌 (防御策略, 降低仓位)
  - range: ADX<=20 → 震荡 (反转策略, 缩短持有期)

用法:
    detector = RegimeDetector(market="a")
    regime = detector.detect(df_index, today)
    params = detector.get_ranker_params(regime)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"


@dataclass
class RegimeParams:
    """不同市场状态下的策略参数。"""
    top_k: int              # 持仓数量
    hold_thresh: int        # 最小持有期
    sell_rank_buffer: int   # 卖出缓冲
    cost_threshold: float   # 成本门槛
    n_drop: int            # 每次最大替换数

    @classmethod
    def for_regime(cls, regime: "Regime", base_top_k: int = 4) -> "RegimeParams":
        if regime == Regime.TREND_UP:
            return cls(
                top_k=base_top_k,
                hold_thresh=5,           # Phase 3.2: 中性持有期
                sell_rank_buffer=2,       # 标准缓冲
                cost_threshold=0.08,      # 低门槛追涨
                n_drop=2,                 # 可多换
            )
        elif regime == Regime.TREND_DOWN:
            return cls(
                top_k=max(2, base_top_k - 2),  # 少持仓
                hold_thresh=7,           # Phase 3.2: 多拿少动
                sell_rank_buffer=3,       # 宽缓冲防止恐慌卖
                cost_threshold=0.15,      # 高门槛减少交易
                n_drop=1,                 # 少换
            )
        else:  # RANGE
            return cls(
                top_k=base_top_k,
                hold_thresh=5,           # 默认
                sell_rank_buffer=2,       # 默认
                cost_threshold=0.15,      # 默认
                n_drop=2,                 # 默认
            )


class RegimeDetector:
    """
    市场状态检测器。

    检测逻辑:
      1. 计算指数 MA60
      2. 计算 ADX(14)
      3. 分类:
         - MA60上升 + ADX>20 → TREND_UP
         - MA60下降 + ADX>20 → TREND_DOWN
         - ADX<=20 → RANGE
    """

    def __init__(self, market: str = "a"):
        self.market = market
        self._index_data: Optional[pd.DataFrame] = None
        self._ma60: Optional[pd.Series] = None
        self._adx: Optional[pd.Series] = None

    def load_index_data(self) -> bool:
        """加载指数数据 (上证综指 / 恒生指数)。"""
        try:
            from data_fetcher import DataFetcher

            if self.market == "hk":
                symbol = "HSI"
            else:
                symbol = "000001"

            df = DataFetcher.fetch(symbol, start_date="20180101", end_date="20260722")
            if df is None or len(df) < 120:
                print(f"  [Regime] 无法加载指数 {symbol} 数据")
                return False

            self._index_data = df

            # MA60
            if "close" in df.columns:
                self._ma60 = df["close"].rolling(60).mean()

            # ADX(14)
            if all(c in df.columns for c in ["high", "low", "close"]):
                self._adx = self._calc_adx(df, period=14)

            return len(df) > 0
        except Exception as e:
            print(f"  [Regime] 指数数据加载失败: {e}")
            return False

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ADX (Average Directional Index)。"""
        high, low, close = df["high"], df["low"], df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # +DM / -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Smoothed DM
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # DX → ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        adx = dx.rolling(period).mean()

        return adx

    def detect(self, today) -> "Regime":
        """
        根据给定日期检测市场状态。

        Args:
          today: 日期 (str or Timestamp)

        Returns:
          Regime 枚举值
        """
        if self._index_data is None or self._ma60 is None:
            return Regime.RANGE  # fallback: neutral

        today_ts = pd.Timestamp(today)
        idx_before = self._index_data["date"] <= today_ts
        if not idx_before.any():
            return Regime.RANGE

        last_idx = self._index_data.index[idx_before][-1]

        # MA60 方向: 近5日MA60斜率
        ma60_series = self._ma60.loc[:last_idx]
        if len(ma60_series) < 65:
            return Regime.RANGE

        ma60_now = ma60_series.iloc[-1]
        ma60_5d_ago = ma60_series.iloc[-6] if len(ma60_series) >= 6 else ma60_series.iloc[0]
        ma60_slope = (ma60_now - ma60_5d_ago) / (ma60_5d_ago + 1e-12)

        # ADX
        adx_now = 0.0
        if self._adx is not None:
            adx_series = self._adx.loc[:last_idx]
            if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]):
                adx_now = adx_series.iloc[-1]

        # Classification
        if adx_now <= 20:
            return Regime.RANGE
        elif ma60_slope > 0.002:  # MA60 上行
            return Regime.TREND_UP
        elif ma60_slope < -0.002:  # MA60 下行
            return Regime.TREND_DOWN
        else:
            return Regime.RANGE

    def get_ranker_params(self, regime: "Regime", base_top_k: int = 4) -> dict:
        """
        获取给定 regime 下的 PortfolioRanker 参数。

        Returns:
          dict of kwargs for PortfolioRanker constructor
        """
        p = RegimeParams.for_regime(regime, base_top_k)
        return {
            "top_k": p.top_k,
            "hold_thresh": p.hold_thresh,
            "sell_rank_buffer": p.sell_rank_buffer,
            "cost_threshold": p.cost_threshold,
            "n_drop": p.n_drop,
        }
