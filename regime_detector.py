"""
市场状态检测 (Regime Detection) + 因子权重自适应

基于指数均线趋势 + ADX 识别三种市场状态:
  - trend_up: 指数在MA60之上, ADX>20 → 趋势上涨 (动量策略, 持有期延长)
  - trend_down: 指数在MA60之下, ADX>20 → 趋势下跌 (防御策略, 降低仓位)
  - range: ADX<=20 → 震荡 (反转策略, 缩短持有期)

因子权重自适应:
  - trend_up: 正IC因子(动量)权重 ×1.5, 负IC因子(防御)权重 ×0.7
  - trend_down: 正IC因子权重 ×0.5, 负IC因子权重 ×1.3
  - range: 不调整

用法:
    # 从本地 parquet 加载 (推荐, 无需网络)
    detector = RegimeDetector.from_benchmark_parquet("data/cache/index_csi1000.parquet")
    regime = detector.detect(today)
    factors = detector.adapt_factor_weights(factors, today)
    turnover_params = detector.get_turnover_params(today)
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


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
    def for_regime(cls, regime: "Regime", base_top_k: int = 5) -> "RegimeParams":
        """获取Regime下的参数 (与PortfolioRanker.set_regime保持一致)"""
        if regime == Regime.TREND_UP:
            return cls(top_k=base_top_k, hold_thresh=10, sell_rank_buffer=2,
                       cost_threshold=0.06, n_drop=3)
        elif regime == Regime.TREND_DOWN:
            return cls(top_k=base_top_k, hold_thresh=14, sell_rank_buffer=3,
                       cost_threshold=0.15, n_drop=1)
        else:  # RANGE
            return cls(top_k=base_top_k, hold_thresh=10, sell_rank_buffer=2,
                       cost_threshold=0.08, n_drop=3)


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

    @classmethod
    def from_benchmark_parquet(cls, path: str) -> "RegimeDetector":
        """
        从本地 parquet 文件加载基准指数 (无需网络)。

        Args:
          path: parquet 文件路径 (如 data/cache/index_csi1000.parquet)

        Returns:
          已初始化的 RegimeDetector 实例
        """
        detector = cls(market="a")
        if not os.path.exists(path):
            print(f"  [Regime] 基准文件不存在: {path}, 使用 RANGE fallback")
            return detector

        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        if len(df) < 120:
            print(f"  [Regime] 基准数据不足 ({len(df)} 行), 使用 RANGE fallback")
            return detector

        detector._index_data = df

        # MA60
        if "close" in df.columns:
            detector._ma60 = df["close"].rolling(60).mean()

        # ADX(14) — 需要 high/low/close
        if all(c in df.columns for c in ["high", "low", "close"]):
            detector._adx = detector._calc_adx(df, period=14)
        else:
            # 只有 close 时用简化版: 用 close 的绝对变动代替
            detector._adx = None

        return detector

    def load_index_data(self) -> bool:
        """加载指数数据 (上证综指 / 恒生指数)。需要网络。"""
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

    # ── 因子权重自适应 ──

    def adapt_factor_weights(self, factors: List[dict], today,
                             regime: "Regime" = None) -> List[dict]:
        """
        根据市场状态调整因子权重 (核心优化 B)。

        策略:
          - TREND_UP (牛市): 正IC因子(动量)权重 ×1.5, 负IC因子(防御)权重 ×0.7
          - TREND_DOWN (熊市): 正IC因子权重 ×0.5, 负IC因子权重 ×1.3
          - RANGE (震荡): 不调整

        Args:
          factors: [{"name": ..., "icir": ..., "weight_multiplier": ...}, ...]
          today: 当前日期
          regime: 可手动指定状态 (跳过检测)

        Returns:
          调整后的因子列表 (新列表, 不修改原始)
        """
        if regime is None:
            regime = self.detect(today)

        adapted = []
        for f in factors:
            new_f = dict(f)
            icir = f["icir"]
            base_mult = f.get("weight_multiplier", 1.0)

            if regime == Regime.TREND_UP:
                # 牛市: 加强动量(正IC), 弱化防御(负IC)
                if icir > 0:
                    new_f["weight_multiplier"] = base_mult * 1.5
                else:
                    new_f["weight_multiplier"] = base_mult * 0.7
            elif regime == Regime.TREND_DOWN:
                # 熊市: 加强防御(负IC), 弱化动量(正IC)
                if icir > 0:
                    new_f["weight_multiplier"] = base_mult * 0.5
                else:
                    new_f["weight_multiplier"] = base_mult * 1.3
            # RANGE: 不调整

            adapted.append(new_f)

        return adapted

    def get_turnover_params(self, today, regime: "Regime" = None) -> dict:
        """
        根据市场状态返回换手控制参数 (核心优化 A)。

        设计原则:
          - n_drop 不能太小 (否则从0建仓到top_k需要太多周期)
          - cost_threshold 不能太大 (否则无法换仓, 持仓僵化)
          - hold_thresh 控制最短持有期, 防止频繁翻转

        Returns:
          {"hold_thresh": int, "n_drop": int, "cost_threshold": float,
           "sell_rank_buffer": int}
        """
        if regime is None:
            regime = self.detect(today)

        if regime == Regime.TREND_UP:
            # 牛市: 允许更积极调仓, 追涨
            return {"hold_thresh": 10, "n_drop": 10, "cost_threshold": 0.03,
                    "sell_rank_buffer": 3}
        elif regime == Regime.TREND_DOWN:
            # 熊市: 减少换手, 但不是完全锁死
            return {"hold_thresh": 15, "n_drop": 6, "cost_threshold": 0.06,
                    "sell_rank_buffer": 4}
        else:
            # 震荡: 中等
            return {"hold_thresh": 12, "n_drop": 8, "cost_threshold": 0.04,
                    "sell_rank_buffer": 3}
