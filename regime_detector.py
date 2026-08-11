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

    # 因子权重乘数 profile: (up_pos, up_neg, down_pos, down_neg)
    REGIME_PROFILES = {
        "original":    (2.0, 0.3, 0.5, 1.5),
        "conservative": (1.3, 0.7, 0.7, 1.3),
        "aggressive":  (3.0, 0.1, 0.3, 2.0),
        "disabled":    (1.0, 1.0, 1.0, 1.0),  # 等同于不做 regime 调整
    }

    def __init__(self, market: str = "a", profile: str = "conservative",
                 vol_source: str = "daily"):
        self.market = market
        self.profile = profile
        # 波动率来源: daily=指数日收益60日std (原逻辑), rv_5m=市场截面已实现波动率
        # (build_market_rv.py 产物, 5m 数据 2022 起, 之前回退 daily)
        self.vol_source = vol_source
        self._index_data: Optional[pd.DataFrame] = None
        self._ma60: Optional[pd.Series] = None
        self._adx: Optional[pd.Series] = None
        self._market_rv: Optional[pd.DataFrame] = None  # date, rv_median, n_stocks
        self._market_rv_vol60: Optional[pd.Series] = None
        self._market_rv_pct: Optional[pd.Series] = None  # 滚动252日分位 (方案A v23)

    @classmethod
    def from_benchmark_parquet(cls, path: str, profile: str = "conservative",
                               vol_source: str = "daily") -> "RegimeDetector":
        """
        从本地 parquet 文件加载基准指数 (无需网络)。

        Args:
          path: parquet 文件路径 (如 data/cache/index_csi1000.parquet)
          profile: regime 参数 profile (conservative/original/aggressive/disabled)
          vol_source: 波动率来源 daily=日收益std / rv_5m=市场截面已实现波动率
            (build_market_rv.py 产物, 5m 数据 2022 起, 之前回退 daily)

        Returns:
          已初始化的 RegimeDetector 实例
        """
        detector = cls(market="a", profile=profile, vol_source=vol_source)
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

        # 市场已实现波动率 (vol_source=rv_5m 时使用; 文件不存在则回退 daily)
        detector._load_market_rv()
        return detector

    def _load_market_rv(self):
        """加载市场已实现波动率序列 (build_market_rv.py 产物)。

        路径: data/cache/market_rv_5m.parquet (date, rv_median, n_stocks)
        5m 数据 2022-01 起; 文件缺失/vol_source=daily 时静默跳过,
        detect_v2 自动回退日收益波动率。

        校准 (方案A v23, 2026-08-11): rv_5m 与 daily 的分布量纲不同
        (rv_5m 均值 0.310/std 0.044 vs daily 0.238/0.071), 固定阈值
        0.30/0.70/0.85 是为 daily 校准的, 直接套用会导致 2025-26 全程
        误判"低波弹性" (v20 模拟考 -2.2pp 根因)。修复: 用滚动 126 日
        分位 (当前值在过去半年的位置, min_periods=60 使 2022-08 起生效,
        覆盖 fold_3 后半段; 252 日窗口会让 2023 前全部回退 daily, 弃用)。
        """
        if self.vol_source != "rv_5m":
            return
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "cache", "market_rv_5m.parquet")
        if not os.path.exists(path):
            return
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            rv = df.set_index("date")["rv_median"]
            # 60日滚动均值 (语义对齐日收益逻辑的 vol60)
            vol60 = rv.rolling(60).mean()
            # 滚动 126 日分位: 当前 vol60 在过去半年中的百分位 (PIT 安全,
            # 只用 ≤ 当天的数据)。min_periods=60: 早期数据不足时回退 daily。
            roll_pct = vol60.rolling(126, min_periods=60).apply(
                lambda x: float((x <= x[-1]).mean()), raw=True)
            self._market_rv_vol60 = vol60
            self._market_rv_pct = roll_pct
            self._market_rv = df
        except Exception:
            self._market_rv = None
            self._market_rv_vol60 = None
            self._market_rv_pct = None

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
        根据市场状态调整因子权重 (优化 B v3 — 软权重, 不做硬筛选)。

        策略 (由 self.profile 控制):
          - conservative (默认): 温和调整, 鲁棒性最佳
          - original: 原始激进参数 (过拟合风险高)
          - aggressive: 极端调整
          - disabled: 不调整 (纯因子 alpha)

        不做硬筛选 (避免regime切换时全仓换血导致换手率爆炸)。

        Args:
          factors: [{"name": ..., "icir": ..., "weight_multiplier": ...}, ...]
          today: 当前日期
          regime: 可手动指定状态 (跳过检测)

        Returns:
          调整后的因子列表 (新列表, 不修改原始)
        """
        if regime is None:
            regime = self.detect(today)

        up_pos, up_neg, down_pos, down_neg = self.REGIME_PROFILES.get(
            self.profile, self.REGIME_PROFILES["conservative"])

        adapted = []
        for f in factors:
            new_f = dict(f)
            icir = f["icir"]
            base_mult = f.get("weight_multiplier", 1.0)

            if regime == Regime.TREND_UP:
                if icir > 0:
                    new_f["weight_multiplier"] = base_mult * up_pos
                else:
                    new_f["weight_multiplier"] = base_mult * up_neg
            elif regime == Regime.TREND_DOWN:
                if icir > 0:
                    new_f["weight_multiplier"] = base_mult * down_pos
                else:
                    new_f["weight_multiplier"] = base_mult * down_neg
            # RANGE: 不调整

            adapted.append(new_f)

        return adapted

    def get_turnover_params(self, today, regime: "Regime" = None) -> dict:
        """
        根据市场状态返回换手控制参数 (优化 C: 适度建仓)。

        设计原则:
          - n_drop=8: 允许较快建仓但不至于全仓换血
          - hold_thresh 适中: 防止频繁翻转
          - cost_threshold 低: 确保信号能执行

        Returns:
          {"hold_thresh": int, "n_drop": int, "cost_threshold": float,
           "sell_rank_buffer": int}
        """
        if regime is None:
            regime = self.detect(today)

        if regime == Regime.TREND_UP:
            # 牛市: 较快调仓
            return {"hold_thresh": 8, "n_drop": 8, "cost_threshold": 0.02,
                    "sell_rank_buffer": 2}
        elif regime == Regime.TREND_DOWN:
            # 熊市: 适度保守
            return {"hold_thresh": 12, "n_drop": 5, "cost_threshold": 0.04,
                    "sell_rank_buffer": 3}
        else:
            # 震荡: 中等
            return {"hold_thresh": 10, "n_drop": 6, "cost_threshold": 0.03,
                    "sell_rank_buffer": 2}

    # ── 双变量检测 + 动量崩溃保护 (方案C v5, 新增 — 不替代上述旧方法) ──

    def detect_v2(self, date_str: str) -> tuple:
        """
        双变量市场状态检测 (方案C v5):
          趋势: 指数 vs MA20/MA60
          波动率: 60日已实现波动率的滚动分位
        返回: (regime, volatility_pctile)
        """
        if self._index_data is None:
            return (Regime.RANGE, 0.5)
        df = self._index_data
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"]
        target = pd.Timestamp(date_str)
        hist = s[s.index <= target]
        if len(hist) < 90:
            return (Regime.RANGE, 0.5)
        price = hist.iloc[-1]
        ma20 = hist.rolling(20).mean().iloc[-1]
        ma60 = hist.rolling(60).mean().iloc[-1]

        # ── 波动率百分位 ──
        # vol_source=rv_5m: 市场截面已实现波动率 (5m, 2022 起, 更精确/响应更快)
        # 数据不足 (2022 前) 或文件缺失 → 回退日收益 60日 std (原逻辑)
        vol_pct = None
        if (self._market_rv is not None
                and self._market_rv_vol60 is not None
                and self._market_rv_pct is not None):
            # 滚动 252 日分位 (方案A v23 校准): 当前 vol60 在过去一年中的位置,
            # 阈值 0.30/0.70/0.85 语义与 daily 路径对齐 (daily 也用历史分位)。
            # PIT: 滚动分位序列只用 ≤ 当天的数据。
            pct_hist = self._market_rv_pct
            pct_hist = pct_hist[pct_hist.index <= target]
            if len(pct_hist) >= 30 and np.isfinite(pct_hist.iloc[-1]):
                vol_pct = float(pct_hist.iloc[-1])

        if vol_pct is None:
            # 60日已实现波动率 (年化)
            rets = hist.pct_change().dropna()
            vol60 = rets.tail(60).std() * np.sqrt(252)
            # 波动率滚动分位 (用全部历史)
            rolling_vol = rets.rolling(60).std() * np.sqrt(252)
            vol_pct = (rolling_vol <= vol60).mean() if rolling_vol.notna().sum() > 20 else 0.5
        # 趋势判定
        if price > ma20 > ma60 and vol_pct < 0.70:
            regime = Regime.TREND_UP
        elif price < ma20 < ma60 or vol_pct > 0.85:
            regime = Regime.TREND_DOWN
        else:
            regime = Regime.RANGE
        return (regime, float(vol_pct))

    def get_weight_multipliers(self, date_str: str) -> dict:
        """
        风格轮动权重乘数 (方案C v5):
          trend_up:   动量×2.0 反转×0.7 价值×1.0
          range:      反转×1.2 价值×1.0 动量×0.8
          trend_down: 反转×1.5 价值×1.3 动量×0.3
        动量崩溃保护: 从高点回撤>15%且波动率>85分位 → 动量×0
        """
        regime, vol_pct = self.detect_v2(date_str)
        if regime == Regime.TREND_UP:
            base = {"momentum": 2.0, "reversal": 0.7, "value": 1.0, "quality": 1.0}
        elif regime == Regime.TREND_DOWN:
            base = {"momentum": 0.3, "reversal": 1.5, "value": 1.3, "quality": 1.3}
        else:
            base = {"momentum": 0.8, "reversal": 1.2, "value": 1.0, "quality": 1.0}
        # 动量崩溃保护 (Daniel & Moskowitz 2016)
        if self._index_data is not None:
            df = self._index_data
            s = df.set_index(pd.to_datetime(df["date"]))["close"]
            hist = s[s.index <= pd.Timestamp(date_str)]
            if len(hist) > 20:
                peak = hist.max()
                dd = hist.iloc[-1] / peak - 1
                if dd < -0.15 and vol_pct > 0.85:
                    base["momentum"] = 0.0
        return base
