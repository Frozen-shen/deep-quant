"""
宏观叠加层 — 大盘环境 + 北向资金 → market_score

数据源:
  - 上证/恒指 K线 (新浪API) → 技术面评分
  - 北向资金 (akshare) → 资金方向评分
  - 综合: market_score (-1熊 ~ +1牛)

用法:
    overlay = MacroOverlay(market="hk")
    overlay.update()  # 拉最新数据
    adjusted = overlay.apply(factor_score)  # 叠加到个股信号
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime


class MacroOverlay:
    """
    宏观叠加层。

    计算大盘技术面 + 北向资金趋势 → market_score。
    应用到个股: factor_score *= (1 + market_score * strength)
    """

    def __init__(self, market: str = "hk", strength: float = 0.4):
        self.market = market
        self.strength = strength  # 宏观影响强度 (±40%)
        self._market_score = 0.0
        self._details = {}

    # ================================================================
    #  数据获取
    # ================================================================
    def _fetch_index(self) -> pd.DataFrame:
        """拉取大盘指数日线。"""
        try:
            if self.market == "hk":
                df = ak.stock_hk_index_daily_sina(symbol="HSI")
            else:
                df = ak.stock_zh_index_daily(symbol="sh000001")
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date")
        except Exception as e:
            print(f"[Macro] 指数获取失败: {e}")
            return pd.DataFrame()

    def _fetch_north_flow(self) -> pd.DataFrame:
        """拉取北向资金数据。"""
        try:
            df = ak.stock_hsgt_hist_em(symbol="沪股通")
            df["date"] = pd.to_datetime(df["日期"])
            df["net_flow"] = pd.to_numeric(df.get("当日成交净买额", 0), errors="coerce").fillna(0)
            return df.sort_values("date")
        except Exception as e:
            print(f"[Macro] 北向资金获取失败: {e}")
            return pd.DataFrame()

    # ================================================================
    #  评分计算
    # ================================================================
    def update(self) -> dict:
        """拉取最新数据并计算当前 market_score。同时缓存全量数据供历史回放。"""
        self._index_cache = self._fetch_index()
        self._north_cache = self._fetch_north_flow()
        self._market_score = self.score_at(datetime.now())
        
        # 打印
        regime = "🟢牛市" if self._market_score > 0.3 else ("🔴熊市" if self._market_score < -0.3 else "🟡震荡")
        print(f"[Macro] {self.market}: {regime} market_score={self._market_score:+.2f}")
        return {"market_score": self._market_score, "regime": self.regime}

    # ================================================================
    #  叠加到个股信号
    # ================================================================
    def score_at(self, date) -> float:
        """
        计算指定日期的宏观评分 (用于历史回放)。
        
        参数: date (str或Timestamp)
        返回: market_score (-1~1)
        """
        date = pd.Timestamp(date)
        
        score = 0.0
        df_idx = self._index_cache
        if df_idx is not None and len(df_idx) > 60:
            # 只用到 date 为止的数据
            hist = df_idx[df_idx["date"] <= date]
            if len(hist) < 60:
                return 0.0
            close = hist["close"]
            last = close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            ma120 = close.rolling(120).mean().iloc[-1] if len(hist) >= 120 else ma60
            
            idx_score = 0.0
            if last > ma20: idx_score += 0.2
            if last > ma60: idx_score += 0.3
            if last > ma120: idx_score += 0.2
            if ma20 > ma60 > ma120: idx_score += 0.2
            
            peak_60 = close.tail(60).max()
            dd_60 = (last / peak_60 - 1) if peak_60 > 0 else 0
            if dd_60 < -0.1: idx_score -= 0.3
            
            ret = close.pct_change().dropna()
            if len(ret) >= 20:
                vol = ret.tail(20).std() * np.sqrt(252)
                if vol > 0.4: idx_score -= 0.2
                elif vol > 0.3: idx_score -= 0.1
            
            idx_score = max(-1.0, min(1.0, idx_score))
            score += idx_score * 0.6

        # 北向资金评分 (到 date 为止)
        df_north = self._north_cache
        if df_north is not None and len(df_north) > 10:
            hist = df_north[df_north["date"] <= date].tail(20)
            if len(hist) >= 5:
                net_sum = hist["net_flow"].sum()
                north_score = 0.0
                if net_sum > 50: north_score += 0.3
                if net_sum > 100: north_score += 0.2
                if net_sum < -50: north_score -= 0.3
                if net_sum < -100: north_score -= 0.3
                north_score = max(-1.0, min(1.0, north_score))
                score += north_score * 0.4

        return max(-1.0, min(1.0, score))
        """
        将宏观评分叠加到个股因子分数。

        牛市(score>0): 放大买入信号, 压制卖出信号
        熊市(score<0): 压制买入信号, 放大卖出信号
        """
        boost = self._market_score * self.strength
        return factor_score * (1 + boost)

    @property
    def market_score(self) -> float:
        return self._market_score

    @property
    def regime(self) -> str:
        if self._market_score > 0.3: return "牛市"
        elif self._market_score < -0.3: return "熊市"
        return "震荡"
