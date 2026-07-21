"""
日内数据获取 — 新浪5/15/30/60分钟K线

用法:
  from intraday_fetcher import IntradayFetcher
  fetcher = IntradayFetcher()
  df = fetcher.fetch("600519", "2026-07-10", scale=5)
"""

import requests
import pandas as pd
from datetime import datetime


class IntradayFetcher:
    """
    新浪分时K线数据获取器。

    支持 scale: 5, 15, 30, 60 (分钟)
    """

    BASE_URL = (
        "https://money.finance.sina.com.cn/quotes_service/api/"
        "json_v2.php/CN_MarketData.getKLineData"
    )

    @staticmethod
    def _to_sina_symbol(symbol: str) -> str:
        """600519 → sh600519, 000001 → sz000001"""
        if symbol.startswith(("sh", "sz")):
            return symbol
        prefix = "sh" if symbol.startswith(("6", "68")) else "sz"
        return f"{prefix}{symbol}"

    def fetch(self, symbol: str, trade_date: str = None,
              scale: int = 5, datalen: int = 48) -> pd.DataFrame:
        """
        拉取分钟K线。

        参数:
          symbol: 股票代码
          trade_date: 交易日期 (YYYY-MM-DD), 默认今天
          scale: 分钟周期 (5/15/30/60)
          datalen: 返回条数

        返回: DataFrame [datetime, open, high, low, close, volume]
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        sina_sym = self._to_sina_symbol(symbol)

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "symbol": sina_sym,
                    "scale": scale,
                    "datalen": datalen,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[Intraday] {symbol} 获取失败: {e}")
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "day": "datetime", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })

        # 类型转换
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    def fetch_intraday_factors(self, symbol: str, trade_date: str = None,
                               scale: int = 5) -> dict:
        """
        计算日内因子。

        返回: {factor_name: value}
        """
        df = self.fetch(symbol, trade_date, scale)
        if len(df) < 10:
            return {}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        n = len(df)

        factors = {}

        # 日内波动率
        factors["intraday_vol"] = float((high.max() / low.min() - 1) * 100)

        # VWAP偏离
        if vol.sum() > 0:
            vwap = (close * vol).sum() / vol.sum()
            factors["vwap_deviation"] = float(close.iloc[-1] / vwap - 1)

        # 尾盘效应 (最后30分钟 vs 前段)
        tail_n = max(1, n // 5)
        factors["tail_return"] = float(close.iloc[-1] / close.iloc[-tail_n] - 1) if n >= tail_n else 0

        # 开盘跳空 (vs 昨日收盘 — 需要日线配合)
        factors["open_change"] = float(close.iloc[0] / close.iloc[-1] - 1)

        # 日内趋势 (开盘到收盘)
        factors["intraday_trend"] = float(close.iloc[-1] / close.iloc[0] - 1)

        # 量比
        if len(vol) >= 5:
            factors["intraday_vol_ratio"] = float(vol.iloc[-1] / vol.tail(5).mean())

        return factors
