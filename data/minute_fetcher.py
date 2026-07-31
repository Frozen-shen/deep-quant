"""
分钟级行情数据拉取模块 — 用于模拟盘执行层的真实成交价模拟

设计原则:
  - 只拉取需要执行交易的股票, 不维护全市场分钟数据
  - 滚动缓存: 保留最近 60 个交易日
  - 5分钟K线 (48根/天, 支持前复权, 比1分钟线历史更长)
  - 按需拉取 + 本地 parquet 缓存

用法:
  from data.minute_fetcher import MinuteFetcher

  mf = MinuteFetcher()
  df = mf.fetch("600519", days=30)         # 拉取最近30天5分钟线
  df = mf.fetch_batch(["600519","000858"]) # 批量拉取
  vwap = mf.get_vwap("600519", "2026-08-03")  # 计算某日VWAP

akshare 接口: ak.stock_zh_a_hist_min_em(symbol, period="5", ...)
返回列: 时间, 开盘, 收盘, 最高, 最低, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 换手率
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "data_cache", "minute")
DEFAULT_PERIOD = "5"        # 5分钟线
DEFAULT_CACHE_DAYS = 60     # 滚动保留天数


def _ensure_cache_dir():
    os.makedirs(MINUTE_CACHE_DIR, exist_ok=True)


class MinuteFetcher:
    """分钟级行情数据获取器。"""

    def __init__(self, period: str = DEFAULT_PERIOD,
                 cache_days: int = DEFAULT_CACHE_DAYS):
        self.period = period
        self.cache_days = cache_days
        _ensure_cache_dir()

    # ════════════════════════════════════════
    #  数据拉取
    # ════════════════════════════════════════

    def _symbol_path(self, symbol: str) -> str:
        return os.path.join(MINUTE_CACHE_DIR, f"{symbol}.parquet")

    def fetch(self, symbol: str, days: int = None,
              end_date: str = None) -> Optional[pd.DataFrame]:
        """
        拉取单只股票的分钟线数据。

        Args:
          symbol: 股票代码 (如 "600519")
          days: 拉取最近多少天 (None=用cache_days)
          end_date: 截止日期 YYYYMMDD (None=今天)

        Returns:
          DataFrame 或 None
        """
        if days is None:
            days = self.cache_days
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        cache_path = self._symbol_path(symbol)

        # ── 先读缓存 ──
        cached = None
        if os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                if len(cached) > 0:
                    cached["时间"] = pd.to_datetime(cached["时间"])
                    latest_cached = cached["时间"].max()
                    # 如果缓存够新, 直接返回
                    cache_end = pd.Timestamp(end_date)
                    if latest_cached.date() >= cache_end.date():
                        return self._clean(cached)
            except Exception:
                cached = None

        # ── 确定拉取范围 ──
        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - timedelta(days=days)

        # akshare 限制: 1分钟线只支持5天, 5分钟线支持更长
        # 格式: "YYYY-MM-DD HH:MM:SS"
        start_str = start_dt.strftime("%Y-%m-%d 09:30:00")
        end_str = end_dt.strftime("%Y-%m-%d 15:00:00")

        try:
            import akshare as ak
            import warnings
            warnings.filterwarnings('ignore')

            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=self.period,
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )

            if df is None or len(df) == 0:
                return self._clean(cached) if cached is not None else None

            df = self._clean(df)

            # ── 合并缓存 ──
            if cached is not None and len(cached) > 0:
                df = pd.concat([cached, df], ignore_index=True)
                df = df.drop_duplicates(subset=["时间"], keep="last")
                df = df.sort_values("时间").reset_index(drop=True)

            # ── 滚动截断: 只保留最近 cache_days 天 ──
            cutoff = pd.Timestamp.now().normalize() - timedelta(days=self.cache_days)
            df = df[df["时间"] >= cutoff]

            df.to_parquet(cache_path, index=False)
            return df

        except Exception as e:
            print(f"[MinuteFetcher] 拉取 {symbol} 失败: {e}")
            return self._clean(cached) if cached is not None else None

    def fetch_batch(self, symbols: List[str], days: int = None,
                    end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        批量拉取分钟线数据。

        Returns:
          {symbol: DataFrame}
        """
        result = {}
        for i, sym in enumerate(symbols):
            df = self.fetch(sym, days=days, end_date=end_date)
            if df is not None:
                result[sym] = df
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)  # 礼貌限速
        return result

    def fetch_positions(self, symbols: List[str], days: int = 10) -> Dict[str, pd.DataFrame]:
        """
        快速拉取持仓股票的近期分钟数据 (用于日内 mark-to-market)。
        只拉最近10天, 减少数据量。
        """
        return self.fetch_batch(symbols, days=days)

    # ════════════════════════════════════════
    #  执行算法
    # ════════════════════════════════════════

    def get_vwap(self, symbol: str, date_str: str) -> Optional[float]:
        """
        计算某只股票在指定日期的 VWAP (成交量加权均价)。

        Args:
          symbol: 股票代码
          date_str: "2026-08-03"

        Returns:
          VWAP 价格或 None
        """
        df = self.fetch(symbol, days=5, end_date=date_str)
        if df is None or len(df) == 0:
            return None

        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None

        # VWAP = Σ(price × volume) / Σ(volume)
        # 注意: akshare 的成交量单位是"手", 需要 ×100 转股
        # 但VWAP比率计算中单位抵消, 直接用原始值即可
        vol = day_bars["成交量"].astype(float)
        price = day_bars["收盘"].astype(float)  # 用收盘价近似

        total_vol = vol.sum()
        if total_vol == 0:
            return float(day_bars["收盘"].mean())

        vwap = (price * vol).sum() / total_vol
        return float(vwap)

    def get_twap(self, symbol: str, date_str: str, slices: int = 8) -> Optional[float]:
        """
        计算 TWAP (时间加权均价) — 将全天分为 N 个等时段取均价。

        Args:
          symbol: 股票代码
          date_str: "2026-08-03"
          slices: 拆分段数 (默认8, 约每30分钟一段)

        Returns:
          TWAP 价格或 None
        """
        df = self.fetch(symbol, days=5, end_date=date_str)
        if df is None or len(df) == 0:
            return None

        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None

        n = len(day_bars)
        if n < slices:
            return float(day_bars["收盘"].mean())

        # 等分
        chunk_size = n // slices
        twap_sum = 0.0
        count = 0
        for i in range(slices):
            chunk = day_bars.iloc[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) > 0:
                twap_sum += chunk["收盘"].mean()
                count += 1

        return float(twap_sum / count) if count > 0 else None

    def get_open_price(self, symbol: str, date_str: str) -> Optional[float]:
        """获取某日第一根K线的开盘价。"""
        df = self.fetch(symbol, days=5, end_date=date_str)
        if df is None or len(df) == 0:
            return None
        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None
        return float(day_bars["开盘"].iloc[0])

    def get_close_price(self, symbol: str, date_str: str) -> Optional[float]:
        """获取某日最后一根K线的收盘价。"""
        df = self.fetch(symbol, days=5, end_date=date_str)
        if df is None or len(df) == 0:
            return None
        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None
        return float(day_bars["收盘"].iloc[-1])

    def get_execution_price(self, symbol: str, date_str: str,
                            algo: str = "vwap") -> Optional[float]:
        """
        统一执行价格接口 — 根据算法返回模拟成交价。

        Args:
          symbol: 股票代码
          date_str: 执行日期
          algo: "open" / "close" / "vwap" / "twap"

        Returns:
          模拟成交价
        """
        if algo == "open":
            return self.get_open_price(symbol, date_str)
        elif algo == "close":
            return self.get_close_price(symbol, date_str)
        elif algo == "twap":
            return self.get_twap(symbol, date_str)
        else:  # vwap (default)
            return self.get_vwap(symbol, date_str)

    # ════════════════════════════════════════
    #  日内数据快照 (用于 mark-to-market)
    # ════════════════════════════════════════

    def get_intraday_close_prices(self, symbols: List[str],
                                  date_str: str) -> Dict[str, float]:
        """
        获取一批股票在指定日期的收盘价 (日末mark-to-market)。

        优先用分钟线收盘价, 回退到日线缓存。
        """
        result = {}
        for sym in symbols:
            px = self.get_close_price(sym, date_str)
            if px is not None:
                result[sym] = px
            else:
                # 回退: 从日线缓存读取
                from data_cache import load as load_daily
                df = load_daily(sym)
                if df is not None and len(df) > 0:
                    df["date"] = pd.to_datetime(df["date"])
                    mask = df["date"] == pd.Timestamp(date_str)
                    if mask.any():
                        result[sym] = float(df[mask]["close"].iloc[-1])
        return result

    # ════════════════════════════════════════
    #  工具方法
    # ════════════════════════════════════════

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化分钟数据的列名和类型。"""
        df = df.copy()
        if "时间" in df.columns:
            df["时间"] = pd.to_datetime(df["时间"])
        # 确保数值列
        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.sort_values("时间").reset_index(drop=True)

    def get_cache_status(self) -> dict:
        """查看分钟数据缓存状态。"""
        if not os.path.exists(MINUTE_CACHE_DIR):
            return {"has_data": False, "file_count": 0}

        files = [f for f in os.listdir(MINUTE_CACHE_DIR) if f.endswith(".parquet")]
        total_size = sum(
            os.path.getsize(os.path.join(MINUTE_CACHE_DIR, f)) for f in files
        )
        return {
            "has_data": True,
            "file_count": len(files),
            "total_size_mb": round(total_size / 1024 / 1024, 1),
            "period": self.period,
        }


# ════════════════════════════════════════
#  CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分钟数据拉取")
    parser.add_argument("--symbol", type=str, default="600519",
                       help="股票代码")
    parser.add_argument("--days", type=int, default=5,
                       help="拉取天数")
    parser.add_argument("--vwap", type=str, default=None,
                       help="计算指定日期的VWAP")
    parser.add_argument("--status", action="store_true",
                       help="查看缓存状态")
    args = parser.parse_args()

    mf = MinuteFetcher()

    if args.status:
        status = mf.get_cache_status()
        for k, v in status.items():
            print(f"  {k}: {v}")

    elif args.vwap:
        vwap = mf.get_vwap(args.symbol, args.vwap)
        twap = mf.get_twap(args.symbol, args.vwap)
        open_px = mf.get_open_price(args.symbol, args.vwap)
        close_px = mf.get_close_price(args.symbol, args.vwap)
        print(f"{args.symbol} {args.vwap}:")
        print(f"  Open:  {open_px}")
        print(f"  VWAP:  {vwap}")
        print(f"  TWAP:  {twap}")
        print(f"  Close: {close_px}")

    else:
        df = mf.fetch(args.symbol, days=args.days)
        if df is not None:
            print(f"{args.symbol} 最近{args.days}天, {len(df)}根{args.period}分钟K线")
            print(f"  日期范围: {df['时间'].min()} ~ {df['时间'].max()}")
            print(f"  列: {list(df.columns)}")
            print(f"\n最近5行:")
            print(df.tail())
        else:
            print(f"拉取 {args.symbol} 失败")
