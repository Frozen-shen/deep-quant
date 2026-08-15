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
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "minute")
DEFAULT_PERIOD = "5"        # 5分钟线
DEFAULT_CACHE_DAYS = 60     # 滚动保留天数


def _ensure_cache_dir():
    os.makedirs(MINUTE_CACHE_DIR, exist_ok=True)


class MinuteFetcher:
    """分钟级行情数据获取器。"""

    def __init__(self, period: str = DEFAULT_PERIOD,
                 cache_days: int = DEFAULT_CACHE_DAYS,
                 allow_network: bool = True):
        self.period = period
        self.cache_days = cache_days
        # 回测场景置 False: 本地无数据直接返回 None (由调用方回退 VWAP/开盘),
        # 不尝试网络拉取 (2022 前日期 akshare 拉不到, 只会失败刷屏)。
        self.allow_network = allow_network
        # 本地 5m 全历史缓存 (回测大量调用时避免重复读 parquet)
        self._local_cache: Dict[str, Optional[pd.DataFrame]] = {}
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

        ★ 优先使用本地全历史缓存 data_store/minute_5m/ (baostock, 2022起,
          5055只全覆盖), 列: day/open/high/low/close/volume/amount。
          该目录无该股/数据不足时才走 akshare 网络拉取 (滚动60天)。

        Args:
          symbol: 股票代码 (如 "600519")
          days: 拉取最近多少天 (None=用cache_days)
          end_date: 截止日期 YYYYMMDD (None=今天)

        Returns:
          DataFrame (统一列: 时间/开盘/收盘/最高/最低/成交量) 或 None
        """
        if days is None:
            days = self.cache_days
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # ── 本地全历史缓存 (baostock minute_5m, 2026-08-12 接入) ──
        if symbol in self._local_cache:
            return self._local_cache[symbol]
        local_path = os.path.join(BASE_DIR, "data_store", "minute_5m", f"{symbol}.parquet")
        if os.path.exists(local_path):
            try:
                ldf = pd.read_parquet(local_path)
                if len(ldf) > 0 and "day" in ldf.columns:
                    # 时间列优先用 datetime (含时分, 如 09:35); day 只有日期
                    if "datetime" in ldf.columns:
                        ldf["时间"] = pd.to_datetime(ldf["datetime"])
                    else:
                        ldf["时间"] = pd.to_datetime(ldf["day"])
                    ldf = ldf.rename(columns={"open": "开盘", "close": "收盘",
                                              "high": "最高", "low": "最低",
                                              "volume": "成交量"})
                    # volume 单位已统一为"股" (fix_minute_volume_units.py 已修复
                    # 2026-08-04~06 起的手单位; 历史为股) — 不做转换
                    if "amount" not in ldf.columns:
                        ldf["成交额"] = ldf["收盘"] * ldf["成交量"]
                    else:
                        ldf["成交额"] = ldf["amount"]
                    end_dt = pd.Timestamp(end_date)
                    ldf = ldf[ldf["时间"] <= end_dt + pd.Timedelta(days=1)]
                    if len(ldf) > 0:
                        out = self._clean(ldf[["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]])
                        self._local_cache[symbol] = out
                        return out
            except Exception:
                pass

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

        # ── 无网络模式 (回测): 本地全历史+滚动缓存都没有 → 直接返回 ──
        if not self.allow_network:
            return self._clean(cached) if cached is not None else None

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

    def get_pov_price(self, symbol: str, date_str: str,
                      order_qty: float, max_participation: float = 0.20,
                      lookback_days: int = 20) -> Optional[float]:
        """
        POV (Percent of Volume) 模拟成交价 — 跟随市场成交量节奏拆单。

        参与率自适应: ρ = min(max_participation, order_qty / 预测日成交量)
          - 大订单 (占日成交量比例高) → 高参与率, 尽快完成
          - 小订单 → 低参与率, 全天分散跟随市场节奏
          下限 0.001 (极小订单也给最小参与, 避免瞬间成交失真)

        执行: 遍历当日每个 5m 时段,
          下单量 = min(该时段实际成交量 × ρ, 剩余订单)
        成交价 = Σ(时段下单量 × 时段价) / 总下单量 (量加权, 模拟自然成交)

        Args:
          symbol: 股票代码
          date_str: 执行日期 "YYYY-MM-DD"
          order_qty: 订单数量 (股)
          max_participation: 参与率上限 (默认 20%)
          lookback_days: 预测日成交量用的历史窗口 (交易日)

        Returns:
          POV 模拟成交价 (加权平均) 或 None (无数据/无法成交)
        """
        if order_qty <= 0:
            return None
        df = self.fetch(symbol, days=lookback_days + 5, end_date=date_str)
        if df is None or len(df) == 0:
            return None

        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None

        # 预测日成交量: 执行日前 lookback_days 个交易日的平均成交量
        hist = df[df["时间"].dt.date < date_dt.date()]
        if len(hist) == 0:
            return None
        hist_daily = hist.groupby(hist["时间"].dt.date)["成交量"].sum()
        pred_vol = float(hist_daily.tail(lookback_days).mean()) if len(hist_daily) else 0.0
        if pred_vol <= 0:
            return None

        # 自适应参与率: 目标"一天内按市场节奏完成"
        rho = min(max_participation, order_qty / pred_vol)
        rho = max(rho, 0.001)

        # 逐时段执行 (POV): 量大的时段多买, 量小的时段少买
        remaining = float(order_qty)
        total_cost = 0.0
        filled = 0.0
        carry = 0.0  # 小数份额累积 (小订单 ρ×V < 1股时累积)
        for _, bar in day_bars.iterrows():
            if remaining <= 0:
                break
            bar_vol = float(bar["成交量"])
            q = min(bar_vol * rho + carry, remaining)
            carry = q - int(q)
            q = int(q)
            if q <= 0:
                continue
            price = float(bar["收盘"])  # 时段价 (5m收盘近似)
            total_cost += q * price
            filled += q
            remaining -= q

        if filled <= 0:
            return None
        # 未完成部分: 按当日最后价补单 (模拟强制完成)
        if remaining > 0:
            last_price = float(day_bars["收盘"].iloc[-1])
            total_cost += remaining * last_price
            filled += remaining
        return float(total_cost / filled)

    def get_pov_fills(self, symbol: str, date_str: str,
                      order_qty: float, max_participation: float = 0.20,
                      lookback_days: int = 20) -> Optional[dict]:
        """
        POV 逐时段成交明细 — 返回每个 5m 时段的成交 (时间/价格/数量)。

        与 get_pov_price 同一执行逻辑, 额外记录时段级 fill:
          fills: [{"time": "09:35", "price": ..., "qty": ...}, ...]

        Returns:
          {"price": 加权均价, "fills": [{time,price,qty}], "n_fills": n}
          或 None (无数据/无法成交)
        """
        if order_qty <= 0:
            return None
        df = self.fetch(symbol, days=lookback_days + 5, end_date=date_str)
        if df is None or len(df) == 0:
            return None

        date_dt = pd.Timestamp(date_str)
        day_bars = df[df["时间"].dt.date == date_dt.date()]
        if len(day_bars) == 0:
            return None

        # 预测日成交量: 执行日前 lookback_days 个交易日的平均成交量
        hist = df[df["时间"].dt.date < date_dt.date()]
        if len(hist) == 0:
            return None
        hist_daily = hist.groupby(hist["时间"].dt.date)["成交量"].sum()
        pred_vol = float(hist_daily.tail(lookback_days).mean()) if len(hist_daily) else 0.0
        if pred_vol <= 0:
            return None

        # 订单占比 < 0.1% (小订单): 对市场无冲击 → 市价单成交。
        # ★ 2026-08-15 修正: 旧假设"按全天VWAP成交"是事后价格(前视偏差);
        # 固定早盘市价也机械 (不可能每次都恰好早盘成交)。真实执行是不确定
        # 的 → 用固定种子在当日 5m bar 中随机选一个时点成交 (可复现,
        # 无前视偏差), 成交价 = 该 bar 收盘价, 残差滑点由 _apply_slippage 承担。
        if order_qty / pred_vol < 0.001:
            if len(day_bars) == 0:
                return None
            rng = random.Random(f"{symbol}-{date_str}")
            bar = day_bars.iloc[rng.randrange(len(day_bars))]
            early_px = float(bar["收盘"])
            bar_time = str(bar["时间"].to_pydatetime().strftime("%H:%M"))
            return {"price": float(early_px),
                    "fills": [{"time": f"市价@{bar_time}",
                               "price": round(early_px, 4), "qty": int(order_qty)}],
                    "n_fills": 1}

        # 自适应参与率: 目标"一天内按市场节奏完成"
        rho = min(max_participation, order_qty / pred_vol)
        rho = max(rho, 0.001)

        # 逐时段执行 (POV): 记录每个时段的成交时间/价格/数量
        remaining = float(order_qty)
        total_cost = 0.0
        filled = 0.0
        fills = []
        carry = 0.0  # 小数份额累积 (小订单 ρ×V < 1股时累积)
        for _, bar in day_bars.iterrows():
            if remaining <= 0:
                break
            bar_vol = float(bar["成交量"])
            q = min(bar_vol * rho + carry, remaining)
            carry = q - int(q)  # 累积不足1股的小数部分
            q = int(q)
            if q <= 0:
                continue
            price = float(bar["收盘"])  # 时段价 (5m收盘近似)
            total_cost += q * price
            filled += q
            remaining -= q
            fills.append({"time": str(bar["时间"].to_pydatetime().strftime("%H:%M")),
                          "price": round(price, 4), "qty": int(q)})

        if filled <= 0:
            return None
        # 未完成部分: 按当日最后价补单 (模拟强制完成)
        if remaining > 0:
            last_price = float(day_bars["收盘"].iloc[-1])
            total_cost += remaining * last_price
            filled += remaining
            fills.append({"time": "15:00", "price": round(last_price, 4),
                          "qty": int(remaining), "force_fill": True})
        return {"price": float(total_cost / filled), "fills": fills,
                "n_fills": len(fills)}

    def get_execution_price(self, symbol: str, date_str: str,
                            algo: str = "vwap",
                            order_qty: float = 0.0) -> Optional[float]:
        """
        统一执行价格接口 — 根据算法返回模拟成交价。

        Args:
          symbol: 股票代码
          date_str: 执行日期
          algo: "open" / "close" / "vwap" / "twap" / "pov"
          order_qty: 订单数量 (POV 需要, 其他算法可忽略)

        Returns:
          模拟成交价
        """
        if algo == "open":
            return self.get_open_price(symbol, date_str)
        elif algo == "close":
            return self.get_close_price(symbol, date_str)
        elif algo == "twap":
            return self.get_twap(symbol, date_str)
        elif algo == "pov":
            return self.get_pov_price(symbol, date_str, order_qty)
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
