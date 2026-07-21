"""
数据获取模块 — 统一支持 A 股 + 港股

A股: 腾讯证券 API (akshare stock_zh_a_hist_tx)
港股: 新浪财经 API (akshare stock_hk_daily)

用法:
    fetcher = DataFetcher()
    df_a  = fetcher.fetch("600519", market="a")        # A股茅台
    df_hk = fetcher.fetch("01810",  market="hk")        # 港股小米
    df    = fetcher.fetch("600519")                     # 自动检测:A股
    df    = fetcher.fetch("01810")                      # 自动检测:港股
"""

import pandas as pd
import akshare as ak
from typing import Optional
import requests as _r
import time as _time

# 禁用代理 (解决部分网络环境 requests 走系统代理的问题)
_r.Session.trust_env = False


# ============================================================================
#  市场配置
# ============================================================================

MARKET_CONFIG = {
    "a": {
        "name": "A股",
        "currency": "CNY",
        "lot_size": 100,
        "commission_default": 0.0003,
        "buy_commission":  0.0003,           # 万三
        "sell_commission": 0.0008,           # 万三 + 印花税0.05%
        "stamp_duty": 0.0005,
        "risk_free_rate": 0.02,
        "t_plus": 1,                        # T+1
        "trading_days_per_year": 252,
        "price_limit": 0.10,                # ±10%
    },
    "hk": {
        "name": "港股",
        "currency": "HKD",
        "lot_size": 200,
        # 逐项费用分解 (2023年8月起)
        "stamp_duty":        0.0013,    # 印花税 0.13% (双边)
        "trading_fee":       0.0000565, # 交易费 (SFC)
        "transaction_levy":  0.000027,  # 交易征费 (HKEX)
        "brokerage":         0.0003,    # 券商佣金 (假设最低)
        # 综合 (由上面加总)
        "buy_commission":  0.0013835,  # ~0.138% (买入总成本)
        "sell_commission": 0.0013835,  # ~0.138% (卖出总成本)
        # 兼容旧代码
        "commission_default": 0.0018,  # 保留旧字段
        "risk_free_rate": 0.035,       # HIBOR ~3.5%
        "t_plus": 0,                   # T+0
        "trading_days_per_year": 247,
        "price_limit": None,
    },
}


class DataFetcher:
    """统一 A 股 + 港股数据获取器"""

    # ================================================================
    #  公开接口
    # ================================================================
    @staticmethod
    def fetch(
        symbol: str = "600519",
        start_date: str = "20180101",
        end_date: str = "20260710",
        adjust: str = "qfq",
        market: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取日线行情 — 自动识别 A 股 / 港股。

        参数
        ----
        symbol : str
            股票代码。A股: "600519"; 港股: "01810"
        start_date, end_date : str
            YYYYMMDD 格式
        adjust : str
            复权方式。A股: "qfq"/"hfq"/""; 港股: "qfq"/"hfq"/""
        market : str or None
            "a" / "hk" / None(自动检测)

        返回
        ----
        pd.DataFrame [date, open, high, low, close, volume, amount]
        """
        market = market or DataFetcher._detect_market(symbol)
        config = MARKET_CONFIG[market]

        if market == "a":
            df = DataFetcher._fetch_a_share(symbol, start_date, end_date, adjust)
        elif market == "hk":
            df = DataFetcher._fetch_hk_stock(symbol, start_date, end_date, adjust)
        else:
            raise ValueError(f"不支持的市场: {market}")

        df.attrs["market"] = market
        df.attrs["currency"] = config["currency"]
        df.attrs["symbol"] = symbol

        return df

    @staticmethod
    def get_config(market: str) -> dict:
        """获取市场配置。"""
        return MARKET_CONFIG.get(market, MARKET_CONFIG["a"]).copy()

    @staticmethod
    def _detect_market(symbol: str) -> str:
        """
        自动检测市场。

        规则:
        - 5位数字 → 港股 (如 01810, 00700, 09988)
        - 6位数字 → A股  (如 600519, 000001, 300750)
        - 以 sh/sz 开头 → A股
        """
        if symbol.startswith(("sh", "sz")):
            return "a"
        if symbol.isdigit() and len(symbol) == 5:
            return "hk"
        return "a"

    # ================================================================
    #  A 股 — 新浪 API (有 volume + turnover)
    # ================================================================
    @staticmethod
    def _fetch_a_share(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        """A股日线 (新浪数据源 — 有volume列)"""
        # 新浪格式: sh600519 / sz000001
        if not symbol.startswith(("sh", "sz")):
            prefix = "sh" if symbol.startswith(("6", "68")) else "sz"
            sina_symbol = f"{prefix}{symbol}"
        else:
            sina_symbol = symbol

        print(f"[DataFetcher:A] 获取 {symbol} ({sina_symbol}) ...")

        raw = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if raw.empty:
            raise ValueError(f"A股 {symbol} 无数据")

        # 新浪返回的已是英文列名: date, open, high, low, close, volume, amount, turnover
        return DataFetcher._clean(raw, start_date, end_date, symbol)

    # ================================================================
    #  港股 — 新浪 API
    # ================================================================
    @staticmethod
    def _fetch_hk_stock(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        """港股日线 (新浪数据源) — 带重试"""
        print(f"[DataFetcher:HK] 获取 {symbol} ...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = ak.stock_hk_daily(symbol=symbol, adjust=adjust)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"  重试 {attempt+1}/{max_retries} (等待{wait}s): {e}")
                    _time.sleep(wait)
                else:
                    raise

        if raw.empty:
            raise ValueError(f"港股 {symbol} 无数据")

        df = pd.DataFrame(raw)
        # stock_hk_daily 返回: date, open, high, low, close, volume, amount
        # 列名已是英文，不需要重命名
        keep_cols = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        df = df[[c for c in keep_cols if c in df.columns]]

        return DataFetcher._clean(df, start_date, end_date, symbol)

    # ================================================================
    #  清洗 (通用)
    # ================================================================
    @staticmethod
    def _clean(df: pd.DataFrame, start_date: str, end_date: str,
               symbol: str) -> pd.DataFrame:
        """统一的清洗逻辑：日期解析 → 过滤 → 排序 → 填充"""
        # 日期
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # 日期过滤
        if len(start_date) == 8:
            start_dt = pd.Timestamp(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
        else:
            start_dt = pd.Timestamp(start_date)
        if len(end_date) == 8:
            end_dt = pd.Timestamp(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}")
        else:
            end_dt = pd.Timestamp(end_date)

        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.sort_values("date").reset_index(drop=True)

        # 数值列
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # OHLC 前向填充（停牌日）
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        # volume/amount 填 0
        for col in ["volume", "amount"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        print(f"[DataFetcher] {symbol}: {len(df)} 条 "
              f"({df['date'].min().date()} ~ {df['date'].max().date()})")
        return df
