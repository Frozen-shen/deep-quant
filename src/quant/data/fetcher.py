"""
Unified data fetcher module.

Merges the legacy data_fetcher.py and data_cache.py into a single,
production-grade class with:
  - Automatic local parquet caching (check cache before network)
  - Rate limiting (max 3 requests/second to akshare)
  - Retry with exponential backoff (3 attempts)
  - Progress reporting for batch downloads
  - Support for A-share (Tencent/Sina via akshare) and HK stocks

Usage:
    from quant.data.fetcher import DataFetcher

    fetcher = DataFetcher()
    df = fetcher.fetch_daily("600519", "20200101", "20260101")
    symbols = fetcher.fetch_universe("000852")  # CSI1000 components
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory: project_root/data_cache/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data_cache"

# Standard output columns after cleaning
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]

# akshare column name mapping (Chinese -> English)
_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "振幅": "amplitude",
}


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Ensures no more than `max_requests` are made within any rolling
    1-second window.
    """

    def __init__(self, max_requests: int = 3):
        self._max_requests = max_requests
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            # Remove timestamps older than 1 second
            self._timestamps = [t for t in self._timestamps if now - t < 1.0]
            if len(self._timestamps) >= self._max_requests:
                sleep_time = 1.0 - (now - self._timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Clean up again after sleeping
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 1.0]
            self._timestamps.append(time.monotonic())


class DataFetcher:
    """
    Unified data fetcher with caching, rate limiting, and retry.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for local parquet cache files. Defaults to
        ``<project_root>/data_cache/``.
    max_requests_per_sec : int
        Rate limit for akshare API calls. Default 3.
    max_retries : int
        Number of retry attempts for failed network calls. Default 3.
    use_cache : bool
        Whether to check/use local cache. Default True.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_requests_per_sec: int = 3,
        max_retries: int = 3,
        use_cache: bool = True,
    ):
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._rate_limiter = RateLimiter(max_requests_per_sec)
        self._max_retries = max_retries
        self._use_cache = use_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str = "qfq",
        market: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a single symbol.

        Automatically checks local cache first. If not cached or the
        cache does not cover the requested date range, fetches from
        network and updates the cache.

        Parameters
        ----------
        symbol : str
            Stock code. A-share: "600519", "000001". HK: "01810", "00700".
        start : str
            Start date in YYYYMMDD format.
        end : str
            End date in YYYYMMDD format.
        adjust : str
            Price adjustment: "qfq" (forward), "hfq" (backward), "" (none).
        market : str or None
            "a" for A-share, "hk" for HK, None for auto-detect.

        Returns
        -------
        pd.DataFrame
            Columns: [date, open, high, low, close, volume, amount, turnover]
            with DatetimeIndex on 'date'.
        """
        market = market or self._detect_market(symbol)

        # Check cache
        if self._use_cache:
            cached = self._load_cache(symbol)
            if cached is not None:
                cached_df = self._filter_by_date(cached, start, end)
                if self._cache_covers(cached_df, start, end):
                    logger.debug("Cache hit for %s", symbol)
                    return cached_df

        # Fetch from network
        df = self._fetch_with_retry(symbol, start, end, adjust, market)

        # Update cache (merge with existing)
        if self._use_cache and df is not None and not df.empty:
            self._update_cache(symbol, df)

        return df

    def fetch_universe(self, index_code: str = "000852") -> List[str]:
        """
        Fetch index constituent stocks.

        Parameters
        ----------
        index_code : str
            Index code: "000300" (CSI300), "000905" (CSI500),
            "000852" (CSI1000).

        Returns
        -------
        List[str]
            List of 6-digit stock codes.
        """
        import akshare as ak

        self._rate_limiter.acquire()
        df = self._retry_call(
            lambda: ak.index_stock_cons_csindex(symbol=index_code)
        )

        if df is None or df.empty:
            logger.warning("No constituents returned for index %s", index_code)
            return []

        # Try known column names
        symbols = []
        for col in ["成分券代码", "品种代码", "stock_code", "代码", "constituent_code"]:
            if col in df.columns:
                symbols = df[col].astype(str).tolist()
                break
        if not symbols:
            symbols = df.iloc[:, 0].astype(str).tolist()

        # Clean: strip prefixes, keep only 6-digit codes
        clean = []
        for s in symbols:
            s = s.strip()
            for prefix in ("sh", "sz", "bj"):
                if s.startswith(prefix):
                    s = s[len(prefix):]
                    break
            if s.isdigit() and len(s) == 6:
                clean.append(s)

        logger.info("Index %s: %d constituents", index_code, len(clean))
        return clean

    def fetch_batch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        adjust: str = "qfq",
        market: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for multiple symbols with progress reporting.

        Parameters
        ----------
        symbols : List[str]
            Stock codes to fetch.
        start, end : str
            Date range in YYYYMMDD format.
        adjust : str
            Price adjustment mode.
        market : str or None
            Market hint. None = auto-detect per symbol.
        progress_callback : callable, optional
            Called as callback(current_index, total, symbol) after each
            symbol is processed.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of symbol -> DataFrame. Failed symbols are omitted.
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)
        failed: List[str] = []

        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch_daily(symbol, start, end, adjust, market)
                if df is not None and not df.empty:
                    results[symbol] = df
                else:
                    failed.append(symbol)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", symbol, e)
                failed.append(symbol)

            if progress_callback:
                progress_callback(i + 1, total, symbol)
            elif (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(
                    "Batch progress: %d/%d (%.0f%%)",
                    i + 1, total, (i + 1) / total * 100,
                )

        if failed:
            logger.warning(
                "Batch complete: %d succeeded, %d failed: %s",
                len(results), len(failed),
                failed[:10] if len(failed) > 10 else failed,
            )
        else:
            logger.info("Batch complete: all %d symbols fetched.", total)

        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str) -> Path:
        """Get the parquet cache file path for a symbol."""
        return self._cache_dir / f"{symbol}.parquet"

    def _load_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol, or None if not cached."""
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning("Corrupt cache for %s: %s", symbol, e)
            return None

    def _update_cache(self, symbol: str, new_df: pd.DataFrame) -> None:
        """Merge new data into existing cache and save."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(symbol)

        existing = self._load_cache(symbol)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        else:
            combined = new_df.copy()

        combined.to_parquet(path, index=False, engine="pyarrow")

    def _cache_covers(self, df: pd.DataFrame, start: str, end: str) -> bool:
        """
        Check if cached data adequately covers the requested range.

        Returns True if the cache has data within 3 trading days of
        both boundaries (to account for weekends/holidays at edges).
        """
        if df is None or df.empty:
            return False

        start_ts = pd.Timestamp(f"{start[:4]}-{start[4:6]}-{start[6:8]}")
        end_ts = pd.Timestamp(f"{end[:4]}-{end[4:6]}-{end[6:8]}")

        data_start = df["date"].min()
        data_end = df["date"].max()

        # Allow 5 calendar days tolerance at start (weekends + holidays)
        start_ok = data_start <= start_ts + pd.Timedelta(days=5)
        # End must be within 3 days (if end is recent, data might legitimately stop)
        end_ok = data_end >= end_ts - pd.Timedelta(days=5)

        return start_ok and end_ok

    # ------------------------------------------------------------------
    # Network fetching
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str,
        market: str,
    ) -> pd.DataFrame:
        """Fetch from akshare with rate limiting and retry."""
        if market == "hk":
            raw = self._retry_call(
                lambda: self._raw_fetch_hk(symbol, adjust)
            )
        else:
            raw = self._retry_call(
                lambda: self._raw_fetch_a(symbol, start, end, adjust)
            )

        if raw is None or raw.empty:
            raise ValueError(f"No data returned for {symbol} ({market})")

        df = self._clean_dataframe(raw, start, end, symbol)
        return df

    def _raw_fetch_a(
        self, symbol: str, start: str, end: str, adjust: str
    ) -> pd.DataFrame:
        """Raw A-share fetch via akshare."""
        import akshare as ak

        self._rate_limiter.acquire()
        logger.debug("Fetching A-share %s (%s ~ %s)", symbol, start, end)

        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        return raw

    def _raw_fetch_hk(self, symbol: str, adjust: str) -> pd.DataFrame:
        """Raw HK stock fetch via akshare."""
        import akshare as ak

        self._rate_limiter.acquire()
        logger.debug("Fetching HK stock %s", symbol)

        raw = ak.stock_hk_daily(symbol=symbol, adjust=adjust)
        return raw

    def _retry_call(self, fn, description: str = "") -> pd.DataFrame:
        """
        Execute a callable with exponential backoff retry.

        Parameters
        ----------
        fn : callable
            Zero-argument callable that returns a DataFrame.
        description : str
            Human-readable description for logging.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        Exception
            Re-raises the last exception after all retries are exhausted.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "Attempt %d/%d failed%s: %s. Retrying in %ds...",
                        attempt, self._max_retries,
                        f" ({description})" if description else "",
                        e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "All %d attempts failed%s: %s",
                        self._max_retries,
                        f" ({description})" if description else "",
                        e,
                    )
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def _clean_dataframe(
        self, raw: pd.DataFrame, start: str, end: str, symbol: str
    ) -> pd.DataFrame:
        """
        Normalize raw akshare output to standard format.

        Handles inconsistent column names (Chinese/English), ensures
        proper dtypes, sorts by date, and fills missing values.
        """
        df = raw.copy()

        # Rename Chinese columns to English
        df = df.rename(columns=_COLUMN_MAP)

        # Ensure 'date' column exists
        if "date" not in df.columns:
            # Some akshare functions use the index as date
            if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "date"})
            else:
                raise ValueError(
                    f"Cannot identify date column for {symbol}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Filter by date range
        start_ts = pd.Timestamp(f"{start[:4]}-{start[4:6]}-{start[6:8]}")
        end_ts = pd.Timestamp(f"{end[:4]}-{end[4:6]}-{end[6:8]}")
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        # Sort
        df = df.sort_values("date").reset_index(drop=True)

        # Ensure numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Forward-fill OHLC (handles suspension days within the range)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Fill volume/amount with 0
        for col in ["volume", "amount"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Keep only standard columns that exist
        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[keep]

        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_market(symbol: str) -> str:
        """
        Auto-detect market from symbol format.

        Rules:
          - 5-digit numeric -> HK (e.g., "01810", "00700")
          - 6-digit numeric -> A-share (e.g., "600519", "000001")
          - Starts with "sh"/"sz" -> A-share
        """
        if symbol.startswith(("sh", "sz")):
            return "a"
        if symbol.isdigit() and len(symbol) == 5:
            return "hk"
        return "a"

    @staticmethod
    def _filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Filter a DataFrame to [start, end] date range."""
        if df is None or df.empty:
            return df
        start_ts = pd.Timestamp(f"{start[:4]}-{start[4:6]}-{start[6:8]}")
        end_ts = pd.Timestamp(f"{end[:4]}-{end[4:6]}-{end[6:8]}")
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        return df[mask].reset_index(drop=True)

    def get_cached_symbols(self) -> List[str]:
        """Return list of symbols that have local cache files."""
        if not self._cache_dir.exists():
            return []
        return sorted(
            f.stem for f in self._cache_dir.glob("*.parquet")
        )
