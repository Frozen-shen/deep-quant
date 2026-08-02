"""
Stock universe management module.

Provides the Universe class for managing stock pools with support for:
  - Static universe (loaded from cached file)
  - Dynamic point-in-time universe (survivorship-bias-free)
  - Filtering: remove ST stocks, remove < 60 trading days, remove limit days
  - Supported indices: csi300, csi500, csi1000, zz_all

Usage:
    from quant.data.universe import Universe

    uni = Universe(index="csi1000")
    symbols = uni.get_symbols()                   # Current constituents
    symbols = uni.get_symbols(as_of_date="2023-06-01")  # Point-in-time
    tradeable = uni.filter_tradeable(symbols, "2023-06-15")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

# Index code mapping
INDEX_MAP = {
    "csi300": "000300",
    "csi500": "000905",
    "csi1000": "000852",
    "zz_all": "000985",  # CSI All Share
    "000300": "000300",
    "000905": "000905",
    "000852": "000852",
    "000985": "000985",
}

DateLike = Union[str, date, datetime, pd.Timestamp]


class Universe:
    """
    Stock universe manager with point-in-time support.

    Parameters
    ----------
    index : str
        Universe identifier. One of "csi300", "csi500", "csi1000",
        "zz_all", or a raw index code like "000852".
    cache_dir : str or Path, optional
        Directory for universe snapshot cache files.
    min_trading_days : int
        Minimum number of trading days a stock must have to be
        included. Default 60.
    exclude_st : bool
        Whether to exclude ST/*ST stocks. Default True.
    """

    def __init__(
        self,
        index: str = "csi1000",
        cache_dir: Optional[Union[str, Path]] = None,
        min_trading_days: int = 60,
        exclude_st: bool = True,
    ):
        self._index_name = index
        self._index_code = INDEX_MAP.get(index, index)
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._min_trading_days = min_trading_days
        self._exclude_st = exclude_st

        # Snapshots: {"YYYY-MM": [symbols]}
        self._snapshots: Dict[str, List[str]] = {}
        self._loaded = False

        # Try loading from cache
        self._load_snapshots()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symbols(self, as_of_date: Optional[DateLike] = None) -> List[str]:
        """
        Get the list of symbols in the universe.

        Parameters
        ----------
        as_of_date : date-like, optional
            If provided, returns the point-in-time universe as of that
            date (survivorship-bias-free). If None, returns the most
            recent snapshot.

        Returns
        -------
        List[str]
            Sorted list of 6-digit stock codes.
        """
        if as_of_date is None:
            # Return the most recent snapshot
            if self._snapshots:
                latest_key = max(self._snapshots.keys())
                return sorted(self._snapshots[latest_key])
            # No snapshots - fetch current from network
            return self._fetch_current()

        # Point-in-time lookup
        ts = pd.Timestamp(as_of_date)
        month_key = ts.strftime("%Y-%m")

        if month_key in self._snapshots:
            return sorted(self._snapshots[month_key])

        # Fall back to nearest earlier snapshot
        if self._snapshots:
            sorted_keys = sorted(self._snapshots.keys())
            for key in reversed(sorted_keys):
                if key <= month_key:
                    return sorted(self._snapshots[key])
            # All snapshots are after the requested date - use earliest
            return sorted(self._snapshots[sorted_keys[0]])

        # No snapshots at all - fetch current (best effort)
        logger.warning(
            "No snapshots available for %s. Fetching current constituents.",
            self._index_name,
        )
        return self._fetch_current()

    def filter_tradeable(
        self,
        symbols: List[str],
        dt: DateLike,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[str]:
        """
        Filter symbols to those that are tradeable on a given date.

        Applies the following filters:
          1. Remove ST/*ST stocks (by name if available)
          2. Remove stocks with fewer than min_trading_days of history
          3. Remove stocks at limit-up or limit-down on the given date

        Parameters
        ----------
        symbols : List[str]
            Candidate symbols to filter.
        dt : date-like
            The trading date to check.
        data : Dict[str, pd.DataFrame], optional
            Pre-loaded data {symbol: DataFrame} for checking trading
            history and limit status. If not provided, only ST filter
            is applied (name-based).

        Returns
        -------
        List[str]
            Filtered list of tradeable symbols.
        """
        ts = pd.Timestamp(dt)
        result = list(symbols)

        # Filter 1: Remove ST stocks
        if self._exclude_st:
            result = self._filter_st(result)

        # Filter 2 & 3: Require data
        if data is not None:
            result = self._filter_insufficient_history(result, ts, data)
            result = self._filter_limit_days(result, ts, data)

        return result

    def build_snapshots(
        self,
        start_date: str = "2018-01-01",
        end_date: str = "2026-07-01",
    ) -> int:
        """
        Build monthly universe snapshots from akshare.

        Note: akshare's index_stock_cons_csindex() returns only current
        constituents. For true point-in-time, a premium data source
        (e.g., tushare index_weight) is needed. This method uses current
        constituents as a proxy for all months, which is a known
        simplification.

        Parameters
        ----------
        start_date, end_date : str
            Date range for snapshot generation (YYYY-MM-DD format).

        Returns
        -------
        int
            Number of symbols in the universe.
        """
        symbols = self._fetch_current()
        if not symbols:
            logger.error("Failed to fetch constituents for %s", self._index_name)
            return 0

        # Generate monthly snapshots
        months = pd.date_range(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            freq="MS",
        )

        for month_ts in months:
            key = month_ts.strftime("%Y-%m")
            self._snapshots[key] = symbols[:]

        self._save_snapshots()
        logger.info(
            "Built %d monthly snapshots for %s (%d symbols)",
            len(months), self._index_name, len(symbols),
        )
        return len(symbols)

    def load_from_file(self, path: Union[str, Path]) -> bool:
        """
        Load universe from a JSON file.

        Expected format: {"YYYY-MM": ["600519", "000858", ...], ...}

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.

        Returns
        -------
        bool
            True if loading succeeded.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Universe file not found: %s", path)
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._snapshots = data
            self._loaded = True
            logger.info(
                "Loaded universe from %s: %d months",
                path, len(self._snapshots),
            )
            return True
        except Exception as e:
            logger.error("Failed to load universe from %s: %s", path, e)
            return False

    @property
    def all_symbols(self) -> List[str]:
        """All unique symbols that ever appeared in any snapshot."""
        all_syms: set = set()
        for syms in self._snapshots.values():
            all_syms.update(syms)
        return sorted(all_syms)

    @property
    def index_code(self) -> str:
        """The resolved index code."""
        return self._index_code

    @property
    def snapshot_count(self) -> int:
        """Number of monthly snapshots loaded."""
        return len(self._snapshots)

    # ------------------------------------------------------------------
    # Internal: fetching
    # ------------------------------------------------------------------

    def _fetch_current(self) -> List[str]:
        """Fetch current index constituents from akshare."""
        import akshare as ak

        logger.info("Fetching constituents for index %s...", self._index_code)
        try:
            df = ak.index_stock_cons_csindex(symbol=self._index_code)
            if df is None or df.empty:
                logger.warning("Empty result for index %s", self._index_code)
                return []

            # Extract symbol column
            symbols = []
            for col in ["成分券代码", "品种代码", "stock_code", "代码", "constituent_code"]:
                if col in df.columns:
                    symbols = df[col].astype(str).tolist()
                    break
            if not symbols:
                symbols = df.iloc[:, 0].astype(str).tolist()

            # Clean
            clean = []
            for s in symbols:
                s = s.strip()
                for prefix in ("sh", "sz", "bj"):
                    if s.startswith(prefix):
                        s = s[len(prefix):]
                        break
                if s.isdigit() and len(s) == 6:
                    clean.append(s)

            logger.info("Fetched %d constituents for %s", len(clean), self._index_code)
            return sorted(clean)

        except Exception as e:
            logger.error("Failed to fetch index constituents: %s", e)
            return []

    # ------------------------------------------------------------------
    # Internal: filtering
    # ------------------------------------------------------------------

    def _filter_st(self, symbols: List[str]) -> List[str]:
        """
        Remove ST/*ST stocks.

        Uses akshare's stock info to identify ST stocks. If the lookup
        fails, returns the input unchanged (fail-open).
        """
        try:
            import akshare as ak

            # Get stock names for ST detection
            info_df = ak.stock_info_a_code_name()
            if info_df is None or info_df.empty:
                return symbols

            # Build ST set
            name_col = None
            for col in ["name", "股票简称", "证券简称"]:
                if col in info_df.columns:
                    name_col = col
                    break
            code_col = None
            for col in ["code", "股票代码", "证券代码", "A股代码"]:
                if col in info_df.columns:
                    code_col = col
                    break

            if name_col is None or code_col is None:
                return symbols

            st_codes = set()
            for _, row in info_df.iterrows():
                name = str(row[name_col])
                if "ST" in name.upper():
                    st_codes.add(str(row[code_col]).strip())

            if not st_codes:
                return symbols

            filtered = [s for s in symbols if s not in st_codes]
            removed = len(symbols) - len(filtered)
            if removed > 0:
                logger.debug("Filtered %d ST stocks", removed)
            return filtered

        except Exception as e:
            logger.warning("ST filter failed (fail-open): %s", e)
            return symbols

    def _filter_insufficient_history(
        self,
        symbols: List[str],
        dt: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Remove stocks with fewer than min_trading_days before dt."""
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            # Count trading days up to dt
            mask = df["date"] <= dt
            count = mask.sum()
            if count >= self._min_trading_days:
                result.append(sym)
        return result

    def _filter_limit_days(
        self,
        symbols: List[str],
        dt: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """
        Remove stocks at limit-up or limit-down on the given date.

        A-share limit: +/-10% (normal), +/-20% (ChiNext/STAR).
        A stock is considered limit-locked if:
          - close == high == low (one-word board, cannot trade)
          - OR pct_change >= 9.8% (limit up, likely cannot buy)
          - OR pct_change <= -9.8% (limit down, likely cannot sell)
        """
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue

            # Find the row for dt
            day_data = df[df["date"] == dt]
            if day_data.empty:
                # No data for this date (suspended) - exclude
                continue

            row = day_data.iloc[0]

            # One-word board: open == high == low == close
            if row["open"] == row["high"] == row["low"] == row["close"]:
                continue

            # Check pct_change if available, else compute
            if "pct_change" in df.columns:
                pct = row.get("pct_change", 0)
            else:
                # Compute from previous close
                idx = df.index[df["date"] == dt]
                if len(idx) > 0 and idx[0] > 0:
                    prev_close = df.iloc[idx[0] - 1]["close"]
                    if prev_close > 0:
                        pct = (row["close"] - prev_close) / prev_close * 100
                    else:
                        pct = 0
                else:
                    pct = 0

            # Determine limit threshold based on board
            limit_pct = self._get_limit_pct(sym)
            if abs(pct) >= limit_pct - 0.2:  # 0.2% tolerance
                continue

            result.append(sym)

        return result

    @staticmethod
    def _get_limit_pct(symbol: str) -> float:
        """
        Get the price limit percentage for a symbol.

        ChiNext (300xxx) and STAR (688xxx): 20%
        Others: 10%
        """
        if symbol.startswith(("300", "688")):
            return 20.0
        return 10.0

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _snapshot_path(self) -> Path:
        """Path to the snapshot cache file."""
        return self._cache_dir / f"universe_{self._index_code}.json"

    def _load_snapshots(self) -> None:
        """Attempt to load snapshots from cache."""
        path = self._snapshot_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._snapshots = json.load(f)
                self._loaded = True
                logger.debug(
                    "Loaded %d universe snapshots from cache",
                    len(self._snapshots),
                )
            except Exception as e:
                logger.warning("Failed to load universe cache: %s", e)
                self._snapshots = {}

    def _save_snapshots(self) -> None:
        """Persist snapshots to cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._snapshots, f, ensure_ascii=False)
        logger.debug("Saved universe snapshots to %s", path)
