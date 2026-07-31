"""
Unified parquet storage with DataPanel.

Provides:
  - DataPanel: the central data structure for all downstream modules.
    Stores per-symbol DataFrames and provides efficient cross-sectional
    and time-series access patterns.
  - DataStore: persistent storage layer managing parquet files on disk
    with metadata tracking and incremental update support.

Usage:
    from quant.data.store import DataStore, DataPanel

    store = DataStore()
    panel = store.build_panel(["600519", "000858"], "20200101", "20260101")

    # Access patterns:
    df = panel.get("600519", "2023-01-01", "2023-06-30")   # Single symbol
    cs = panel.get_cross_section("2023-06-15", "close")     # All symbols, one date
    mat = panel.get_panel("close", "2023-01-01", "2023-06-30")  # Date x Symbol matrix
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORE_DIR = _PROJECT_ROOT / "data_store"


class DataPanel:
    """
    Unified market data panel.

    Internal storage: Dict[str, pd.DataFrame] where each DataFrame has:
      - index: RangeIndex (reset)
      - 'date' column: DatetimeIndex values (trading dates)
      - columns: [date, open, high, low, close, volume, amount, turnover, ...]
      - One DataFrame per symbol

    Provides efficient access patterns for quantitative research:
      - get(symbol, start, end) -> DataFrame
      - get_cross_section(date, field) -> Series (all symbols for one date)
      - get_panel(field, start, end) -> DataFrame (date x symbol matrix)
      - symbols property -> List[str]
      - date_range property -> tuple of (min_date, max_date)
    """

    def __init__(self, data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize a DataPanel.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame], optional
            Pre-built mapping of symbol -> DataFrame. Each DataFrame must
            have a 'date' column with datetime values.
        """
        self._data: Dict[str, pd.DataFrame] = {}
        if data:
            for symbol, df in data.items():
                self._data[symbol] = self._validate_df(df, symbol)

    # ------------------------------------------------------------------
    # Access methods
    # ------------------------------------------------------------------

    def get(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get data for a single symbol, optionally filtered by date range.

        Parameters
        ----------
        symbol : str
            Stock code.
        start : str, optional
            Start date (YYYY-MM-DD or YYYYMMDD). Inclusive.
        end : str, optional
            End date (YYYY-MM-DD or YYYYMMDD). Inclusive.

        Returns
        -------
        pd.DataFrame
            Subset of the symbol's data. Empty DataFrame if symbol not found.
        """
        df = self._data.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        if start is not None:
            start_ts = pd.Timestamp(start)
            df = df[df["date"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            df = df[df["date"] <= end_ts]

        return df.reset_index(drop=True)

    def get_cross_section(
        self,
        dt: Union[str, pd.Timestamp],
        field: str = "close",
    ) -> pd.Series:
        """
        Get a cross-section: one field for all symbols on a single date.

        Parameters
        ----------
        dt : str or pd.Timestamp
            The target date.
        field : str
            Column name to extract (e.g., "close", "volume", "turnover").

        Returns
        -------
        pd.Series
            Index = symbol, values = field value. Symbols without data
            on that date are excluded.
        """
        ts = pd.Timestamp(dt)
        values = {}

        for symbol, df in self._data.items():
            if field not in df.columns:
                continue
            row = df[df["date"] == ts]
            if not row.empty:
                values[symbol] = row.iloc[0][field]

        return pd.Series(values, name=field)

    def get_panel(
        self,
        field: str = "close",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get a date x symbol matrix for a single field.

        Parameters
        ----------
        field : str
            Column name (e.g., "close", "volume").
        start : str, optional
            Start date filter.
        end : str, optional
            End date filter.

        Returns
        -------
        pd.DataFrame
            Index = DatetimeIndex (dates), Columns = symbols.
            Missing values are NaN.
        """
        frames = {}
        for symbol, df in self._data.items():
            if field not in df.columns:
                continue
            subset = df[["date", field]].copy()
            if start is not None:
                subset = subset[subset["date"] >= pd.Timestamp(start)]
            if end is not None:
                subset = subset[subset["date"] <= pd.Timestamp(end)]
            if not subset.empty:
                series = subset.set_index("date")[field]
                series.name = symbol
                frames[symbol] = series

        if not frames:
            return pd.DataFrame()

        panel = pd.DataFrame(frames)
        panel.index.name = "date"
        panel = panel.sort_index()
        return panel

    def get_returns(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute daily simple returns as a date x symbol matrix.

        Returns
        -------
        pd.DataFrame
            Index = dates, Columns = symbols, values = daily returns.
        """
        prices = self.get_panel("close", start, end)
        if prices.empty:
            return prices
        return prices.pct_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> List[str]:
        """List of all symbols in the panel."""
        return sorted(self._data.keys())

    @property
    def date_range(self) -> tuple:
        """
        Global date range across all symbols.

        Returns
        -------
        tuple of (pd.Timestamp, pd.Timestamp)
            (earliest_date, latest_date). Returns (None, None) if empty.
        """
        if not self._data:
            return (None, None)

        min_dates = []
        max_dates = []
        for df in self._data.values():
            if not df.empty:
                min_dates.append(df["date"].min())
                max_dates.append(df["date"].max())

        if not min_dates:
            return (None, None)
        return (min(min_dates), max(max_dates))

    @property
    def shape(self) -> tuple:
        """Approximate shape: (total_rows, n_symbols)."""
        total_rows = sum(len(df) for df in self._data.values())
        return (total_rows, len(self._data))

    def __len__(self) -> int:
        """Number of symbols."""
        return len(self._data)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data

    def __repr__(self) -> str:
        dr = self.date_range
        if dr[0] is not None:
            range_str = f"{dr[0].date()} ~ {dr[1].date()}"
        else:
            range_str = "empty"
        return (
            f"DataPanel(symbols={len(self._data)}, "
            f"range={range_str}, "
            f"total_rows={self.shape[0]})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ensure DataFrame has required structure."""
        if df.empty:
            return df
        if "date" not in df.columns:
            raise ValueError(f"DataFrame for {symbol} missing 'date' column")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df


class DataStore:
    """
    Persistent storage for market data.

    Directory layout:
      data_store/
        000001.SZ.parquet
        600519.SH.parquet
        ...
        _meta.json

    Metadata (_meta.json):
      {
        "symbols": ["000001.SZ", "600519.SH", ...],
        "date_range": {"start": "2018-01-02", "end": "2026-07-10"},
        "last_update": "2026-07-31T10:30:00",
        "row_counts": {"000001.SZ": 2100, ...}
      }

    Parameters
    ----------
    store_dir : str or Path, optional
        Storage directory. Defaults to ``<project_root>/data_store/``.
    fetcher : DataFetcher, optional
        A DataFetcher instance for network access. If None, one is
        created with default settings.
    """

    def __init__(
        self,
        store_dir: Optional[Union[str, Path]] = None,
        fetcher=None,
    ):
        self._store_dir = Path(store_dir) if store_dir else _DEFAULT_STORE_DIR
        self._meta_path = self._store_dir / "_meta.json"

        if fetcher is None:
            from quant.data.fetcher import DataFetcher
            self._fetcher = DataFetcher(cache_dir=self._store_dir)
        else:
            self._fetcher = fetcher

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Save or update data for one symbol.

        Parameters
        ----------
        symbol : str
            Stock code (e.g., "600519" or "600519.SH").
        df : pd.DataFrame
            Data to save. Must have a 'date' column.
        """
        self._store_dir.mkdir(parents=True, exist_ok=True)
        filename = self._symbol_to_filename(symbol)
        path = self._store_dir / filename

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Merge with existing if present
        if path.exists():
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"])
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            combined.to_parquet(path, index=False, engine="pyarrow")
        else:
            df.to_parquet(path, index=False, engine="pyarrow")

        self._update_meta(symbol, df)

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load data for one symbol from disk.

        Parameters
        ----------
        symbol : str
            Stock code.

        Returns
        -------
        pd.DataFrame or None
            None if the symbol is not stored.
        """
        filename = self._symbol_to_filename(symbol)
        path = self._store_dir / filename
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def load_panel(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> DataPanel:
        """
        Load multiple symbols from disk into a DataPanel.

        Parameters
        ----------
        symbols : List[str]
            Symbols to load.
        start : str, optional
            Start date filter (YYYYMMDD or YYYY-MM-DD).
        end : str, optional
            End date filter.

        Returns
        -------
        DataPanel
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.load(symbol)
            if df is not None and not df.empty:
                if start:
                    df = df[df["date"] >= pd.Timestamp(start)]
                if end:
                    df = df[df["date"] <= pd.Timestamp(end)]
                if not df.empty:
                    data[symbol] = df.reset_index(drop=True)

        if not data:
            logger.warning("No data loaded for %d symbols", len(symbols))

        return DataPanel(data)

    def build_panel(
        self,
        symbols: List[str],
        start: str,
        end: str,
        adjust: str = "qfq",
    ) -> DataPanel:
        """
        Build a DataPanel by fetching data (with caching) for all symbols.

        This is the primary entry point for getting data. It will:
          1. Check local store for existing data
          2. Fetch missing data from network (via DataFetcher)
          3. Save fetched data to store
          4. Return a unified DataPanel

        Parameters
        ----------
        symbols : List[str]
            Stock codes to include.
        start : str
            Start date (YYYYMMDD).
        end : str
            End date (YYYYMMDD).
        adjust : str
            Price adjustment mode.

        Returns
        -------
        DataPanel
        """
        data: Dict[str, pd.DataFrame] = {}
        to_fetch: List[str] = []

        # Check what we already have
        for symbol in symbols:
            df = self.load(symbol)
            if df is not None and not df.empty:
                # Check coverage
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                filtered = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
                if not filtered.empty and self._covers_range(filtered, start_ts, end_ts):
                    data[symbol] = filtered.reset_index(drop=True)
                    continue
            to_fetch.append(symbol)

        # Fetch missing
        if to_fetch:
            logger.info(
                "Fetching %d/%d symbols from network...",
                len(to_fetch), len(symbols),
            )
            fetched = self._fetcher.fetch_batch(
                to_fetch, start, end, adjust=adjust
            )
            for symbol, df in fetched.items():
                if df is not None and not df.empty:
                    self.save(symbol, df)
                    data[symbol] = df.reset_index(drop=True)

        return DataPanel(data)

    def update(
        self,
        symbols: Optional[List[str]] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> int:
        """
        Incremental update: fetch new data from last stored date to end_date.

        Parameters
        ----------
        symbols : List[str], optional
            Symbols to update. If None, updates all stored symbols.
        end_date : str, optional
            End date for update (YYYYMMDD). Defaults to today.
        adjust : str
            Price adjustment mode.

        Returns
        -------
        int
            Number of symbols updated.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        if symbols is None:
            symbols = self.get_stored_symbols()

        updated = 0
        for symbol in symbols:
            df = self.load(symbol)
            if df is None or df.empty:
                # No existing data - skip (use build_panel for initial load)
                continue

            last_date = df["date"].max()
            # Start from the day after the last stored date
            next_start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
            end_dt = pd.Timestamp(end_date)

            if last_date >= end_dt:
                continue  # Already up to date

            try:
                new_df = self._fetcher.fetch_daily(symbol, next_start, end_date, adjust)
                if new_df is not None and not new_df.empty:
                    self.save(symbol, new_df)
                    updated += 1
            except Exception as e:
                logger.warning("Update failed for %s: %s", symbol, e)

        logger.info("Incremental update: %d/%d symbols updated", updated, len(symbols))
        self._refresh_meta()
        return updated

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_meta(self) -> dict:
        """
        Get store metadata.

        Returns
        -------
        dict
            Contains: symbols, date_range, last_update, row_counts.
        """
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # Rebuild metadata from disk
        return self._refresh_meta()

    def get_stored_symbols(self) -> List[str]:
        """List all symbols that have parquet files in the store."""
        if not self._store_dir.exists():
            return []
        symbols = []
        for f in self._store_dir.glob("*.parquet"):
            # Convert filename back to symbol
            sym = self._filename_to_symbol(f.name)
            if sym:
                symbols.append(sym)
        return sorted(symbols)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _symbol_to_filename(self, symbol: str) -> str:
        """
        Convert a symbol to a safe filename.

        "600519" -> "600519.SH.parquet"
        "000001" -> "000001.SZ.parquet"
        "600519.SH" -> "600519.SH.parquet"
        """
        if "." in symbol:
            return f"{symbol}.parquet"

        # Infer exchange suffix
        if symbol.startswith(("6", "688")):
            suffix = "SH"
        elif symbol.startswith(("0", "3", "002")):
            suffix = "SZ"
        elif symbol.startswith(("4", "8")):
            suffix = "BJ"
        else:
            suffix = "SZ"  # Default

        return f"{symbol}.{suffix}.parquet"

    def _filename_to_symbol(self, filename: str) -> Optional[str]:
        """Convert a filename back to a symbol string."""
        if not filename.endswith(".parquet"):
            return None
        return filename.replace(".parquet", "")

    def _covers_range(
        self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> bool:
        """Check if data adequately covers [start, end]."""
        if df.empty:
            return False
        data_start = df["date"].min()
        data_end = df["date"].max()
        # Allow 7 calendar days tolerance (long holidays)
        return (
            data_start <= start + pd.Timedelta(days=7)
            and data_end >= end - pd.Timedelta(days=7)
        )

    def _update_meta(self, symbol: str, df: pd.DataFrame) -> None:
        """Update metadata after saving a symbol."""
        meta = self._load_meta_raw()

        # Update symbols list
        symbols = set(meta.get("symbols", []))
        symbols.add(symbol)
        meta["symbols"] = sorted(symbols)

        # Update row counts
        row_counts = meta.get("row_counts", {})
        # Read full file to get total count
        filename = self._symbol_to_filename(symbol)
        path = self._store_dir / filename
        if path.exists():
            full_df = pd.read_parquet(path, columns=["date"])
            row_counts[symbol] = len(full_df)
        meta["row_counts"] = row_counts

        # Update date range
        if not df.empty:
            existing_range = meta.get("date_range", {})
            new_start = df["date"].min().strftime("%Y-%m-%d")
            new_end = df["date"].max().strftime("%Y-%m-%d")

            if existing_range:
                if new_start < existing_range.get("start", "9999"):
                    existing_range["start"] = new_start
                if new_end > existing_range.get("end", "0000"):
                    existing_range["end"] = new_end
            else:
                existing_range = {"start": new_start, "end": new_end}
            meta["date_range"] = existing_range

        meta["last_update"] = datetime.now().isoformat()
        self._save_meta(meta)

    def _refresh_meta(self) -> dict:
        """Rebuild metadata by scanning all parquet files."""
        symbols = self.get_stored_symbols()
        row_counts = {}
        global_min = None
        global_max = None

        for symbol in symbols:
            filename = self._symbol_to_filename(symbol)
            path = self._store_dir / filename
            if path.exists():
                try:
                    df = pd.read_parquet(path, columns=["date"])
                    df["date"] = pd.to_datetime(df["date"])
                    row_counts[symbol] = len(df)
                    if not df.empty:
                        s = df["date"].min()
                        e = df["date"].max()
                        if global_min is None or s < global_min:
                            global_min = s
                        if global_max is None or e > global_max:
                            global_max = e
                except Exception:
                    continue

        meta = {
            "symbols": symbols,
            "date_range": {
                "start": global_min.strftime("%Y-%m-%d") if global_min else None,
                "end": global_max.strftime("%Y-%m-%d") if global_max else None,
            },
            "last_update": datetime.now().isoformat(),
            "row_counts": row_counts,
        }
        self._save_meta(meta)
        return meta

    def _load_meta_raw(self) -> dict:
        """Load raw metadata from disk."""
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_meta(self, meta: dict) -> None:
        """Persist metadata to disk."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
