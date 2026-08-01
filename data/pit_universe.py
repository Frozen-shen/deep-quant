"""
data/pit_universe.py — Point-in-Time Universe Builder

消除幸存者偏差: 每个调仓日只返回当时存在的股票。
数据源: data/cache/universe_000300.json + universe_000852.json (月度成分)
"""

import json
from pathlib import Path

# Module-level cache to avoid re-reading JSON files
_constituents_cache: dict[str, dict[str, list[str]]] = {}

_CACHE_DIR = Path(__file__).parent / "cache"
_DATA_CACHE_DIR = Path(__file__).parent.parent / "data_cache"


def _load_constituents(filename: str) -> dict[str, list[str]]:
    """Load a constituents JSON file, using module-level cache."""
    if filename not in _constituents_cache:
        filepath = _CACHE_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                _constituents_cache[filename] = json.load(f)
        else:
            _constituents_cache[filename] = {}
    return _constituents_cache[filename]


def _find_month_key(data: dict[str, list[str]], month_key: str) -> str | None:
    """Find the most recent available month key <= requested month_key."""
    if month_key in data:
        return month_key
    # Find most recent month <= requested
    candidates = [k for k in data if k <= month_key]
    if candidates:
        return max(candidates)
    return None


def get_universe(date: str, min_list_days: int = 250) -> list:
    """
    Get point-in-time universe for a given date.

    Args:
        date: Date string in "YYYY-MM-DD" format.
        min_list_days: Minimum listing days filter (reserved for future use).

    Returns:
        Sorted list of unique 6-digit stock codes that were index
        constituents at the given date.
    """
    # Parse date to month key
    month_key = date[:7]  # "YYYY-MM"

    # Load both index constituent files
    csi300 = _load_constituents("universe_000300.json")
    csi1000 = _load_constituents("universe_000852.json")

    # Find the appropriate month for each index
    key_300 = _find_month_key(csi300, month_key)
    key_1000 = _find_month_key(csi1000, month_key)

    stocks: set[str] = set()

    if key_300 is not None:
        stocks.update(csi300[key_300])
    if key_1000 is not None:
        stocks.update(csi1000[key_1000])

    # If no constituent data available at all, fall back to all stocks
    if not stocks:
        return get_all_trading_stocks()

    return sorted(stocks)


def get_all_trading_stocks() -> list:
    """
    Scan data_cache/ directory for *.parquet files and return sorted stock codes.

    WARNING: This fallback has survivorship bias — it includes all stocks
    that currently have data files, regardless of whether they existed
    at any particular historical date.

    Returns:
        Sorted list of 6-digit numeric stock code stems.
    """
    if not _DATA_CACHE_DIR.exists():
        return []
    codes = []
    for f in _DATA_CACHE_DIR.glob("*.parquet"):
        stem = f.stem
        if stem.isdigit() and len(stem) == 6:
            codes.append(stem)
    return sorted(codes)
