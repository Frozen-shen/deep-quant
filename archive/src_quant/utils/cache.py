"""
Disk-based caching with TTL for quant-starter.

Provides transparent caching of expensive computations (data fetches,
factor calculations, model predictions). Supports DataFrame caching
via parquet format and generic object caching via pickle.

Features:
    - Disk-based with configurable TTL (time-to-live)
    - Key derived from hash of input parameters
    - Thread-safe via file locking
    - DataFrame-aware (uses parquet for efficient storage)
    - Decorator interface for easy adoption

Usage:
    from quant.utils.cache import DiskCache, cached

    # Explicit cache usage
    cache = DiskCache("data_store/cache", ttl_hours=24)
    cache.set("my_key", dataframe)
    result = cache.get("my_key")

    # Decorator usage
    @cached(ttl_hours=12)
    def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
        ...
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd

F = TypeVar("F", bound=Callable)

# Sentinel for cache misses
_MISS = object()

# Lock for thread safety on metadata operations
_meta_lock = threading.Lock()


def _make_key(*args, **kwargs) -> str:
    """
    Generate a deterministic hash key from function arguments.

    Handles common types: str, int, float, bool, None, list, tuple, dict.
    For complex objects, falls back to repr().
    """
    key_parts = []

    for arg in args:
        key_parts.append(_serialize_for_hash(arg))

    for k in sorted(kwargs.keys()):
        key_parts.append(f"{k}={_serialize_for_hash(kwargs[k])}")

    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _serialize_for_hash(obj: Any) -> str:
    """Convert an object to a stable string representation for hashing."""
    if obj is None:
        return "None"
    if isinstance(obj, (str, int, float, bool)):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        inner = ",".join(_serialize_for_hash(x) for x in obj)
        return f"[{inner}]"
    if isinstance(obj, dict):
        inner = ",".join(
            f"{k}={_serialize_for_hash(v)}" for k, v in sorted(obj.items())
        )
        return f"{{{inner}}}"
    if isinstance(obj, pd.DataFrame):
        return f"df:{obj.shape}:{hash(pd.util.hash_pandas_object(obj).sum())}"
    if isinstance(obj, pd.Series):
        return f"ser:{obj.shape}:{hash(pd.util.hash_pandas_object(obj).sum())}"
    if isinstance(obj, Path):
        return str(obj)
    # Fallback: use repr
    return repr(obj)


class DiskCache:
    """
    Thread-safe disk-based cache with TTL support.

    Stores data as parquet (for DataFrames) or pickle (for other objects)
    with a JSON metadata sidecar tracking creation time and TTL.

    Args:
        cache_dir: Root directory for cached files.
        ttl_hours: Default time-to-live in hours. None means no expiry.
    """

    def __init__(self, cache_dir: str | Path, ttl_hours: Optional[float] = None):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600 if ttl_hours is not None else None
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create a per-key lock for fine-grained concurrency."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _data_path(self, key: str) -> Path:
        """Path for the cached data file."""
        return self._cache_dir / f"{key}.data"

    def _meta_path(self, key: str) -> Path:
        """Path for the metadata sidecar file."""
        return self._cache_dir / f"{key}.meta.json"

    def _is_expired(self, key: str) -> bool:
        """Check if a cached entry has expired."""
        if self._ttl_seconds is None:
            return False

        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return True

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            created_at = meta.get("created_at", 0)
            return (time.time() - created_at) > self._ttl_seconds
        except (json.JSONDecodeError, OSError):
            return True

    def get(self, key: str, default: Any = _MISS) -> Any:
        """
        Retrieve a cached value by key.

        Args:
            key: Cache key string.
            default: Value to return on miss. If not provided, returns None.

        Returns:
            The cached value, or default/None if not found or expired.
        """
        lock = self._get_lock(key)
        with lock:
            data_path = self._data_path(key)
            if not data_path.exists():
                return None if default is _MISS else default

            if self._is_expired(key):
                self._remove_files(key)
                return None if default is _MISS else default

            try:
                return self._load_data(key)
            except Exception:
                # Corrupted cache entry, remove and return miss
                self._remove_files(key)
                return None if default is _MISS else default

    def set(self, key: str, value: Any, ttl_hours: Optional[float] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key string.
            value: Value to cache. DataFrames are stored as parquet.
            ttl_hours: Override the default TTL for this entry.
        """
        lock = self._get_lock(key)
        with lock:
            self._save_data(key, value)
            self._save_meta(key, ttl_hours)

    def has(self, key: str) -> bool:
        """Check if a non-expired entry exists for the given key."""
        data_path = self._data_path(key)
        if not data_path.exists():
            return False
        return not self._is_expired(key)

    def delete(self, key: str) -> bool:
        """
        Remove a cached entry.

        Returns:
            True if an entry was removed, False if it didn't exist.
        """
        lock = self._get_lock(key)
        with lock:
            existed = self._data_path(key).exists()
            self._remove_files(key)
            return existed

    def clear(self) -> int:
        """
        Remove all cached entries.

        Returns:
            Number of entries removed.
        """
        count = 0
        with _meta_lock:
            for data_file in self._cache_dir.glob("*.data"):
                key = data_file.stem
                self._remove_files(key)
                count += 1
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        entries = list(self._cache_dir.glob("*.data"))
        total_size = sum(f.stat().st_size for f in entries)
        expired = sum(1 for f in entries if self._is_expired(f.stem))
        return {
            "entries": len(entries),
            "expired": expired,
            "active": len(entries) - expired,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self._cache_dir),
        }

    def _save_data(self, key: str, value: Any) -> None:
        """Serialize and write data to disk."""
        data_path = self._data_path(key)

        if isinstance(value, pd.DataFrame):
            value.to_parquet(data_path, engine="pyarrow", index=True)
        elif isinstance(value, pd.Series):
            value.to_frame("__series__").to_parquet(
                data_path, engine="pyarrow", index=True
            )
        else:
            with open(data_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_data(self, key: str) -> Any:
        """Load and deserialize data from disk."""
        data_path = self._data_path(key)

        # Try parquet first (for DataFrames/Series)
        try:
            df = pd.read_parquet(data_path, engine="pyarrow")
            # Check if it was originally a Series
            if list(df.columns) == ["__series__"] and df.index.name is not None:
                return df["__series__"]
            return df
        except Exception:
            pass

        # Fall back to pickle
        with open(data_path, "rb") as f:
            return pickle.load(f)

    def _save_meta(self, key: str, ttl_hours: Optional[float] = None) -> None:
        """Write metadata sidecar."""
        meta_path = self._meta_path(key)
        ttl = ttl_hours * 3600 if ttl_hours is not None else self._ttl_seconds
        meta = {
            "created_at": time.time(),
            "ttl_seconds": ttl,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _remove_files(self, key: str) -> None:
        """Remove data and metadata files for a key."""
        for path in [self._data_path(key), self._meta_path(key)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def cached(
    ttl_hours: Optional[float] = None,
    cache_dir: Optional[str | Path] = None,
    key_func: Optional[Callable[..., str]] = None,
    enabled: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that caches function results to disk.

    The cache key is automatically derived from the function name
    and its arguments. DataFrames are stored as parquet for efficiency.

    Args:
        ttl_hours: Time-to-live in hours. None means cache never expires.
        cache_dir: Directory for cache files. Defaults to "data_store/cache".
        key_func: Custom function to generate cache key from args/kwargs.
                  Signature: key_func(*args, **kwargs) -> str.
        enabled: If False, the decorator is a no-op (useful for debugging).

    Returns:
        Decorated function with caching behavior.

    Example:
        @cached(ttl_hours=24)
        def fetch_daily_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
            # expensive network call
            ...
    """
    if cache_dir is None:
        cache_dir = Path("data_store") / "cache"
    else:
        cache_dir = Path(cache_dir)

    def decorator(func: F) -> F:
        if not enabled:
            return func

        _cache = DiskCache(cache_dir / func.__qualname__.replace(".", "_"), ttl_hours)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            if key_func is not None:
                cache_key = key_func(*args, **kwargs)
            else:
                # Include function name in key to avoid collisions
                raw_key = f"{func.__module__}.{func.__qualname__}|{_make_key(*args, **kwargs)}"
                cache_key = hashlib.sha256(raw_key.encode()).hexdigest()[:32]

            # Try cache hit
            result = _cache.get(cache_key)
            if result is not None:
                return result

            # Cache miss: compute and store
            result = func(*args, **kwargs)
            if result is not None:
                _cache.set(cache_key, result)
            return result

        # Expose cache for manual management
        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.cache_clear = _cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = _cache.stats  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
