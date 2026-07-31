"""Unified data storage: consolidated parquet with MultiIndex (date, symbol)."""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
STORE_PATH = BASE_DIR / "data" / "store" / "all_stocks.parquet"


def build_store(symbols=None, use_float32=True):
    """Merge all individual parquet files into one consolidated file."""
    if symbols is None:
        symbols = [os.path.basename(f).replace('.parquet', '')
                   for f in glob.glob(str(CACHE_DIR / "*.parquet"))]

    frames = []
    for i, sym in enumerate(symbols):
        path = CACHE_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['symbol'] = sym
        frames.append(df)
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(symbols)}", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    if use_float32:
        float_cols = combined.select_dtypes(include='float64').columns
        combined[float_cols] = combined[float_cols].astype(np.float32)

    combined = combined.set_index(['date', 'symbol']).sort_index()

    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(STORE_PATH, engine='pyarrow', compression='snappy')
    size_mb = STORE_PATH.stat().st_size / 1e6
    print(f"Store built: {len(combined)} rows, {len(symbols)} stocks, {size_mb:.1f} MB")
    return combined


def load_panel(symbols=None, start=None, end=None, columns=None):
    """Load panel from consolidated store. Falls back to individual files."""
    if STORE_PATH.exists():
        df = pd.read_parquet(STORE_PATH)
        if symbols:
            df = df[df.index.get_level_values('symbol').isin(symbols)]
        if start:
            df = df[df.index.get_level_values('date') >= pd.Timestamp(start)]
        if end:
            df = df[df.index.get_level_values('date') <= pd.Timestamp(end)]
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]
        return df
    else:
        # Fallback: individual files
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from data_cache import load, get_cached_symbols
        if symbols is None:
            symbols = get_cached_symbols()
        frames = []
        for sym in symbols:
            d = load(sym)
            if d is not None and len(d) > 0:
                d['symbol'] = sym
                frames.append(d)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.set_index(['date', 'symbol']).sort_index()
        if start:
            combined = combined[combined.index.get_level_values('date') >= pd.Timestamp(start)]
        if end:
            combined = combined[combined.index.get_level_values('date') <= pd.Timestamp(end)]
        if columns:
            available = [c for c in columns if c in combined.columns]
            combined = combined[available]
        return combined
