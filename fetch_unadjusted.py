"""
Fetch unadjusted (不复权) daily OHLCV data for all cached stocks.
Saves to data_store/unadjusted/SYMBOL.parquet.

Usage: python fetch_unadjusted.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from data_cache import get_cached_symbols, UNADJUSTED_DIR
from data_fetcher import DataFetcher

START_DATE = "20180101"
END_DATE = "20260710"


def log(msg):
    print(msg, flush=True)


def fetch_all_unadjusted():
    symbols = get_cached_symbols()

    if not symbols:
        log("No cached symbols found. Run data_cache.py --fetch first.")
        return

    log(f"Found {len(symbols)} cached symbols.")
    log(f"Saving unadjusted data to: {UNADJUSTED_DIR}")
    log(f"Date range: {START_DATE} ~ {END_DATE}")
    log("adjust='' (不复权 / unadjusted)")
    log("-" * 60)

    os.makedirs(UNADJUSTED_DIR, exist_ok=True)

    fetcher = DataFetcher()
    success = 0
    skipped = 0
    failed = 0
    failures = []

    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        path = os.path.join(UNADJUSTED_DIR, f"{sym}.parquet")

        if os.path.exists(path):
            skipped += 1
            if i % 20 == 0 or i == len(symbols):
                elapsed = time.time() - t0
                log(f"  [{i}/{len(symbols)}] {sym} skipped (exists) "
                      f"| {success} ok, {skipped} skip, {failed} fail "
                      f"| {elapsed:.0f}s")
            continue

        try:
            df = fetcher.fetch(
                str(sym),
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="",       # NO adjustment = 不复权
            )
            df.to_parquet(path, index=False)
            success += 1
        except Exception as e:
            failed += 1
            failures.append((sym, str(e)))
            log(f"  [{i}/{len(symbols)}] {sym} FAILED: {e}")

        if i % 20 == 0 or i == len(symbols):
            elapsed = time.time() - t0
            log(f"  [{i}/{len(symbols)}] {sym} done "
                  f"| {success} ok, {skipped} skip, {failed} fail "
                  f"| {elapsed:.0f}s")

    elapsed = time.time() - t0
    log("-" * 60)
    log(f"COMPLETE: {success} fetched, {skipped} skipped, {failed} failed "
          f"in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if failures:
        log(f"\nFAILURES ({len(failures)}):")
        for sym, err in failures:
            log(f"  {sym}: {err}")

    final_files = [f for f in os.listdir(UNADJUSTED_DIR) if f.endswith(".parquet")]
    log(f"\nFiles on disk: {len(final_files)} / {len(symbols)}")


if __name__ == "__main__":
    fetch_all_unadjusted()
