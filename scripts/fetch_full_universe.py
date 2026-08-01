"""
Fetch full A-share universe daily data.
Target: 3000+ stocks, 2018-01-01 to present.
Rate limited to 3 requests/second to avoid akshare bans.

Data sources (with automatic fallback):
  Primary:  akshare stock_zh_a_spot_em / stock_zh_a_hist (eastmoney)
  Fallback: SSE/SZSE stock list + Tencent kline API

Usage:
  py scripts/fetch_full_universe.py                  # Full fetch (2+ hours)
  py scripts/fetch_full_universe.py --limit 10       # Test with 10 stocks
  py scripts/fetch_full_universe.py --force           # Re-fetch everything
  py scripts/fetch_full_universe.py --start 20200101  # Override start date
  py scripts/fetch_full_universe.py --proxy http://127.0.0.1:7897  # Use proxy
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_STORE = os.path.join(BASE_DIR, "data_store")
META_FILE = os.path.join(DATA_STORE, "_meta.json")

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Column mapping: akshare Chinese -> English ─────────────────────────
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

# Canonical column order
KEEP_COLS = ["date", "open", "high", "low", "close", "volume",
             "amount", "amplitude", "pct_change", "change", "turnover"]


# ═══════════════════════════════════════════════════════════════════════
#  Proxy handling
# ═══════════════════════════════════════════════════════════════════════

def setup_proxy(proxy_url: Optional[str] = None):
    """
    Configure proxy for all HTTP requests (requests + akshare).

    On Windows, requests reads proxy from the system registry which may
    point to a local proxy (e.g. Clash) that interferes with eastmoney.
    This function patches requests to either bypass the proxy entirely
    or use a user-specified proxy.
    """
    import requests
    import requests.sessions

    if proxy_url:
        _orig_merge = requests.Session.merge_environment_settings

        def _merge_with_proxy(self, url, proxies, stream, verify, cert):
            settings = _orig_merge(self, url, proxies, stream, verify, cert)
            settings["proxies"] = {"http": proxy_url, "https": proxy_url}
            return settings

        requests.Session.merge_environment_settings = _merge_with_proxy
        log.info(f"Using proxy: {proxy_url}")
    else:
        # Bypass system proxy (direct connection)
        _orig_merge = requests.Session.merge_environment_settings

        def _merge_no_proxy(self, url, proxies, stream, verify, cert):
            settings = _orig_merge(self, url, proxies, stream, verify, cert)
            settings["proxies"] = {}
            return settings

        requests.Session.merge_environment_settings = _merge_no_proxy

        # Also patch module-level requests.get (used by stock_zh_a_hist)
        _orig_get = requests.get

        def _get_no_proxy(url, **kwargs):
            with requests.Session() as s:
                s.trust_env = False
                return s.get(url, **kwargs)

        requests.get = _get_no_proxy
        log.info("System proxy bypassed (direct connection)")


def _make_session():
    """Create a requests session that ignores system proxy."""
    import requests
    s = requests.Session()
    s.trust_env = False
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return s


# ═══════════════════════════════════════════════════════════════════════
#  Stock list
# ═══════════════════════════════════════════════════════════════════════

def get_stock_list_eastmoney() -> pd.DataFrame:
    """Get stock list via akshare stock_zh_a_spot_em (eastmoney)."""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    df = df[["代码", "名称"]].copy()
    df.columns = ["code", "name"]
    return df


def get_stock_list_exchange() -> pd.DataFrame:
    """
    Get stock list directly from SSE + SZSE websites.
    Fallback when eastmoney is unreachable.
    Skips Beijing exchange (we filter it out anyway).
    """
    import akshare as ak

    frames = []

    # Shanghai: main board A + STAR market (科创板)
    for label in ("主板A股", "科创板"):
        try:
            sh = ak.stock_info_sh_name_code(symbol=label)
            sh = sh[["证券代码", "证券简称"]].copy()
            sh.columns = ["code", "name"]
            frames.append(sh)
        except Exception as e:
            log.warning(f"  SSE {label} failed: {e}")

    # Shenzhen: A-share list
    try:
        sz = ak.stock_info_sz_name_code(symbol="A股列表")
        sz["A股代码"] = sz["A股代码"].astype(str).str.zfill(6)
        sz = sz[["A股代码", "A股简称"]].copy()
        sz.columns = ["code", "name"]
        frames.append(sz)
    except Exception as e:
        log.warning(f"  SZSE A-share list failed: {e}")

    if not frames:
        raise RuntimeError("All exchange stock list sources failed")

    return pd.concat(frames, ignore_index=True)


def get_stock_list() -> pd.DataFrame:
    """
    Fetch all A-share stocks with automatic fallback.

    Filters out:
      - ST / *ST stocks
      - Beijing exchange (codes starting with 8 or 4)
    Keeps:
      - Shanghai: 6xxxxx
      - Shenzhen: 0xxxxx, 3xxxxx
    """
    # Try eastmoney first, fall back to exchange websites
    try:
        log.info("Fetching stock list from eastmoney...")
        df = get_stock_list_eastmoney()
        source = "eastmoney"
    except Exception as e:
        log.warning(f"Eastmoney stock list failed: {e}")
        log.info("Falling back to SSE/SZSE exchange websites...")
        df = get_stock_list_exchange()
        source = "SSE/SZSE"

    total = len(df)
    log.info(f"  Total stocks ({source}): {total}")

    # Normalize code column
    df["code"] = df["code"].astype(str).str.zfill(6)

    # Filter by exchange: keep SH (6) and SZ (0, 3)
    mask = df["code"].str.startswith(("6", "0", "3"))
    df = df[mask].copy()
    log.info(f"  After exchange filter (SH/SZ only): {len(df)}")

    # Filter out ST stocks
    name = df["name"].astype(str)
    st_mask = name.str.contains("ST", case=False, na=False)
    df = df[~st_mask].copy()
    log.info(f"  After ST filter: {len(df)}")

    df = df.reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Historical data fetching
# ═══════════════════════════════════════════════════════════════════════

def _fetch_hist_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV via akshare stock_zh_a_hist (eastmoney push2his)."""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date, adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=COLUMN_MAP)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fetch_hist_tencent(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fallback: fetch daily OHLCV from Tencent finance API.

    The API returns at most ~640 recent data points per request, so we
    fetch in chunks by walking the end_date backwards until we cover the
    full requested range.

    Returns date, open, high, low, close, volume plus computed fields.
    amount and turnover are not available from this source (set to NaN).
    """
    from datetime import datetime as _dt, timedelta as _td

    prefix = "sh" if code.startswith("6") else "sz"
    symbol = f"{prefix}{code}"

    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    session = _make_session()
    all_klines = []
    current_end = ed
    max_chunks = 10  # safety limit

    for _ in range(max_chunks):
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={symbol},day,{sd},{current_end},2000,qfq"
        )
        resp = session.get(url, timeout=15)
        data = resp.json()

        # Response may have data as dict (success) or list (empty/error)
        payload = data.get("data")
        if not isinstance(payload, dict):
            break

        stock_data = payload.get(symbol)
        if not isinstance(stock_data, dict):
            break

        klines = stock_data.get("qfqday") or stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines  # prepend older data

        # Walk end_date back to day before the oldest point in this chunk
        first_date = _dt.strptime(klines[0][0], "%Y-%m-%d") - _td(days=1)
        current_end = first_date.strftime("%Y-%m-%d")

        if current_end < sd:
            break

    if not all_klines:
        return pd.DataFrame()

    # Tencent format: [date, open, close, high, low, volume]
    rows = []
    for k in all_klines:
        rows.append({
            "date": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Deduplicate (chunks may overlap by one day)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    # Compute derived fields
    df["pct_change"] = df["close"].pct_change() * 100
    df["change"] = df["close"].diff()
    prev_close = df["close"].shift(1)
    df["amplitude"] = (df["high"] - df["low"]) / prev_close * 100

    # Not available from Tencent
    df["amount"] = np.nan
    df["turnover"] = np.nan

    # Reorder columns
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    return df


def fetch_stock(code: str, start_date: str, end_date: str,
                max_retries: int = 3,
                use_tencent_fallback: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV for a single stock with exponential backoff.

    When use_tencent_fallback is True (eastmoney known to be down),
    goes directly to Tencent API without wasting time on akshare retries.

    Returns DataFrame with English column names, or None on failure.
    """
    # Skip akshare entirely if we already know eastmoney is down
    if not use_tencent_fallback:
        last_err = None
        for attempt in range(max_retries):
            try:
                df = _fetch_hist_akshare(code, start_date, end_date)
                if df is not None and not df.empty:
                    return df
                if df is not None and df.empty:
                    log.warning(f"  {code}: empty DataFrame from akshare")
                    return None
            except Exception as e:
                last_err = e
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                if attempt < max_retries - 1:
                    log.warning(f"  {code}: akshare attempt {attempt+1} failed "
                                f"({e}), retrying in {wait}s...")
                    time.sleep(wait)

        # Akshare failed - try Tencent fallback
        log.info(f"  {code}: akshare failed ({last_err}), "
                 f"trying Tencent fallback...")

    # Tencent fallback (either primary fallback or after akshare failure)
    try:
        df = _fetch_hist_tencent(code, start_date, end_date)
        if df is not None and not df.empty:
            return df
        log.warning(f"  {code}: Tencent returned empty data")
    except Exception as e:
        log.error(f"  {code}: Tencent fallback failed: {e}")

    log.error(f"  {code}: all attempts failed")
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_stock(code: str, df: pd.DataFrame,
                   source: str = "akshare") -> Tuple[bool, str]:
    """
    Validate fetched data for a single stock.

    Checks:
      - At least 100 rows
      - No all-NaN columns (except amount/turnover when using Tencent)
      - Dates monotonically increasing
      - Prices positive

    Returns (is_valid, reason).
    """
    if len(df) < 100:
        return False, f"only {len(df)} rows (need >= 100)"

    # Check for all-NaN columns
    # Tencent fallback doesn't provide amount/turnover - that's expected
    skip_cols = {"amount", "turnover"} if source == "tencent" else set()
    all_nan = [c for c in df.columns[df.isna().all()].tolist()
               if c not in skip_cols]
    if all_nan:
        return False, f"all-NaN columns: {all_nan}"

    # Check dates monotonically increasing
    if not df["date"].is_monotonic_increasing:
        return False, "dates not monotonically increasing"

    # Check prices positive
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            n_bad = int((df[col] <= 0).sum())
            if n_bad > 0:
                return False, f"{col} has {n_bad} non-positive values"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
#  Main fetch loop
# ═══════════════════════════════════════════════════════════════════════

def run(start_date: str = "20180101",
        end_date: str = "20260731",
        force: bool = False,
        limit: Optional[int] = None):
    """
    Main entry: fetch daily data for all A-share stocks.
    """
    os.makedirs(DATA_STORE, exist_ok=True)

    # ── Stock list ──
    stocks = get_stock_list()
    codes = stocks["code"].tolist()

    if limit:
        codes = codes[:limit]
        log.info(f"Limited to first {limit} stocks for testing")

    total = len(codes)

    # ── Detect if eastmoney is reachable ──
    use_tencent = False
    try:
        log.info("Probing eastmoney API availability...")
        _fetch_hist_akshare("000001", "20240101", "20240105")
        log.info("  Eastmoney API: OK")
    except Exception:
        log.warning("  Eastmoney API: unreachable, will use Tencent fallback")
        use_tencent = True

    log.info(f"Starting fetch: {total} stocks, "
             f"{start_date} -> {end_date}, force={force}")
    if use_tencent:
        log.info("Data source: Tencent finance API (fallback)")
    else:
        log.info("Data source: akshare / eastmoney")

    # ── Stats ──
    done = 0
    skipped = 0
    failed: List[str] = []
    invalid: List[str] = []
    all_dates_min = None
    all_dates_max = None
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        out_path = os.path.join(DATA_STORE, f"{code}.parquet")

        # Resume support
        if not force and os.path.exists(out_path):
            skipped += 1
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                pct = i / total * 100
                log.info(f"[{i}/{total}] {code} skipped (cached) "
                         f"({pct:.1f}%) [{elapsed:.0f}s]")
            continue

        # Rate limiting: ~3 req/s
        if done > 0:
            time.sleep(0.34)

        # Fetch
        df = fetch_stock(code, start_date, end_date,
                         use_tencent_fallback=use_tencent)

        if df is None:
            failed.append(code)
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                pct = i / total * 100
                log.info(f"[{i}/{total}] {code} FAILED ({pct:.1f}%) "
                         f"[{elapsed:.0f}s]")
            continue

        # Validate
        source_label = "tencent" if use_tencent else "akshare"
        is_valid, reason = validate_stock(code, df, source=source_label)
        if not is_valid:
            log.warning(f"  {code}: validation issue - {reason}")
            invalid.append(code)

        # Save
        df.to_parquet(out_path, index=False, engine="pyarrow")
        done += 1

        # Track date range
        dmin, dmax = df["date"].min(), df["date"].max()
        if all_dates_min is None or dmin < all_dates_min:
            all_dates_min = dmin
        if all_dates_max is None or dmax > all_dates_max:
            all_dates_max = dmax

        # Progress every 50 stocks
        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            pct = i / total * 100
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            log.info(f"[{i}/{total}] {code} done ({pct:.1f}%) "
                     f"[{elapsed:.0f}s, ~{eta:.0f}s remaining]")

    # ── Summary ──
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"Fetch complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"  Fetched:  {done}")
    log.info(f"  Skipped:  {skipped} (already cached)")
    log.info(f"  Failed:   {len(failed)}")
    log.info(f"  Invalid:  {len(invalid)}")

    if failed:
        log.info(f"  Failed codes: {failed[:20]}"
                 f"{'...' if len(failed) > 20 else ''}")
    if invalid:
        log.info(f"  Invalid codes: {invalid[:20]}"
                 f"{'...' if len(invalid) > 20 else ''}")

    # ── Metadata ──
    meta = {
        "n_symbols": done + skipped,
        "date_range": [
            all_dates_min.strftime("%Y-%m-%d") if all_dates_min else None,
            all_dates_max.strftime("%Y-%m-%d") if all_dates_max else None,
        ],
        "last_update": datetime.now().isoformat(timespec="seconds"),
        "failed": failed,
        "invalid": invalid,
        "source": ("tencent_fallback" if use_tencent
                   else "akshare stock_zh_a_hist"),
        "start_date": start_date,
        "end_date": end_date,
        "adjust": "qfq",
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"Metadata written to {META_FILE}")

    return meta


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fetch full A-share universe daily OHLCV data")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all stocks (ignore cached parquet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Fetch only N stocks (for testing)")
    parser.add_argument("--start", type=str, default="20180101",
                        help="Start date YYYYMMDD (default: 20180101)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYYMMDD (default: today)")
    parser.add_argument("--proxy", type=str, default=None,
                        help="Proxy URL (e.g. http://127.0.0.1:7897). "
                             "Default: bypass system proxy.")

    args = parser.parse_args()
    end_date = args.end or datetime.now().strftime("%Y%m%d")

    # Setup proxy handling before any network calls
    setup_proxy(args.proxy)

    meta = run(
        start_date=args.start,
        end_date=end_date,
        force=args.force,
        limit=args.limit,
    )

    # Exit code based on failure rate
    total_attempted = len(meta["failed"]) + meta["n_symbols"]
    if meta["failed"] and total_attempted > 0:
        fail_rate = len(meta["failed"]) / total_attempted
        if fail_rate > 0.3:
            log.error(f"High failure rate: {fail_rate:.0%}")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
