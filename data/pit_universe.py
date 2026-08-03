"""
data/pit_universe.py — Point-in-Time Universe Builder

消除幸存者偏差: 每个调仓日只返回当时存在的股票。

两种模式 (方案C v4):
  index 模式 (兼容旧代码):
    数据源: data/cache/universe_000300.json + universe_000852.json (月度成分)
    注意: 000852 用 ZZ500 代理 CSI1000 (~500只), 仅50%覆盖
  liquid 模式 (方案C推荐, IC域=回测域):
    数据源: data/cache/universe_liquid.json (月度快照)
    规则: 全市场 + 上市>=min_list_days天 + 20日均成交额>=min_amount
    天然PIT: 只使用<=T的行情, 无幸存者偏差
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Module-level cache to avoid re-reading JSON files
_constituents_cache: dict[str, dict[str, list[str]]] = {}

_CACHE_DIR = Path(__file__).parent / "cache"
_DATA_CACHE_DIR = Path(__file__).parent.parent / "data_cache"
_DATA_STORE_DIR = Path(__file__).parent.parent / "data_store"

# liquid universe 构建参数 (与 config.yaml universe 段同步)
LIQUID_MIN_LIST_DAYS = 250       # 最少上市交易日 (~1年)
LIQUID_MIN_AMOUNT = 5_000_000    # 20日均成交额下限 (元)
LIQUID_LOOKBACK = 20             # 流动性回看窗口 (交易日)
_LIQUID_CACHE = None             # dict[str, list[str]] 月度快照缓存


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
    Get point-in-time universe for a given date (index constituents).

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


def get_liquid_universe(date: str,
                        min_list_days: int = LIQUID_MIN_LIST_DAYS,
                        min_amount: float = LIQUID_MIN_AMOUNT) -> list:
    """
    Get point-in-time LIQUID universe (方案C推荐, 全市场+流动性过滤).

    月度快照规则 (每个月底 T):
      1. 上市天数 = 截至 T 的可交易天数 >= min_list_days
      2. 20日均成交额 = T 前最后 LIQUID_LOOKBACK 个交易日的
         amount 均值 (amount 为 NaN 时用 volume*close 估算) >= min_amount

    天然 PIT: 每个月份只使用该月末之前的数据构建, 无幸存者偏差。

    Args:
        date: Date string in "YYYY-MM-DD" format.
        min_list_days: Minimum listing days filter.
        min_amount: Minimum 20-day average amount (CNY).

    Returns:
        Sorted list of 6-digit stock codes in the liquid universe.
    """
    month_key = date[:7]
    snapshots = _load_liquid_universe()
    if not snapshots:
        return get_all_trading_stocks()
    key = _find_month_key(snapshots, month_key)
    if key is None:
        return get_all_trading_stocks()
    return sorted(snapshots[key])


def _load_liquid_universe() -> dict[str, list[str]]:
    """Load the monthly liquid universe snapshot (cached in module)."""
    global _LIQUID_CACHE
    if _LIQUID_CACHE is not None:
        return _LIQUID_CACHE
    filepath = _CACHE_DIR / "universe_liquid.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            _LIQUID_CACHE = json.load(f)
    else:
        _LIQUID_CACHE = {}
    return _LIQUID_CACHE


def build_liquid_universe_cache(start: str = "2015-01-01",
                                end: str = "2026-07-31",
                                min_list_days: int = LIQUID_MIN_LIST_DAYS,
                                min_amount: float = LIQUID_MIN_AMOUNT,
                                progress: bool = True) -> dict:
    """
    Build the monthly liquid universe snapshot from data_store parquet files.

    对每只股票只读 date/close/volume/amount 四列, 按月末构建候选集。
    输出: data/cache/universe_liquid.json
      { "2025-01": ["000001", ...], ... }
    """
    global _LIQUID_CACHE

    if not _DATA_STORE_DIR.exists():
        raise FileNotFoundError(f"data_store 不存在: {_DATA_STORE_DIR}")

    files = sorted(_DATA_STORE_DIR.glob("*.parquet"))
    files = [f for f in files if f.stem.isdigit() and len(f.stem) == 6]

    months = pd.period_range(start[:7], end[:7], freq="M")
    month_ends = [str(m.end_time.date()) for m in months]
    monthly_stocks: dict[str, set[str]] = {str(m): set() for m in months}
    n_failed = 0

    for i, f in enumerate(files):
        code = f.stem
        try:
            df = pd.read_parquet(f, columns=["date", "close", "volume", "amount"])
        except Exception:
            n_failed += 1
            continue
        if df.empty:
            n_failed += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()

        # 估算成交额: amount NaN 时用 volume*close (volume 单位=手, 一手=100股)
        amount = df["amount"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        est = np.where(np.isnan(amount), volume * close * 100.0, amount)
        valid_close = ~np.isnan(close)

        # 对每个月末 T 判断是否入选 (只使用 <=T 的数据 → 天然 PIT)
        for mi, mkey in enumerate(monthly_stocks):
            me = month_ends[mi]
            pos = int(np.searchsorted(dates, me, side="right"))
            if pos == 0:
                continue
            # 1. 上市天数 = 截至该月末的可交易行数
            if pos < min_list_days:
                continue
            # 2. 流动性: 最后 LOOKBACK 个交易日的成交额均值
            lo = max(0, pos - LIQUID_LOOKBACK)
            amt_slice = est[lo:pos]
            vc_slice = valid_close[lo:pos]
            if vc_slice.sum() < LIQUID_LOOKBACK * 0.5:  # 停牌/缺数据过多
                continue
            avg_amt = float(np.nanmean(amt_slice)) if amt_slice.size else 0.0
            if not np.isnan(avg_amt) and avg_amt >= min_amount:
                monthly_stocks[mkey].add(code)

        if progress and (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] 股票已扫描")

    out = {mkey: sorted(codes) for mkey, codes in monthly_stocks.items()}
    out_path = _CACHE_DIR / "universe_liquid.json"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    _LIQUID_CACHE = out
    sizes = [len(v) for v in out.values()]
    print(f"liquid universe 已保存: {out_path}")
    print(f"  月份数: {len(out)}, 平均股票数: "
          f"{int(np.mean(sizes)) if sizes else 0}, 读取失败: {n_failed}")
    print(f"  规模范围: {min(sizes)} ~ {max(sizes)}")
    return out


def get_all_trading_stocks() -> list:
    """
    Scan data_store/ directory for *.parquet files and return sorted stock codes.

    WARNING: This fallback has survivorship bias — it includes all stocks
    that currently have data files, regardless of whether they existed
    at any particular historical date.

    Returns:
        Sorted list of 6-digit numeric stock code stems.
    """
    store_dir = _DATA_STORE_DIR if _DATA_STORE_DIR.exists() else _DATA_CACHE_DIR
    if not store_dir.exists():
        return []
    codes = []
    for f in store_dir.glob("*.parquet"):
        stem = f.stem
        if stem.isdigit() and len(stem) == 6:
            codes.append(stem)
    return sorted(codes)
