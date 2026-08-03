"""
scripts/active/fetch_minute_pit_universe.py — 为 PIT universe 缺失的股票批量拉取 5 分钟数据

数据源 (按优先级):
  1. EastMoney  ak.stock_zh_a_hist_min_em  (最近 ~60 个交易日)
  2. Sina       ak.stock_zh_a_minute       (最近 ~40 个交易日, EM 被墙时回退)

缓存: data_cache/minute/{symbol}.parquet
Schema: day, open, high, low, close, volume, amount (与现有 563 个文件一致)

用法:
  py scripts/active/fetch_minute_pit_universe.py                # 800股PIT宇宙(Dec2025-Jun2026并集)
  py scripts/active/fetch_minute_pit_universe.py --include-july # 含7月1300宇宙
  py scripts/active/fetch_minute_pit_universe.py --max 50       # 只拉前50只(测试)
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "data_cache", "minute")
REQUEST_INTERVAL = 0.5  # 秒, 限速

# 800股 PIT 宇宙的月份 (CSI300+部分, Jul 2026 起扩为 1300)
UNIVERSE_DATES_800 = [
    "2025-12-15", "2026-01-15", "2026-02-16", "2026-03-16",
    "2026-04-15", "2026-05-15", "2026-06-15",
]
UNIVERSE_DATE_JULY = "2026-07-15"


def get_target_universe(include_july: bool = False) -> list:
    """目标股票列表 = 800股PIT宇宙并集 (+ 可选7月1300宇宙)。"""
    from data.pit_universe import get_universe
    universe = set()
    for d in UNIVERSE_DATES_800:
        universe.update(get_universe(d))
    if include_july:
        universe.update(get_universe(UNIVERSE_DATE_JULY))
    return sorted(universe)


def already_cached(symbol: str) -> bool:
    """缓存文件最新 bar 已是最近交易日则跳过。"""
    path = os.path.join(MINUTE_CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path, columns=["day"])
        if len(df) == 0:
            return False
        latest = pd.to_datetime(df["day"]).max()
        return latest.date() >= datetime(2026, 7, 31).date()
    except Exception:
        return False


def fetch_em(symbol: str) -> pd.DataFrame:
    """EastMoney 5分钟线。成交量单位: 手 → ×100 转股。"""
    import akshare as ak
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol, period="5", adjust="qfq",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.rename(columns={
        "时间": "day", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    })
    df["day"] = pd.to_datetime(df["day"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"] * 100  # 手 → 股
    df = df.sort_values("day").reset_index(drop=True)
    return df[["day", "open", "high", "low", "close", "volume", "amount"]]


def fetch_sina(symbol: str) -> pd.DataFrame:
    """Sina 5分钟线 (回退源)。成交量单位已是股。"""
    import akshare as ak
    sina_symbol = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"
    df = ak.stock_zh_a_minute(symbol=sina_symbol, period="5", adjust="qfq")
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df["day"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    df = df.sort_values("day").reset_index(drop=True)
    return df[["day", "open", "high", "low", "close", "volume", "amount"]]


def fetch_one(symbol: str) -> tuple:
    """先试 EM, 失败回退 Sina。返回 (DataFrame, source)。"""
    try:
        df = fetch_em(symbol)
        if len(df) > 0:
            return df, "em"
    except Exception:
        pass
    try:
        df = fetch_sina(symbol)
        if len(df) > 0:
            return df, "sina"
    except Exception:
        pass
    return pd.DataFrame(), None


def main():
    parser = argparse.ArgumentParser(description="拉取PIT宇宙5分钟数据")
    parser.add_argument("--include-july", action="store_true",
                        help="包含7月1300宇宙")
    parser.add_argument("--max", type=int, default=0,
                        help="最多拉取数量 (0=全部)")
    args = parser.parse_args()

    os.makedirs(MINUTE_CACHE_DIR, exist_ok=True)

    universe = get_target_universe(args.include_july)
    cached = [s for s in universe if already_cached(s)]
    todo = [s for s in universe if not already_cached(s)]
    if args.max > 0:
        todo = todo[:args.max]

    print(f"PIT 宇宙: {len(universe)} 只, 已有最新缓存: {len(cached)}")
    print(f"待拉取: {len(todo)} 只, 限速 {REQUEST_INTERVAL}s")
    print(f"预计耗时: ~{len(todo) * (REQUEST_INTERVAL + 0.8) / 60:.0f} 分钟")
    print()

    success = 0
    fail = []
    src_count = {"em": 0, "sina": 0}
    t0 = time.time()

    for i, sym in enumerate(todo):
        df, src = fetch_one(sym)
        if len(df) > 0:
            df.to_parquet(os.path.join(MINUTE_CACHE_DIR, f"{sym}.parquet"),
                          index=False)
            success += 1
            src_count[src] += 1
        else:
            fail.append(sym)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(todo) - i - 1)
            print(f"  [{i+1}/{len(todo)}] ok={success} fail={len(fail)} "
                  f"(em={src_count['em']}, sina={src_count['sina']}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

        time.sleep(REQUEST_INTERVAL)

    elapsed = time.time() - t0
    print(f"\n完成: {success} 成功 (em={src_count['em']}, sina={src_count['sina']}), "
          f"{len(fail)} 失败, 耗时 {elapsed/60:.1f}min")
    if fail:
        print(f"失败列表 (前20): {fail[:20]}")
        import json
        with open(os.path.join(BASE_DIR, "data", "cache",
                               "minute_fetch_failed.json"), "w") as f:
            json.dump(fail, f)

    # 统计最终覆盖率
    minute_files = set(
        f.replace(".parquet", "") for f in os.listdir(MINUTE_CACHE_DIR)
        if f.endswith(".parquet")
    )
    u800 = set(get_target_universe(include_july=False))
    print(f"\n覆盖率: {len(minute_files)} 个分钟文件, "
          f"800股PIT宇宙覆盖 {len(u800 & minute_files)}/{len(u800)}")


if __name__ == "__main__":
    main()
