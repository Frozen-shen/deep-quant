"""
scripts/fetch_baostock_minute.py — 用 baostock 批量拉取全市场分钟线数据

数据源: baostock (免费, 无需注册, 历史深度 2022-至今)
  - 频率: 5分钟 / 15分钟
  - 列: date, time, code, open, high, low, close, volume, amount
  - 前复权

缓存: data_store/minute_15m/{symbol}.parquet 或 data_store/minute_5m/{symbol}.parquet

用法:
  py scripts/fetch_baostock_minute.py                          # 全市场 15分钟
  py scripts/fetch_baostock_minute.py --freq 5                 # 全市场 5分钟
  py scripts/fetch_baostock_minute.py --offset 0 --max 600     # 分片并行
  py scripts/fetch_baostock_minute.py --offset 600 --max 600
  py scripts/fetch_baostock_minute.py --offset 1200 --max 600
  py scripts/fetch_baostock_minute.py --offset 1800 --max 600
  py scripts/fetch_baostock_minute.py --offset 2400            # 剩余全部

预计耗时 (15分钟线, 2022-2026):
  单进程: ~11.5s/只 × 3000 ≈ 9.5 小时
  5进程并行: ~2 小时
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# 配置
START_DATE = "2022-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_FIRST = True  # 跳过已有缓存的股票


def get_cache_dir(freq: str) -> str:
    return os.path.join(BASE_DIR, "data_store", f"minute_{freq}m")


def get_universe() -> list:
    """获取 A 股代码列表 — 用回测实际使用的宇宙 (data_store/)。"""
    store_dir = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(store_dir):
        files = [f.replace(".parquet", "") for f in os.listdir(store_dir)
                 if f.endswith(".parquet") and not f.startswith("index_")
                 and "minute" not in f]
        if len(files) > 100:
            return sorted(files)
    raise RuntimeError("data_store/ 中未找到日线数据")


def to_baostock_symbol(code: str) -> str:
    """600519 -> sh.600519, 000001 -> sz.000001"""
    if code.startswith("6"):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def fetch_one(bs, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """拉取单只股票的分钟线数据。"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,time,code,open,high,low,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency=freq,
        adjustflag="2",  # 前复权
    )
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 解析时间列: "20240102093500000" -> datetime
    df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
    df["day"] = pd.to_datetime(df["date"])

    # 只保留有效行
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    return df[["datetime", "day", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Baostock 分钟线批量拉取")
    parser.add_argument("--freq", type=str, default="15", choices=["5", "15"],
                        help="K线频率: 5 或 15 (分钟)")
    parser.add_argument("--offset", type=int, default=0,
                        help="起始偏移 (用于并行分片)")
    parser.add_argument("--max", type=int, default=None,
                        help="最多拉取数量")
    parser.add_argument("--start", type=str, default=START_DATE,
                        help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=END_DATE,
                        help="结束日期 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true",
                        help="强制重新拉取 (忽略缓存)")
    args = parser.parse_args()

    freq = args.freq
    cache_dir = get_cache_dir(freq)
    os.makedirs(cache_dir, exist_ok=True)

    universe = get_universe()
    total = len(universe)

    # 分片
    symbols = universe[args.offset:]
    if args.max:
        symbols = symbols[:args.max]

    print(f"═══ Baostock {freq}分钟线拉取 ═══")
    print(f"  宇宙: {total} 只, 本次: {len(symbols)} 只 (offset={args.offset})")
    print(f"  日期: {args.start} ~ {args.end}")
    print(f"  缓存: {cache_dir}")
    print(f"  强制: {args.force}")
    print()

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return

    success = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, code in enumerate(symbols):
        cache_path = os.path.join(cache_dir, f"{code}.parquet")

        # 缓存检查
        if not args.force and os.path.exists(cache_path):
            try:
                existing = pd.read_parquet(cache_path)
                if len(existing) > 100:  # 至少有几天数据
                    skipped += 1
                    continue
            except Exception:
                pass

        # 拉取
        bs_sym = to_baostock_symbol(code)
        try:
            df = fetch_one(bs, bs_sym, freq, args.start, args.end)
            if len(df) > 0:
                df.to_parquet(cache_path, index=False)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [ERROR] {code}: {e}")

        # 进度
        done = i + 1
        if done % 50 == 0 or done == len(symbols):
            elapsed = time.time() - t0
            speed = done / elapsed * 60
            eta = (len(symbols) - done) / max(speed / 60, 0.01) / 60
            print(f"  [{done}/{len(symbols)}] 成功={success} 跳过={skipped} "
                  f"失败={failed} | {speed:.0f}只/分 | ETA {eta:.0f}min")

    bs.logout()

    elapsed = time.time() - t0
    print(f"\n═══ 完成 ═══")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  成功: {success}, 跳过(缓存): {skipped}, 失败: {failed}")
    print(f"  缓存目录: {cache_dir}")

    # 统计
    files = [f for f in os.listdir(cache_dir) if f.endswith(".parquet")]
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files)
    print(f"  总文件: {len(files)}, 总大小: {total_size/1024/1024:.0f}MB")


if __name__ == "__main__":
    main()
