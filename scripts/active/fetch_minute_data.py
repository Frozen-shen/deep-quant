"""
scripts/fetch_minute_data.py — 批量拉取全市场 5 分钟 K 线数据

数据源: Sina (ak.stock_zh_a_minute)
  - 历史深度: ~40 个交易日 (滚动)
  - 频率: 5 分钟 (48 根/天)
  - 列: day, open, high, low, close, volume, amount

缓存: data_store/minute/{symbol}.parquet (增量合并)

用法:
  py scripts/fetch_minute_data.py                    # 全市场
  py scripts/fetch_minute_data.py --max 100          # 只拉前100只
  py scripts/fetch_minute_data.py --symbols 600519,000858  # 指定股票

预计耗时: ~3000只 × 0.5s ≈ 25 分钟
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

MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "data_store", "minute")
REQUEST_INTERVAL = 0.5  # 秒


def ensure_dir():
    os.makedirs(MINUTE_CACHE_DIR, exist_ok=True)


def get_universe() -> list:
    """获取 A 股代码列表 — 用回测实际使用的宇宙 (data_store/)。"""
    # 优先: data_store/ (回测用的完整宇宙, ~3000只)
    store_dir = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(store_dir):
        files = [f.replace(".parquet", "") for f in os.listdir(store_dir)
                 if f.endswith(".parquet") and not f.startswith("index_")]
        if len(files) > 100:
            return sorted(files)

    # 次选: data_store/ 主库 (旧 data_cache 已冻结为 legacy, 不再扫描)
    cache_dir = os.path.join(BASE_DIR, "data_store")
    if os.path.exists(cache_dir):
        files = [f.replace(".parquet", "") for f in os.listdir(cache_dir)
                 if f.endswith(".parquet") and not f.startswith("index_")]
        if files:
            return sorted(files)

    # 回退: 全市场
    import akshare as ak
    try:
        df = ak.stock_info_a_code_name()
        symbols = df["code"].tolist()
        symbols = [s for s in symbols if s.startswith(("0", "3", "6"))]
        return sorted(symbols)
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def fetch_single(symbol: str) -> pd.DataFrame:
    """
    拉取单只股票的 5 分钟数据 (Sina 源)。

    Returns:
      DataFrame 或空 DataFrame
    """
    import akshare as ak
    import warnings
    warnings.filterwarnings("ignore")

    # Sina 需要前缀: sh/sz
    if symbol.startswith("6"):
        sina_symbol = f"sh{symbol}"
    else:
        sina_symbol = f"sz{symbol}"

    try:
        df = ak.stock_zh_a_minute(symbol=sina_symbol, period="5", adjust="qfq")
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 标准化
        df["day"] = pd.to_datetime(df["day"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "amount" not in df.columns:
            # Sina 有时不返回 amount, 用 close*volume 近似
            df["amount"] = df["close"] * df["volume"]

        df = df.sort_values("day").reset_index(drop=True)
        return df[["day", "open", "high", "low", "close", "volume", "amount"]]

    except Exception:
        return pd.DataFrame()


def merge_and_save(symbol: str, new_df: pd.DataFrame):
    """增量合并并保存。"""
    cache_path = os.path.join(MINUTE_CACHE_DIR, f"{symbol}.parquet")

    if os.path.exists(cache_path):
        try:
            old_df = pd.read_parquet(cache_path)
            if len(old_df) > 0:
                old_df["day"] = pd.to_datetime(old_df["day"])
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["day"], keep="last")
                combined = combined.sort_values("day").reset_index(drop=True)
                combined.to_parquet(cache_path, index=False)
                return
        except Exception:
            pass

    new_df.to_parquet(cache_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="批量拉取分钟数据")
    parser.add_argument("--max", type=int, default=0, help="最大股票数 (0=全部)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前N只 (用于并行)")
    parser.add_argument("--symbols", type=str, default="", help="逗号分隔的代码")
    args = parser.parse_args()

    ensure_dir()

    # 确定标的列表
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_universe()
        if args.offset > 0:
            symbols = symbols[args.offset:]
        if args.max > 0:
            symbols = symbols[:args.max]

    print(f"分钟数据批量获取: {len(symbols)} 只, 限速 {REQUEST_INTERVAL}s")
    print(f"缓存目录: {MINUTE_CACHE_DIR}")
    print(f"预计耗时: ~{len(symbols) * REQUEST_INTERVAL / 60:.0f} 分钟")
    print()

    success = 0
    fail = 0
    start_time = time.time()

    for i, sym in enumerate(symbols):
        # 检查缓存是否已经够新 (今天已拉取过则跳过)
        cache_path = os.path.join(MINUTE_CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                if len(cached) > 0:
                    latest = pd.to_datetime(cached["day"]).max()
                    if latest.date() >= datetime.now().date():
                        success += 1
                        continue
            except Exception:
                pass

        df = fetch_single(sym)
        if len(df) > 0:
            merge_and_save(sym, df)
            success += 1
        else:
            fail += 1

        # 进度
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(symbols)}] 成功 {success}, 失败 {fail}, "
                  f"耗时 {elapsed:.0f}s")

        time.sleep(REQUEST_INTERVAL)

    elapsed = time.time() - start_time
    print(f"\n完成: {success}/{len(symbols)} 只成功, {fail} 只失败")
    print(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 统计缓存状态
    files = [f for f in os.listdir(MINUTE_CACHE_DIR) if f.endswith(".parquet")]
    total_size = sum(os.path.getsize(os.path.join(MINUTE_CACHE_DIR, f)) for f in files)
    print(f"缓存: {len(files)} 个文件, {total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
