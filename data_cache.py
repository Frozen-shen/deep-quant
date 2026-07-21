"""
数据缓存 — 一次性拉取,本地存储,秒级加载

用法:
  python data_cache.py --fetch    # 拉取并缓存所有股票
  python data_cache.py --status   # 查看缓存状态
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from data_fetcher import DataFetcher

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# 我们的股票池
SYMBOLS = [
    "688981","002371","603986","002049","688012","300782","688396",
    "300033","002230","688111","300454","688561","300750","002594","601012",
    "300274","688005","600519","000858","000568","002714","601318","600036",
    "000001","300760","600276","300122","688180","600760","601668",
]
MARKET, START, END = "a", "20180101", "20260710"


def fetch_all():
    """拉取所有股票并缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetcher = DataFetcher()
    for sym in SYMBOLS:
        path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            print(f"  {sym} ✅ 已缓存,跳过")
            continue
        try:
            print(f"  {sym} 拉取中...", end=" ")
            df = fetcher.fetch(sym, START, END, "qfq", market=MARKET)
            df.to_parquet(path, index=False)
            print(f"{len(df)}条 ✓")
        except Exception as e:
            print(f"❌ {e}")
    print(f"\n缓存完成: {len(os.listdir(CACHE_DIR))} 个文件")


def load(symbol: str) -> pd.DataFrame:
    """从缓存加载单只股票。"""
    path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all(symbols: list = None) -> dict:
    """加载所有缓存股票。返回 {symbol: DataFrame}。"""
    symbols = symbols or SYMBOLS
    result = {}
    for sym in symbols:
        df = load(sym)
        if df is not None:
            result[sym] = df
    return result


def status():
    """查看缓存状态。"""
    if not os.path.exists(CACHE_DIR):
        print("缓存目录不存在。运行 python data_cache.py --fetch")
        return
    files = os.listdir(CACHE_DIR)
    total = 0
    for f in sorted(files):
        path = os.path.join(CACHE_DIR, f)
        df = pd.read_parquet(path)
        size_kb = os.path.getsize(path) / 1024
        total += size_kb
        print(f"  {f.replace('.parquet','')}: {len(df)}条, {size_kb:.0f}KB")
    print(f"  总计: {len(files)}文件, {total:.0f}KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch_all()
    else:
        status()
