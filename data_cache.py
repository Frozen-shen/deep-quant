"""
数据缓存 — 一次性拉取,本地存储,秒级加载

用法:
  python data_cache.py --fetch           # 拉取默认股票池并缓存
  python data_cache.py --fetch-index 000300  # 拉取 CSI300 成分股
  python data_cache.py --status          # 查看缓存状态
"""

import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from data_fetcher import DataFetcher

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# 默认股票池 (向后兼容)
SYMBOLS = [
    "688981","002371","603986","002049","688012","300782","688396",
    "300033","002230","688111","300454","688561","300750","002594","601012",
    "300274","688005","600519","000858","000568","002714","601318","600036",
    "000001","300760","600276","300122","688180","600760","601668",
]
MARKET, START, END = "a", "20180101", "20260710"


def fetch_all(symbols: list = None):
    """拉取所有股票并缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetcher = DataFetcher()
    syms = symbols or SYMBOLS
    for sym in syms:
        path = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            print(f"  {sym} ✅ 已缓存,跳过")
            continue
        try:
            print(f"  {sym} 拉取中...", end=" ")
            df = fetcher.fetch(str(sym), START, END, "qfq", market=MARKET)
            df.to_parquet(path, index=False)
            print(f"{len(df)}条 ✓")
        except Exception as e:
            print(f"❌ {e}")
    print(f"\n缓存完成: {len(os.listdir(CACHE_DIR))} 个文件")


def fetch_index_components(index_code: str = "000300", max_stocks: int = 80):
    """
    Phase 2.2: 获取指数成分股并缓存。

    策略: 从 CSI 300 中选取 Top-N 按成交金额排序的股票,
          避免微盘股和流动性极差的标的。
    """
    symbols = DataFetcher.fetch_index_components(index_code)
    if not symbols:
        print(f"  ⚠️ 指数 {index_code} 无成分股, 回退到默认股票池")
        return SYMBOLS

    # 过滤: 只保留已在缓存中的,或能被成功获取的
    # 实际使用时会由 load_all 自动过滤
    print(f"  准备拉取 {len(symbols)} 只成分股...")
    fetch_all(symbols[:max_stocks])
    return symbols[:max_stocks]


def get_cached_symbols() -> list:
    """获取已缓存的所有股票代码。"""
    if not os.path.exists(CACHE_DIR):
        return []
    return sorted([f.replace(".parquet", "") for f in os.listdir(CACHE_DIR)
                   if f.endswith(".parquet")])


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
    parser.add_argument("--fetch-index", type=str, default=None, help="指数代码, 如 000300")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.fetch_index:
        fetch_index_components(args.fetch_index)
    elif args.fetch:
        fetch_all()
    else:
        status()
