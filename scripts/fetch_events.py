"""
批量拉取事件源数据 → 本地 parquet 缓存
后台运行, 不限时, 拉完为止

用法: python scripts/fetch_events.py --source insider
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "event_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_insider_trades():
    """拉取高管/股东增减持数据"""
    print("[insider] 拉取高管增减持...")
    all_data = []

    # 尝试多个日期范围
    for year in [2020, 2021, 2022, 2023, 2024]:
        for quarter in ['0331', '0630', '0930', '1231']:
            date_str = f"{year}{quarter}"
            cache_path = os.path.join(CACHE_DIR, f"insider_{date_str}.parquet")
            if os.path.exists(cache_path):
                df = pd.read_parquet(cache_path)
                all_data.append(df)
                print(f"  {date_str}: {len(df)}条 (缓存)")
                continue

            try:
                df = ak.stock_ggcg_em(symbol=date_str)
                if df is not None and len(df) > 0:
                    df.to_parquet(cache_path, index=False)
                    all_data.append(df)
                    print(f"  {date_str}: {len(df)}条 ✓")
            except Exception as e:
                print(f"  {date_str}: {e}")

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        out = os.path.join(CACHE_DIR, "insider_all.parquet")
        df.to_parquet(out, index=False)
        print(f"[insider] 总计 {len(df)}条 → {out}")


def fetch_restricted_shares():
    """拉取限售解禁数据"""
    print("[restricted] 拉取限售解禁...")
    try:
        df = ak.stock_restricted_release_queue_em()
        out = os.path.join(CACHE_DIR, "restricted_shares.parquet")
        df.to_parquet(out, index=False)
        print(f"[restricted] {len(df)}条 → {out}")
    except Exception as e:
        print(f"[restricted] Error: {e}")


def fetch_analyst_ratings(stock_list: list = None):
    """拉取分析师评级 (需要股票列表, 逐只拉取)"""
    print("[analyst] 拉取分析师评级...")
    if stock_list is None:
        from data_cache import get_cached_symbols
        stock_list = get_cached_symbols()[:200]  # 前200只

    all_data = []
    for i, sym in enumerate(stock_list):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(stock_list)}")
        cache_path = os.path.join(CACHE_DIR, f"analyst_{sym}.parquet")
        if os.path.exists(cache_path):
            all_data.append(pd.read_parquet(cache_path))
            continue
        try:
            df = ak.stock_analyst_recommend_em(symbol=sym)
            if df is not None and len(df) > 0:
                df['股票代码'] = sym
                df.to_parquet(cache_path, index=False)
                all_data.append(df)
        except:
            pass

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        out = os.path.join(CACHE_DIR, "analyst_all.parquet")
        df.to_parquet(out, index=False)
        print(f"[analyst] 总计 {len(df)}条 → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["insider", "restricted", "analyst", "all"], default="all")
    args = parser.parse_args()

    if args.source in ("insider", "all"):
        fetch_insider_trades()
    if args.source in ("restricted", "all"):
        fetch_restricted_shares()
    if args.source in ("analyst", "all"):
        fetch_analyst_ratings()
