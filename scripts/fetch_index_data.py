"""
fetch_index_data.py — 获取A股主要指数日线数据

产出:
  data/cache/index_csi300.parquet   (沪深300, 000300)
  data/cache/index_csi500.parquet   (中证500, 000905)
  data/cache/index_csi1000.parquet  (中证1000, 000852) ← 主基准

数据源: 腾讯财经 (stock_zh_index_daily_tx)
"""
import os
import sys

# 覆盖系统代理设置，避免连接失败
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")

INDICES = {
    "sh000300": "csi300",
    "sh000905": "csi500",
    "sh000852": "csi1000",
}

START_DATE = "2018-01-01"
END_DATE = "2026-07-31"


def fetch_index(symbol: str, name: str) -> pd.DataFrame:
    """获取单个指数日线 (腾讯数据源)。"""
    import akshare as ak

    print(f"  获取 {name} ({symbol})...", flush=True)
    df = ak.stock_zh_index_daily_tx(symbol=symbol)

    # 腾讯源列: date, open, close, high, low, amount(实为成交量)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"amount": "volume"})

    # 按日期范围过滤
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["close"].pct_change()

    print(f"    {len(df)} 行, {df['date'].min().date()} → {df['date'].max().date()}", flush=True)
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    for symbol, name in INDICES.items():
        out_path = os.path.join(CACHE_DIR, f"index_{name}.parquet")
        try:
            df = fetch_index(symbol, name)
            df.to_parquet(out_path, index=False)
            print(f"  ✓ 已保存: {out_path}", flush=True)
        except Exception as e:
            print(f"  ✗ 失败: {name} — {e}", flush=True)

    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
