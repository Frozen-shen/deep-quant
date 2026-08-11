"""
scripts/active/build_market_rv.py — 构建市场已实现波动率序列 (路线A v20)

背景: 东财对分钟线 klt=5 限流 (网关/镜像均空返回), baostock 不支持指数分钟线。
替代方案: 用已有全市场个股 5m 数据 (data_store/minute_5m/, 5055只, 2022-01 起)
  计算每日"截面中位数已实现波动率"作为市场波动率状态序列。

市场 RV_t = median( std(intraday_ret) * sqrt(48*252) )  over 全市场股票 (当日≥16根bar)
输出: data/cache/market_rv_5m.parquet (date, rv_median, n_stocks)

用法:
  py scripts/active/build_market_rv.py                # 全量
  py scripts/active/build_market_rv.py --max 500      # 抽样 (测试)
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

MINUTE_DIR = os.path.join(BASE_DIR, "data_store", "minute_5m")
OUT_PATH = os.path.join(BASE_DIR, "data", "cache", "market_rv_5m.parquet")
MIN_BARS_PER_DAY = 16  # 至少半天bar才算有效日 (5m=48根/天)
ANNUALIZE = np.sqrt(48 * 252)


def _day_rv(df: pd.DataFrame) -> pd.Series:
    """按日计算已实现波动率 (5m): std(intraday_ret) * sqrt(48*252)"""
    close = df["close"]
    ret = close.groupby(df["day"]).apply(
        lambda s: np.diff(s.values) / s.values[:-1] if len(s) > 1 else np.array([np.nan]))
    rv = ret.apply(lambda r: float(np.nanstd(r) * ANNUALIZE)
                   if np.isfinite(r).sum() >= MIN_BARS_PER_DAY - 1 else np.nan)
    rv.index = pd.to_datetime(rv.index)
    return rv


def main():
    parser = argparse.ArgumentParser(description="构建市场已实现波动率序列")
    parser.add_argument("--max", type=int, default=0, help="抽样股票数 (0=全部)")
    args = parser.parse_args()

    if not os.path.isdir(MINUTE_DIR):
        print(f"分钟目录不存在: {MINUTE_DIR}")
        return 1
    files = sorted(f for f in os.listdir(MINUTE_DIR) if f.endswith(".parquet"))
    if args.max > 0:
        files = files[:args.max]
    print(f"市场RV构建: {len(files)} 只 (5m, {MINUTE_DIR})")
    print(f"输出: {OUT_PATH}")
    print()

    all_rv = []  # [Series(date->rv)]
    t0 = time.time()
    n_ok = 0
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(os.path.join(MINUTE_DIR, f),
                                 columns=["day", "close"])
            df["day"] = pd.to_datetime(df["day"])
            rv = _day_rv(df)
            rv = rv[rv.notna()]
            if len(rv) > 0:
                all_rv.append(rv)
                n_ok += 1
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(files)}] 有效 {n_ok}, {el:.0f}s")

    if not all_rv:
        print("无有效数据")
        return 1

    # 逐日截面中位数 (对齐交易日: outer join 后取 median)
    rv_panel = pd.concat(all_rv, axis=1)
    rv_panel = rv_panel.sort_index()
    daily = pd.DataFrame({
        "rv_median": rv_panel.median(axis=1, skipna=True),
        "n_stocks": rv_panel.notna().sum(axis=1),
    }).reset_index()
    daily.columns = ["date", "rv_median", "n_stocks"]
    daily["date"] = pd.to_datetime(daily["date"])

    # 过滤: 当日至少 100 只有效 (早期数据不足)
    daily = daily[daily["n_stocks"] >= 100]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    daily.to_parquet(OUT_PATH, index=False)

    el = time.time() - t0
    print(f"\n完成: {len(files)} 只扫描, {n_ok} 只有效, {el:.0f}s")
    print(f"输出: {len(daily)} 个交易日 ({daily['date'].min().date()} ~ {daily['date'].max().date()})")
    print(f"RV中位数范围: {daily['rv_median'].min():.2f} ~ {daily['rv_median'].max():.2f} (年化)")
    print(f"样本: {daily.head(3).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
