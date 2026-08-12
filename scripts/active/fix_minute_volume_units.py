"""
scripts/active/fix_minute_volume_units.py — 修复分钟数据 volume 单位混用

背景 (2026-08-12 审计): 5m/15m 数据历史 volume 单位=股, 但 2026-08-04~06
起增量补拉的数据被写成"手" (另一数据源), 单位在文件内部混用。

修复: 逐日检测 (当日 amount/vol 中位数 vs 当日 close 中位数比值),
  50<ratio<200 → 手 → ×100 转股; 0.5<ratio<2 → 股 (不变)。
输出到原路径 (先备份到 minute_<freq>_bak/)。

用法:
  py scripts/active/fix_minute_volume_units.py [--freq 5m] [--dry-run]
"""
import argparse
import os
import shutil
import sys

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE_DIR, "data_store")


def fix_file(path: str, dry_run: bool) -> int:
    """修复单文件, 返回修正的日期数。"""
    df = pd.read_parquet(path)
    if "volume" not in df.columns or len(df) == 0:
        return 0
    df = df.copy()
    day_col = "day" if "day" in df.columns else "时间"
    df["_day"] = pd.to_datetime(df[day_col]).dt.date
    mask = (df["volume"] > 0) & (df["close"] > 0) & (df["amount"] > 0)
    sub = df[mask].copy()
    if len(sub) == 0:
        return 0
    sub["avg"] = sub["amount"] / sub["volume"]
    # 逐日判定单位
    daily = sub.groupby("_day").agg(avg=("avg", "median"), close=("close", "median"))
    daily["ratio"] = daily["avg"] / daily["close"]
    hand_days = set(daily[daily["ratio"].between(50, 200)].index)  # 手
    if not hand_days:
        return 0
    if not dry_run:
        hand_mask = df["_day"].isin(hand_days)
        df.loc[hand_mask, "volume"] = df.loc[hand_mask, "volume"] * 100.0
        df = df.drop(columns=["_day"])
        df.to_parquet(path, index=False)
    return len(hand_days)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=str, default="all", choices=["5m", "15m", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="最多处理文件数 (0=全部)")
    args = parser.parse_args()

    freqs = ["5m", "15m"] if args.freq == "all" else [args.freq]
    total_files = total_days = 0
    for freq in freqs:
        d = os.path.join(DATA, f"minute_{freq}")
        if not os.path.isdir(d):
            print(f"目录不存在: {d}")
            continue
        # 备份 (首次)
        bak = os.path.join(DATA, f"minute_{freq}_bak")
        if not os.path.isdir(bak):
            os.makedirs(bak)
        files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
        if args.limit:
            files = files[:args.limit]
        n_fix = n_ok = 0
        for f in files:
            path = os.path.join(d, f)
            if not args.dry_run and not os.path.exists(os.path.join(bak, f)):
                shutil.copy2(path, os.path.join(bak, f))
            try:
                days = fix_file(path, args.dry_run)
                if days:
                    n_fix += 1
                    total_days += days
                n_ok += 1
            except Exception:
                pass
        total_files += n_fix
        print(f"[minute_{freq}] 处理 {n_ok} 只, 修正单位 {n_fix} 只 "
              f"({total_days} 天手→股)" + (" [dry-run]" if args.dry_run else ""))

    print(f"\n总计: {total_files} 只文件被修正")
    if not args.dry_run:
        print("备份在 data_store/minute_*_bak/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
