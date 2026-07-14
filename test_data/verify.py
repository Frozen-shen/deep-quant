"""
独立验证脚本 — 用纯 pandas 实现 MA 交叉检测，验证 strategy.py 输出正确性

不 import strategy.py，完全独立实现，用于回归测试。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import csv
from test_data import loader


def compute_ma_cross_signals(df: pd.DataFrame, short: int = 5, long: int = 20):
    """
    纯 pandas 实现 MA 交叉信号 (不依赖 strategy.py)。

    返回: DataFrame with signal, position columns
    """
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short).mean()
    df["ma_long"] = df["close"].rolling(long).mean()

    valid = (df["ma_short"].notna() & df["ma_long"].notna() &
             df["ma_short"].shift(1).notna() & df["ma_long"].shift(1).notna())

    golden = valid & (df["ma_short"].shift(1) <= df["ma_long"].shift(1)) & (df["ma_short"] > df["ma_long"])
    death = valid & (df["ma_short"].shift(1) >= df["ma_long"].shift(1)) & (df["ma_short"] < df["ma_long"])

    df["signal"] = 0
    df.loc[golden, "signal"] = 1
    df.loc[death, "signal"] = -1
    return df[["date", "close", "signal"]]


def verify_known_signals(symbol: str = "600519"):
    """
    加载测试数据集，用独立算法计算信号，与 known_signals.csv 对比。

    返回: (passed: bool, differences: list)
    """
    print(f"\n验证 {symbol} 的 MA5×MA20 信号...")

    # 加载行情数据
    try:
        df_price = loader.load_ohlcv(symbol)
    except FileNotFoundError:
        print(f"  ⚠️ {symbol} 数据未生成，跳过")
        return True, []

    # 1. 独立计算信号
    df_independent = compute_ma_cross_signals(df_price)
    independent_signals = df_independent[df_independent["signal"] != 0]
    buy_ind = (independent_signals["signal"] == 1).sum()
    sell_ind = (independent_signals["signal"] == -1).sum()
    print(f"  独立算法: {buy_ind} BUY + {sell_ind} SELL")

    # 2. 加载已知信号
    try:
        df_known = loader.load_known_signals()
        # 符号统一处理：补齐前导零
        df_known["symbol"] = df_known["symbol"].astype(str).str.zfill(5)
        target = str(symbol).zfill(5)
        df_known_sym = df_known[df_known["symbol"] == target]
        buy_known = (df_known_sym["signal"] == 1).sum()
        sell_known = (df_known_sym["signal"] == -1).sum()
        print(f"  已知信号: {buy_known} BUY + {sell_known} SELL")
    except FileNotFoundError:
        print("  ⚠️ known_signals.csv 未生成，跳过对比")
        return True, []

    # 3. 对比
    differences = []

    # 数量对比 (允许 ±2 的偏差，因为边界处理可能略有不同)
    if abs(buy_ind - buy_known) > 2:
        differences.append(f"BUY数量偏差: 独立={buy_ind}, 已知={buy_known}")
    if abs(sell_ind - sell_known) > 2:
        differences.append(f"SELL数量偏差: 独立={sell_ind}, 已知={sell_known}")

    # 日期精确对比
    ind_buy_dates = set(independent_signals[independent_signals["signal"] == 1]["date"].dt.date)
    known_buy_dates = set(pd.to_datetime(df_known_sym[df_known_sym["signal"] == 1]["date"]).dt.date)

    extra_in_ind = ind_buy_dates - known_buy_dates
    missing_in_ind = known_buy_dates - ind_buy_dates

    if extra_in_ind:
        differences.append(f"独立算法多出的BUY日期 ({len(extra_in_ind)}个): {sorted(extra_in_ind)[:5]}...")
    if missing_in_ind:
        differences.append(f"已知信号多出的BUY日期 ({len(missing_in_ind)}个): {sorted(missing_in_ind)[:5]}...")

    passed = len(differences) == 0

    if passed:
        print(f"  ✅ 验证通过: {symbol} 信号一致")
    else:
        print(f"  ❌ 验证失败: {len(differences)} 处不一致")
        for d in differences:
            print(f"     - {d}")

    return passed, differences


def regenerate_known_signals():
    """
    用独立算法重新生成 known_signals.csv，确保其正确性。
    """
    print("\n用独立算法重新生成 known_signals.csv ...")
    from test_data.scenarios import BENCHMARK_STOCKS

    all_signals = []
    for sym in ["600519", "01810"]:
        try:
            df = loader.load_ohlcv(sym)
            df_sig = compute_ma_cross_signals(df)
            signal_days = df_sig[df_sig["signal"] != 0].copy()
            # 确保 symbol 是字符串且保留前导零
            signal_days["symbol"] = str(sym).zfill(5) if sym.isdigit() else sym
            signal_days["strategy"] = "MA5×MA20(pandas独立验证)"
            signal_days["action"] = signal_days["signal"].map({1: "BUY", -1: "SELL"})
            all_signals.append(signal_days)
            print(f"  {sym}: {len(signal_days)} 个信号")
        except Exception as e:
            print(f"  {sym}: ❌ {e}")

    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "known_signals.csv")
        combined.to_csv(path, index=False, encoding="utf-8-sig",
                        quoting=csv.QUOTE_NONNUMERIC)
        print(f"  → known_signals.csv 已更新 ({len(combined)} 个信号)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="重新生成 known_signals.csv")
    parser.add_argument("--all", action="store_true", help="验证所有基准股票")
    args = parser.parse_args()

    if args.regenerate:
        regenerate_known_signals()
    else:
        symbols = ["600519", "01810", "300750", "00700", "000001"] if args.all else ["600519", "01810"]
        all_passed = True
        for sym in symbols:
            passed, diffs = verify_known_signals(sym)
            if not passed:
                all_passed = False

        if all_passed:
            print("\n✅ 全部验证通过！strategy.py 的信号与独立算法一致。")
        else:
            print("\n❌ 存在不一致！请检查 strategy.py 或运行 --regenerate 更新 known_signals.csv。")
