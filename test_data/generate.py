"""
测试数据集生成器 — 一键生成所有测试数据

用法:
    python test_data/generate.py              # 生成全部
    python test_data/generate.py --quick      # 只生成基准(5只股票)
    python test_data/generate.py --verify     # 验证已有数据

生成内容:
    datasets/ohlcv_{symbol}.csv     行情基准 (5只股票)
    datasets/ohlcv_multi.csv        多股票合并
    datasets/scenarios.csv          场景定义
    datasets/mock_events.csv        模拟事件
    datasets/known_signals.csv      已知信号(回归测试用)
    manifest.json                   数据集元信息
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import DataFetcher, MARKET_CONFIG
from strategy import MACrossoverStrategy
from test_data.scenarios import (
    BENCHMARK_STOCKS, SCENARIOS, KNOWN_RESULTS, MOCK_EVENTS
)

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")


def ensure_dir():
    os.makedirs(DATASETS_DIR, exist_ok=True)


# ================================================================
#  1. 行情基准集
# ================================================================

def generate_ohlcv_benchmark(quick: bool = False):
    """拉取基准股票的日线数据并保存为 CSV。"""
    print("\n" + "=" * 60)
    print("  [1/4] 生成行情基准集")
    print("=" * 60)

    fetcher = DataFetcher()
    stocks_to_fetch = BENCHMARK_STOCKS

    if quick:
        # 只取前2只
        stocks_to_fetch = dict(list(BENCHMARK_STOCKS.items())[:2])

    all_frames = []

    for sym, cfg in stocks_to_fetch.items():
        print(f"\n  {sym} ({cfg['name']}, {cfg['market']})...")
        try:
            df = fetcher.fetch(sym, cfg["start"], cfg["end"], "qfq", market=cfg["market"])
            df.insert(0, "symbol", sym)
            df.insert(1, "name", cfg["name"])

            # 保存单个文件
            path = os.path.join(DATASETS_DIR, f"ohlcv_{sym}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"    → ohlcv_{sym}.csv ({len(df)} 行)")

            all_frames.append(df)
        except Exception as e:
            print(f"    ❌ 失败: {e}")

    # 合并文件
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        path = os.path.join(DATASETS_DIR, "ohlcv_multi.csv")
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  → ohlcv_multi.csv ({len(combined)} 行, {combined['symbol'].nunique()} 只股票)")

    return len(all_frames)


# ================================================================
#  2. 场景数据集
# ================================================================

def generate_scenarios():
    """保存场景定义和场景对应的价格数据片段。"""
    print("\n" + "=" * 60)
    print("  [2/4] 生成场景数据集")
    print("=" * 60)

    fetcher = DataFetcher()

    # 保存场景定义
    scenarios_rows = []
    for sid, sc in SCENARIOS.items():
        sym = sc.get("symbol", sc.get("symbols", [None])[0])
        row = {
            "scenario_id": sid,
            "name": sc["name"],
            "start": sc["start"],
            "end": sc["end"],
            "primary_symbol": sym,
            "description": sc["description"],
        }
        scenarios_rows.append(row)

    df_scenarios = pd.DataFrame(scenarios_rows)
    path = os.path.join(DATASETS_DIR, "scenarios.csv")
    df_scenarios.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → scenarios.csv ({len(df_scenarios)} 个场景)")

    # 为每个场景保存对应的价格数据片段
    count = 0
    for sid, sc in SCENARIOS.items():
        syms = sc.get("symbols", [sc.get("symbol")])
        for sym in syms:
            if not sym:
                continue
            try:
                market = "hk" if sym.isdigit() and len(sym) == 5 else "a"
                df = fetcher.fetch(sym, sc["start"], sc["end"], "qfq", market=market)
                df.insert(0, "symbol", sym)
                df.insert(1, "scenario", sid)
                path = os.path.join(DATASETS_DIR, f"scenario_{sid}_{sym}.csv")
                df.to_csv(path, index=False, encoding="utf-8-sig")
                count += 1
            except Exception as e:
                print(f"    {sid}/{sym}: ❌ {e}")

    print(f"  → {count} 个场景数据文件")
    return count


# ================================================================
#  3. 已知信号集 (回归测试用)
# ================================================================

def generate_known_signals():
    """根据已知配置计算策略信号，保存为回归测试基准。"""
    print("\n" + "=" * 60)
    print("  [3/4] 生成已知信号集")
    print("=" * 60)

    fetcher = DataFetcher()
    all_signals = []

    for key, cfg in KNOWN_RESULTS.items():
        sym = cfg["symbol"]
        market = cfg["market"]
        print(f"  {key} ...")

        try:
            df = fetcher.fetch(sym, cfg["start"], cfg["end"], "qfq", market=market)
            strategy = MACrossoverStrategy(short_window=5, long_window=20)
            df = strategy.generate_signals(df)

            # 提取所有信号日
            signal_days = df[df["signal"] != 0][["date", "signal", "close"]].copy()
            signal_days["symbol"] = sym
            signal_days["strategy"] = cfg["strategy"]
            signal_days["action"] = signal_days["signal"].map({1: "BUY", -1: "SELL"})

            # 统计
            buy_count = (signal_days["signal"] == 1).sum()
            sell_count = (signal_days["signal"] == -1).sum()
            print(f"    BUY:{buy_count} SELL:{sell_count}")

            all_signals.append(signal_days)
        except Exception as e:
            print(f"    ❌ {e}")

    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        path = os.path.join(DATASETS_DIR, "known_signals.csv")
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  → known_signals.csv ({len(combined)} 个信号)")

    return len(all_signals)


# ================================================================
#  4. 模拟事件
# ================================================================

def generate_mock_events():
    """保存模拟公司事件数据。"""
    print("\n" + "=" * 60)
    print("  [4/4] 生成模拟事件")
    print("=" * 60)

    df = pd.DataFrame(MOCK_EVENTS)
    path = os.path.join(DATASETS_DIR, "mock_events.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  → mock_events.csv ({len(df)} 条事件)")

    return len(df)


# ================================================================
#  Manifest
# ================================================================

def write_manifest(datasets_info: Dict):
    """写入数据集清单。"""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "generator_version": "1.0",
        "datasets": datasets_info,
        "stocks": {sym: {"name": cfg["name"], "market": cfg["market"],
                         "sector": cfg["sector"]}
                   for sym, cfg in BENCHMARK_STOCKS.items()},
        "usage": {
            "回测测试": "python -c 'from test_data import loader; df = loader.load_ohlcv(\"600519\")'",
            "场景测试": "python -c 'from test_data import loader; df = loader.load_scenario(\"bear_2018\")'",
            "信号验证": "python -c 'from test_data import loader; df = loader.load_known_signals()'",
            "LLM测试": "python -c 'from test_data import loader; df = loader.load_mock_events()'",
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  → manifest.json")


# ================================================================
#  入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="生成量化测试数据集")
    parser.add_argument("--quick", action="store_true", help="只生成基准集(2只股票)")
    parser.add_argument("--verify", action="store_true", help="验证已有数据集")
    args = parser.parse_args()

    if args.verify:
        verify_datasets()
        return

    ensure_dir()
    datasets = {}

    t0 = time.time()

    n1 = generate_ohlcv_benchmark(quick=args.quick)
    datasets["ohlcv"] = {"files": n1, "path": "datasets/ohlcv_*.csv"}

    if not args.quick:
        n2 = generate_scenarios()
        datasets["scenarios"] = {"files": n2, "path": "datasets/scenario_*.csv"}

        n3 = generate_known_signals()
        datasets["known_signals"] = {"signals": n3, "path": "datasets/known_signals.csv"}

        n4 = generate_mock_events()
        datasets["mock_events"] = {"count": n4, "path": "datasets/mock_events.csv"}

    write_manifest(datasets)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅ 测试数据集生成完成 (耗时 {elapsed:.0f}s)")
    print(f"  目录: {DATASETS_DIR}")
    print(f"{'='*60}")


def verify_datasets():
    """验证已有数据集。"""
    print("\n验证测试数据集...\n")
    errors = []

    # 检查关键文件
    required = ["ohlcv_600519.csv", "ohlcv_01810.csv", "known_signals.csv",
                "mock_events.csv", "manifest.json"]
    for f in required:
        path = os.path.join(DATASETS_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ {f} 缺失")
            errors.append(f)

    if errors:
        print(f"\n  {len(errors)} 个文件缺失，请运行: python test_data/generate.py")
    else:
        # 验证数据内容
        df = pd.read_csv(os.path.join(DATASETS_DIR, "ohlcv_600519.csv"))
        print(f"\n  茅台数据: {len(df)} 行, {df['date'].min()} ~ {df['date'].max()}")
        assert len(df) > 1000, "数据量不足"
        assert "close" in df.columns, "缺少close列"
        assert "volume" in df.columns, "缺少volume列"

        df_sig = pd.read_csv(os.path.join(DATASETS_DIR, "known_signals.csv"))
        buys = (df_sig["signal"] == 1).sum()
        sells = (df_sig["signal"] == -1).sum()
        print(f"  已知信号: {buys} BUY + {sells} SELL")

        print(f"\n  ✅ 验证通过!")


if __name__ == "__main__":
    main()
