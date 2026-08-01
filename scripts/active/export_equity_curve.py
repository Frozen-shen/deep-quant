"""
净值曲线导出与可视化 — 从 equity_log 读取数据并生成图表

用法:
  python scripts/export_equity_curve.py                    # 导出 CSV + PNG
  python scripts/export_equity_curve.py --benchmark 000300 # 与沪深300对比
"""

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import storage

OUTPUT_DIR = os.path.join(BASE_DIR, "data")


def load_benchmark_data(benchmark: str = "000300") -> pd.DataFrame:
    """
    加载基准指数数据 (从 parquet 缓存或 akshare 拉取)。

    Args:
      benchmark: 指数代码 (000300=沪深300, 000905=中证500, 000852=中证1000)

    Returns:
      DataFrame with columns [date, close]
    """
    # 尝试从 parquet 缓存加载
    cache_path = os.path.join(BASE_DIR, "data_cache", f"{benchmark}.parquet")
    if not os.path.exists(cache_path):
        # 尝试从 akshare 拉取
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol=f"sh{benchmark}")
            df = df.rename(columns={"date": "date"})
            if "close" in df.columns:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_parquet(cache_path, index=False)
                print(f"  基准指数已缓存: {cache_path}")
            return df
        except Exception as e:
            print(f"  无法加载基准指数 {benchmark}: {e}")
            return None

    df = pd.read_parquet(cache_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def export_equity_curve(benchmark: str = None,
                        output_csv: str = None,
                        output_png: str = None) -> dict:
    """
    导出权益曲线数据。

    Returns:
      {"csv_path": str, "stats": dict}
    """
    storage.init_db()
    initial = float(storage.get_config("initial_capital", "100000"))

    # ── 读取权益日志 ──
    rows = storage.get_equity_log(limit=99999)
    if not rows:
        print("⚠️ 无权益日志数据, 请先运行模拟盘")
        return {"csv_path": None, "stats": {}}

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 计算累计收益率
    df["cumulative_return"] = df["total_equity"] / initial - 1

    # ── 汇总统计 ──
    stats = {
        "start_date": str(df["date"].iloc[0].date()),
        "end_date": str(df["date"].iloc[-1].date()),
        "trading_days": len(df),
        "initial_capital": initial,
        "final_equity": round(float(df["total_equity"].iloc[-1]), 2),
        "total_return_pct": round(float(df["cumulative_return"].iloc[-1]) * 100, 2),
        "max_equity": round(float(df["total_equity"].max()), 2),
        "min_equity": round(float(df["total_equity"].min()), 2),
    }

    # 计算回撤
    peak = df["total_equity"].cummax()
    drawdown = (df["total_equity"] - peak) / peak
    stats["max_drawdown_pct"] = round(float(drawdown.min()) * 100, 2)

    # 计算日收益率统计
    if "daily_return" in df.columns:
        daily_rets = df["daily_return"].dropna()
        if len(daily_rets) > 0:
            stats["daily_return_mean"] = round(float(daily_rets.mean()) * 100, 4)
            stats["daily_return_std"] = round(float(daily_rets.std()) * 100, 4)
            stats["daily_sharpe"] = round(
                float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)), 2
            ) if daily_rets.std() > 0 else 0
            stats["daily_win_rate"] = round(
                float((daily_rets > 0).sum() / len(daily_rets)), 4
            )

    # ── 写入 CSV ──
    if output_csv is None:
        output_csv = os.path.join(OUTPUT_DIR, "paper_equity.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  权益 CSV: {output_csv}")

    # ── 基准对比 ──
    if benchmark:
        bench_df = load_benchmark_data(benchmark)
        if bench_df is not None and "date" in bench_df.columns:
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            # 对齐到策略起始日期
            start_date = df["date"].iloc[0]
            bench_df = bench_df[bench_df["date"] >= start_date]
            if len(bench_df) > 0:
                bench_start_price = bench_df["close"].iloc[0]
                bench_df["benchmark_return"] = bench_df["close"] / bench_start_price - 1
                df = df.merge(
                    bench_df[["date", "benchmark_return"]],
                    on="date", how="left"
                )
                df["benchmark_return"] = df["benchmark_return"].ffill()
                stats["benchmark_return_pct"] = round(
                    float(df["benchmark_return"].iloc[-1]) * 100, 2
                )
                stats["excess_return_pct"] = round(
                    stats["total_return_pct"] - stats["benchmark_return_pct"], 2
                )
                # 更新 CSV 加入基准列
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ── 生成图表 ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={'height_ratios': [2, 1]})

        # 上图: 权益曲线
        ax1.plot(df["date"], df["total_equity"] / initial * 100,
                label=f"策略 (累计收益 {stats['total_return_pct']:+.1f}%)",
                color="#2196F3", linewidth=2)
        if benchmark and "benchmark_return" in df.columns:
            ax1.plot(df["date"], (df["benchmark_return"] + 1) * 100,
                    label=f"基准 {benchmark} ({stats.get('benchmark_return_pct', 0):+.1f}%)",
                    color="#9E9E9E", linewidth=1.5, linestyle="--")
        ax1.axhline(y=100, color="gray", linestyle=":", alpha=0.5)
        ax1.set_ylabel("净值 (初始=100)")
        ax1.set_title(f"模拟盘权益曲线 ({stats['start_date']} ~ {stats['end_date']})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 下图: 回撤曲线
        ax2.fill_between(df["date"], 0, drawdown * 100,
                         color="#F44336", alpha=0.5, label="回撤")
        ax2.axhline(y=-5, color="orange", linestyle="--", alpha=0.5, label="-5%")
        ax2.axhline(y=-10, color="red", linestyle="--", alpha=0.5, label="-10%")
        ax2.set_ylabel("回撤 (%)")
        ax2.set_xlabel("日期")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_png is None:
            output_png = os.path.join(OUTPUT_DIR, "paper_equity.png")
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  权益图表: {output_png}")

    except ImportError:
        print("  ⚠️ matplotlib 未安装, 跳过图表生成")

    # ── 打印统计 ──
    print(f"\n  {'='*40}")
    print(f"  📊 模拟盘统计")
    print(f"  {'='*40}")
    print(f"  交易天数:   {stats['trading_days']}")
    print(f"  初始资金:   ¥{stats['initial_capital']:,.0f}")
    print(f"  最终权益:   ¥{stats['final_equity']:,.0f}")
    print(f"  总收益率:   {stats['total_return_pct']:+.2f}%")
    print(f"  最大回撤:   {stats['max_drawdown_pct']:+.2f}%")
    if benchmark:
        print(f"  基准收益:   {stats.get('benchmark_return_pct', 0):+.2f}%")
        print(f"  超额收益:   {stats.get('excess_return_pct', 0):+.2f}%")
    if "daily_sharpe" in stats:
        print(f"  年化夏普:   {stats['daily_sharpe']:.2f}")
    if "daily_win_rate" in stats:
        print(f"  日胜率:     {stats['daily_win_rate']:.1%}")

    return {"csv_path": output_csv, "stats": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出权益曲线")
    parser.add_argument("--benchmark", type=str, default=None,
                       help="基准指数代码 (如 000300)")
    parser.add_argument("--output-csv", type=str, default=None,
                       help="CSV 输出路径")
    parser.add_argument("--output-png", type=str, default=None,
                       help="PNG 输出路径")
    args = parser.parse_args()

    export_equity_curve(
        benchmark=args.benchmark,
        output_csv=args.output_csv,
        output_png=args.output_png,
    )
