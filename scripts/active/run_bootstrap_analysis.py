"""
scripts/active/run_bootstrap_analysis.py — Bootstrap 置信区间分析

对回测结果的日度超额收益做 Bootstrap resampling,
计算年化超额、IR、MaxDD 的置信区间。

用法:
  py scripts/active/run_bootstrap_analysis.py
  py scripts/active/run_bootstrap_analysis.py --n-boot 20000
  py scripts/active/run_bootstrap_analysis.py --input data/ic_validation/corrected_backtest.json

输出:
  data/ic_validation/bootstrap_results.json

方法论:
  - 使用 circular block bootstrap (保留自相关结构)
  - Block length = 20 (约1个月, 与调仓周期一致)
  - 10000 次 resampling
  - 报告 90%/95%/99% CI
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

IC_DIR = os.path.join(BASE_DIR, "data", "ic_validation")
DEFAULT_INPUT = os.path.join(IC_DIR, "corrected_backtest.json")
OUTPUT_PATH = os.path.join(IC_DIR, "bootstrap_results.json")


def circular_block_bootstrap(data, block_length, n_boot, rng):
    """Circular block bootstrap — 保留序列自相关。

    Args:
        data: 1D array of daily returns
        block_length: 每个 block 的长度
        n_boot: bootstrap 次数
        rng: numpy random generator

    Returns:
        (n_boot, len(data)) array of resampled returns
    """
    n = len(data)
    # 将数据首尾相连 (circular)
    extended = np.tile(data, 2)
    samples = np.empty((n_boot, n))

    for i in range(n_boot):
        # 随机选择 block 起始点
        starts = rng.integers(0, n, size=(n // block_length + 1))
        blocks = [extended[s:s + block_length] for s in starts]
        sample = np.concatenate(blocks)[:n]
        samples[i] = sample

    return samples


def compute_metrics(daily_returns, bench_daily=None):
    """从日度收益计算关键指标。"""
    n = len(daily_returns)
    n_years = n / 252.0

    # 年化收益
    total_ret = np.prod(1 + daily_returns) - 1
    annual_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    # Sharpe
    rf_daily = 0.025 / 252
    excess = daily_returns - rf_daily
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0

    # MaxDD
    equity = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # IR (如果有基准)
    ir = 0.0
    if bench_daily is not None and len(bench_daily) >= n:
        active = daily_returns - bench_daily[:n]
        ir = np.mean(active) / np.std(active) * np.sqrt(252) if np.std(active) > 0 else 0

    return {
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "ir": ir,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 置信区间分析")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="回测结果 JSON")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap 次数")
    parser.add_argument("--block-length", type=int, default=20, help="Block 长度 (天)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("  Bootstrap 置信区间分析")
    print(f"  输入: {args.input}")
    print(f"  次数: {args.n_boot}, Block: {args.block_length} 天")
    print("=" * 60)

    # 加载回测结果
    with open(args.input, "r", encoding="utf-8") as f:
        bt_results = json.load(f)

    rng = np.random.default_rng(args.seed)
    all_bootstrap = {}

    for period_key, period_data in bt_results.get("results", {}).items():
        daily_rets = np.array(period_data.get("daily_returns", []))
        active_rets = np.array(period_data.get("daily_active_returns", []))

        if len(daily_rets) < 30:
            print(f"\n  [{period_key}] 日度数据不足 ({len(daily_rets)}), 跳过")
            continue

        print(f"\n  [{period_key}] N={len(daily_rets)} 天")
        print(f"    原始: 年化={period_data['annual_return']:+.1f}%, "
              f"IR={period_data['ir']:.2f}, MaxDD={period_data['max_drawdown']:.1f}%")

        # Bootstrap 日度收益
        boot_samples = circular_block_bootstrap(
            daily_rets, args.block_length, args.n_boot, rng)

        # 计算每次 bootstrap 的指标
        boot_annual = np.empty(args.n_boot)
        boot_sharpe = np.empty(args.n_boot)
        boot_maxdd = np.empty(args.n_boot)
        boot_ir = np.empty(args.n_boot)

        # 如果有 active returns, 也 bootstrap
        if len(active_rets) >= 30:
            boot_active = circular_block_bootstrap(
                active_rets, args.block_length, args.n_boot, rng)
        else:
            boot_active = None

        for i in range(args.n_boot):
            m = compute_metrics(boot_samples[i])
            boot_annual[i] = m["annual_return"]
            boot_sharpe[i] = m["sharpe"]
            boot_maxdd[i] = m["max_drawdown"]

            if boot_active is not None:
                active_i = boot_active[i]
                ir_i = (np.mean(active_i) / np.std(active_i) * np.sqrt(252)
                        if np.std(active_i) > 0 else 0)
                boot_ir[i] = ir_i

        # 置信区间
        def ci(arr, levels=[0.01, 0.05, 0.10]):
            result = {}
            for alpha in levels:
                lo = np.percentile(arr, alpha / 2 * 100)
                hi = np.percentile(arr, (1 - alpha / 2) * 100)
                result[f"{int((1-alpha)*100)}%_CI"] = [round(float(lo), 4), round(float(hi), 4)]
            return result

        annual_ci = ci(boot_annual * 100)  # 转为百分比
        sharpe_ci = ci(boot_sharpe)
        maxdd_ci = ci(boot_maxdd * 100)
        ir_ci = ci(boot_ir) if len(active_rets) >= 30 else {}

        # 关键判断: 95% CI 是否排除 0
        annual_excludes_zero = annual_ci["95%_CI"][0] > 0
        ir_excludes_zero = ir_ci.get("95%_CI", [0, 0])[0] > 0 if ir_ci else False

        print(f"    Bootstrap 年化收益 95% CI: [{annual_ci['95%_CI'][0]:+.1f}%, {annual_ci['95%_CI'][1]:+.1f}%]")
        print(f"    Bootstrap IR 95% CI: [{ir_ci.get('95%_CI', [0,0])[0]:.2f}, {ir_ci.get('95%_CI', [0,0])[1]:.2f}]")
        print(f"    Bootstrap MaxDD 95% CI: [{maxdd_ci['95%_CI'][0]:.1f}%, {maxdd_ci['95%_CI'][1]:.1f}%]")
        print(f"    年化>0 (95%): {'✅ YES' if annual_excludes_zero else '❌ NO'}")
        print(f"    IR>0 (95%):   {'✅ YES' if ir_excludes_zero else '❌ NO'}")

        all_bootstrap[period_key] = {
            "n_days": len(daily_rets),
            "n_bootstrap": args.n_boot,
            "block_length": args.block_length,
            "original": {
                "annual_return_pct": period_data["annual_return"],
                "ir": period_data["ir"],
                "max_drawdown_pct": period_data["max_drawdown"],
            },
            "bootstrap_annual_return_pct": annual_ci,
            "bootstrap_sharpe": sharpe_ci,
            "bootstrap_max_drawdown_pct": maxdd_ci,
            "bootstrap_ir": ir_ci,
            "annual_excludes_zero_95pct": annual_excludes_zero,
            "ir_excludes_zero_95pct": ir_excludes_zero,
        }

    # 保存
    output = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Circular Block Bootstrap 置信区间分析",
            "n_bootstrap": args.n_boot,
            "block_length": args.block_length,
            "seed": args.seed,
            "input_file": args.input,
        },
        "results": all_bootstrap,
    }

    os.makedirs(IC_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  结果: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()