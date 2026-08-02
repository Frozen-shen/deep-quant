"""
批量拉取北向持仓 + 分析师预测并缓存

用法:
  py scripts/fetch_smart_money.py                  # 拉取 data_store 中所有股票
  py scripts/fetch_smart_money.py --symbols 600519,000858
  py scripts/fetch_smart_money.py --max-stocks 100 # 限制数量
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_money_fetcher import (
    batch_fetch_northbound,
    fetch_analyst_consensus,
    REQUEST_INTERVAL,
)
from data_cache import get_cached_symbols


def main():
    parser = argparse.ArgumentParser(description="聪明钱数据批量获取")
    parser.add_argument("--symbols", type=str, default="",
                        help="指定股票(逗号分隔), 默认使用 data_store 缓存")
    parser.add_argument("--max-stocks", type=int, default=500,
                        help="单次最大请求数 (默认500)")
    parser.add_argument("--skip-analyst", action="store_true",
                        help="跳过分析师预期拉取")
    args = parser.parse_args()

    # 确定股票池
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_cached_symbols()

    print(f"标的来源: {'指定' if args.symbols else 'data_store缓存'} ({len(symbols)} 只)")
    print(f"限速: {REQUEST_INTERVAL}s/请求, 上限: {args.max_stocks}")
    print()

    # 1. 拉取分析师一致预期 (一次调用, 全市场)
    if not args.skip_analyst:
        print("=" * 50)
        print("Step 1: 分析师一致预期 (全市场)")
        print("=" * 50)
        adf = fetch_analyst_consensus()
        if len(adf) > 0:
            print(f"  ✅ 获取 {len(adf)} 条分析师预测")
        else:
            print("  ⚠️ 分析师预期拉取失败或为空")
        print()

    # 2. 批量拉取北向资金持仓
    print("=" * 50)
    print("Step 2: 北向资金持仓 (逐只拉取)")
    print("=" * 50)
    results = batch_fetch_northbound(symbols, max_stocks=args.max_stocks)
    print(f"\n北向数据: {len(results)}/{len(symbols)} 只成功")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
