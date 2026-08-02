"""
fetch_flow_data.py — 每日运行: 缓存当日资金流快照

数据源: 新浪财经资金流排名 (via akshare)
产出: data/flow/flow_{period}_{YYYYMMDD}.parquet

用法:
  py scripts/fetch_flow_data.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_fetcher import cache_all_periods, show_status


def main():
    print("=" * 50)
    print("每日资金流快照缓存")
    print("=" * 50)
    print()

    # 显示当前状态
    show_status()
    print()

    # 缓存所有周期
    print("开始获取当日快照...")
    print()
    results = cache_all_periods()

    # 汇总
    print()
    print("结果汇总:")
    for period, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {period}: {status}")

    print()
    print("完成。")


if __name__ == "__main__":
    main()
