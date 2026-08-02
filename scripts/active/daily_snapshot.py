"""
daily_snapshot.py — 每日快照数据积累

每个交易日收盘后运行，缓存当日:
  1. 资金流快照 (Sina, 5个周期)
  2. 分析师一致预期 (EM, 全市场)
  3. 分钟K线增量更新 (Sina, 5分钟线)

用法:
  py scripts/daily_snapshot.py

建议通过 scheduler.py 或系统定时任务在每交易日 16:30 后运行。
"""
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def is_trading_day():
    """简单判断是否为交易日 (周一~周五)。"""
    today = datetime.now()
    if today.weekday() >= 5:
        return False
    # TODO: 接入 data/calendar.py 做精确判断
    return True


def cache_flow_snapshots():
    """缓存资金流快照。"""
    from flow_fetcher import cache_all_periods
    print(f"[{datetime.now():%H:%M:%S}] 缓存资金流快照...")
    try:
        cache_all_periods()
        print("  ✓ 资金流快照完成")
    except Exception as e:
        print(f"  ✗ 资金流快照失败: {e}")


def cache_analyst_consensus():
    """缓存分析师一致预期。"""
    from smart_money_fetcher import fetch_analyst_consensus
    print(f"[{datetime.now():%H:%M:%S}] 缓存分析师预测...")
    try:
        df = fetch_analyst_consensus()
        print(f"  ✓ 分析师预测完成: {len(df)} 只股票")
    except Exception as e:
        print(f"  ✗ 分析师预测失败: {e}")


def update_minute_data():
    """增量更新分钟K线数据 (只拉取有新数据的股票)。"""
    from data.minute_fetcher import MinuteFetcher
    print(f"[{datetime.now():%H:%M:%S}] 更新分钟K线...")
    try:
        mf = MinuteFetcher(period="5", cache_days=60)
        # 获取日线缓存中的股票列表
        cache_dir = os.path.join(BASE_DIR, "data_cache")
        if not os.path.exists(cache_dir):
            print("  ✗ 无日线缓存目录")
            return
        symbols = [f.replace(".parquet", "") for f in os.listdir(cache_dir)
                   if f.endswith(".parquet") and not f.startswith("index_")]
        # 只更新有新数据的 (fetch 内部会检查缓存是否够新)
        result = mf.fetch_batch(symbols[:500], days=5)  # 每日只拉最近5天增量
        print(f"  ✓ 分钟K线更新: {len(result)}/{len(symbols[:500])} 只")
    except Exception as e:
        print(f"  ✗ 分钟K线更新失败: {e}")


def main():
    if not is_trading_day():
        print("今日非交易日, 跳过")
        return

    print("=" * 50)
    print(f"  每日快照数据积累 — {datetime.now():%Y-%m-%d}")
    print("=" * 50)

    cache_flow_snapshots()
    time.sleep(2)
    cache_analyst_consensus()
    time.sleep(2)
    update_minute_data()

    print(f"\n[{datetime.now():%H:%M:%S}] 全部完成")


if __name__ == "__main__":
    main()
