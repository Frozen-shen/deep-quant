"""
模拟盘账户初始化脚本 — 一次性运行, 设置初始资金和起始日期

用法:
  python scripts/init_paper_account.py                    # 默认: 100000元, A股
  python scripts/init_paper_account.py --capital 500000   # 自定义资金
  python scripts/init_paper_account.py --market hk        # 港股
  python scripts/init_paper_account.py --reset            # 重置所有状态
"""

import os
import sys
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import storage


def init_account(capital: float = 100_000, market: str = "a",
                 start_date: str = None, reset: bool = False):
    """
    初始化模拟盘账户。

    Args:
      capital: 初始资金
      market: 市场 ('a' 或 'hk')
      start_date: 起始日期 YYYY-MM-DD (默认今天)
      reset: 是否清空所有历史数据
    """
    storage.init_db()

    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    print(f"{'='*50}")
    print(f"  模拟盘账户初始化")
    print(f"{'='*50}")
    print(f"  市场:     {market} ({'A股' if market == 'a' else '港股'})")
    print(f"  初始资金: ¥{capital:,.0f}")
    print(f"  起始日期: {start_date}")
    print(f"  重置模式: {'是' if reset else '否'}")

    if reset:
        print(f"\n  ⚠️ 将清空所有现有数据!")
        confirm = input("  确认? (y/N): ")
        if confirm.lower() != 'y':
            print("  已取消")
            return

    # 检查是否已初始化
    existing_capital = storage.get_config("initial_capital")
    if existing_capital and not reset:
        print(f"\n  ⚠️ 账户已初始化 (初始资金: ¥{float(existing_capital):,.0f})")
        print(f"  使用 --reset 重置, 或继续使用现有账户")
        return

    # ── 清零 ──
    if reset:
        conn = storage.get_db()
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM equity_log")
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM config")
        conn.execute("DELETE FROM backtests")
        conn.commit()
        conn.close()
        print("  ✅ 已清空所有数据")

    # ── 写入初始配置 ──
    storage.set_config("initial_capital", str(capital))
    storage.set_config("market", market)
    storage.set_config("paper_start_date", start_date)
    storage.set_config("last_date", "")
    storage.set_config("circuit_breaker", "active")  # 初始状态: 正常运行

    print(f"\n  ✅ 账户初始化完成")
    print(f"  现在可以运行:")
    print(f"    python scripts/run_paper_signal.py")


def show_status():
    """显示当前账户状态。"""
    storage.init_db()
    capital = storage.get_config("initial_capital")
    market = storage.get_config("market")
    start = storage.get_config("paper_start_date")
    last = storage.get_config("last_date")

    print(f"{'='*50}")
    print(f"  模拟盘账户状态")
    print(f"{'='*50}")

    if not capital:
        print("  ❌ 未初始化, 请运行: python scripts/init_paper_account.py")
        return

    print(f"  初始资金: ¥{float(capital):,.0f}")
    print(f"  市场:     {market}")
    print(f"  起始日期: {start}")
    print(f"  最后交易日: {last or '(未开始)'}")

    # 持仓
    positions = storage.get_all_positions()
    print(f"\n  当前持仓: {len(positions)} 只")
    for p in positions:
        print(f"    {p['symbol']}: {p['qty']}股 @ ¥{p['avg_cost']:.2f}")

    # 交易统计
    trades = storage.get_trades(limit=99999)
    buys = sum(1 for t in trades if t["action"] == "BUY")
    sells = sum(1 for t in trades if t["action"] == "SELL")
    print(f"\n  历史交易: {len(trades)} 笔 (买{buys} 卖{sells})")

    # 权益
    equity = storage.get_equity_log(limit=5)
    if equity:
        print(f"\n  最近权益:")
        for e in equity[:5]:
            ret = (e["total_equity"] / float(capital) - 1) * 100
            print(f"    {e['date']}: ¥{e['total_equity']:,.0f} ({ret:+.2f}%)")

    cb = storage.get_config("circuit_breaker", "active")
    print(f"\n  风控状态: {cb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模拟盘账户初始化")
    parser.add_argument("--capital", type=float, default=100_000,
                       help="初始资金 (默认100000)")
    parser.add_argument("--market", type=str, default="a",
                       help="市场 a/hk (默认a)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="起始日期 YYYY-MM-DD")
    parser.add_argument("--reset", action="store_true",
                       help="重置所有状态")
    parser.add_argument("--status", action="store_true",
                       help="仅显示当前状态")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        init_account(
            capital=args.capital,
            market=args.market,
            start_date=args.start_date,
            reset=args.reset,
        )
