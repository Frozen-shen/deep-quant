"""
交易日调度骨架 — 交易日感知 + 盘后自动化流程

功能:
  - is_trading_day(date) / next_trading_day / prev_trading_day
  - 每日盘后自动: 数据更新 → 校验 → 信号生成 → 模拟执行 → 快照
  - 支持 apscheduler 定时 / 手动触发 / dry-run

用法:
  python scheduler.py                  # 启动定时调度 (16:30 每个交易日)
  python scheduler.py --run-once       # 手动触发一次完整流程
  python scheduler.py --dry-run        # 预览将要执行的步骤

扩展了 data/calendar.py 的交易日历功能:
  - 添加 is_trading_day(), next_trading_day(), prev_trading_day()
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List
from dataclasses import dataclass, field

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.calendar import (
    get_trading_days, is_trading_day, next_trading_day, prev_trading_day
)
    """计算两个日期之间的交易日数量。"""
    _ensure_calendar()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return sum(1 for d in _trading_days_list if start_dt <= d <= end_dt)


@dataclass
class SchedulerRun:
    """一次调度运行的状态。"""
    date: str
    started_at: str
    finished_at: str = ""
    steps: List[dict] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def add_step(self, name: str, success: bool, detail: str = ""):
        self.steps.append({
            "name": name,
            "success": success,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })
        if not success:
            self.success = False


# ── 调度运行历史 ──
RUN_LOG = os.path.join(BASE_DIR, "data", "scheduler_runs.jsonl")


def _log_run(run: SchedulerRun):
    """记录调度运行历史。"""
    os.makedirs(os.path.dirname(RUN_LOG), exist_ok=True)
    entry = {
        "date": run.date,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "success": run.success,
        "error": run.error,
        "steps": run.steps,
    }
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ════════════════════════════════════════
#  每日执行流程
# ════════════════════════════════════════

def run_daily_pipeline(date_str: str = None, dry_run: bool = False) -> SchedulerRun:
    """
    执行每日完整流程。

    步骤:
      1. 交易日检查
      2. 增量数据更新
      3. 数据质量校验
      4. 因子重算 (触发预计算刷新)
      5. 生成交易信号
      6. 模拟执行下单
      7. 权益快照
      8. 异常告警检查

    Args:
      date_str: 交易日期 YYYY-MM-DD, None=上一个交易日
      dry_run: 仅预览, 不实际执行
    """
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    today_dt = pd.Timestamp(today)

    # 找到最近的交易日
    if not is_trading_day(today_dt):
        actual = prev_trading_day(today_dt)
        print(f"[Scheduler] {today} 非交易日, 使用上一个交易日: {actual.date()}")
        today_dt = actual
        today = str(actual.date())

    run = SchedulerRun(date=today, started_at=datetime.now().isoformat())
    print(f"\n{'='*60}")
    print(f"  📅 每日调度: {today}")
    print(f"  {'🔍 DRY-RUN 模式 (仅预览)' if dry_run else '▶️  执行模式'}")
    print(f"{'='*60}")

    def _step(name: str, fn, *args, **kwargs):
        """执行一个步骤并记录结果。"""
        if dry_run:
            print(f"  [DRY-RUN] {name}: 将执行 {fn.__name__ if hasattr(fn, '__name__') else str(fn)}")
            run.add_step(name, None, "dry-run skipped")
            return True
        try:
            print(f"  ▶ {name}...", end=" ", flush=True)
            result = fn(*args, **kwargs)
            print("✅")
            run.add_step(name, True, str(result)[:200] if result else "")
            return result
        except Exception as e:
            print(f"❌ {e}")
            run.add_step(name, False, str(e))
            raise

    try:
        # Step 1: 交易日检查 (已在上面完成)
        run.add_step("交易日检查", True, f"{today} 是交易日")

        # Step 2: 增量数据更新
        if not dry_run:
            from scripts.update_daily_data import update_all
        _step("数据更新", lambda: None if dry_run else update_all(
            end_date=today_dt.strftime("%Y%m%d"), max_failures=5))

        # Step 3: 数据质量校验
        if not dry_run:
            from data.validator import validate_all
        _step("数据校验", lambda: None if dry_run else validate_all())

        # Step 4: 生成交易信号
        if not dry_run:
            from scripts.run_paper_signal import generate_signal
        signal = _step("信号生成", lambda: None if dry_run else generate_signal(today))

        # Step 5: 模拟执行 (Phase 2 - 待 PaperExecutor 完成后接入)
        try:
            from execution.paper_executor import PaperExecutor
            executor = PaperExecutor()
            state = executor.load_state()
            if signal and state:
                print(f"  ▶ 模拟执行...", end=" ", flush=True)
                if not dry_run:
                    report = executor.execute_orders(
                        signal.get("buy", []),
                        signal.get("sell", []),
                        today_dt,
                    )
                    executor.snapshot(today)
                    print(f"✅ 买入{len(report.buy_filled)} 卖出{len(report.sell_filled)}")
                    run.add_step("模拟执行", True,
                                f"buy:{len(report.buy_filled)} sell:{len(report.sell_filled)}")
                else:
                    print("[DRY-RUN]")
            else:
                run.add_step("模拟执行", True, "无待执行信号或状态未初始化")
        except ImportError:
            run.add_step("模拟执行", True, "PaperExecutor 尚未创建, 跳过")
        except Exception as e:
            print(f"❌ 模拟执行失败: {e}")
            run.add_step("模拟执行", False, str(e))

        # Step 6: 异常告警检查
        try:
            from alerter import check_and_alert
            _step("告警检查", lambda: None if dry_run else check_and_alert())
        except ImportError:
            run.add_step("告警检查", True, "alerter 模块尚未创建")

        print(f"\n  ✅ 每日流程完成")

    except Exception as e:
        print(f"\n  ❌ 流程中断: {e}")
        run.error = str(e)
        run.success = False

    run.finished_at = datetime.now().isoformat()
    if not dry_run:
        _log_run(run)

    return run


# ════════════════════════════════════════
#  APScheduler 定时调度
# ════════════════════════════════════════

def start_scheduler(run_time: str = "16:30"):
    """
    启动 APScheduler 定时调度。

    Args:
      run_time: 每日运行时间 HH:MM (收盘后)
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("❌ apscheduler 未安装, 请运行: pip install apscheduler")
        print("   或使用手动模式: python scheduler.py --run-once")
        return

    hour, minute = map(int, run_time.split(":"))

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_daily_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute),
        id="daily_pipeline",
        name="每日盘后自动化",
        replace_existing=True,
    )

    scheduler.start()
    print(f"⏰ 调度器已启动: 每交易日 {run_time} 执行")
    print(f"   按 Ctrl+C 停止")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n⏹️  调度器已停止")
        scheduler.shutdown()


# ════════════════════════════════════════
#  CLI
# ════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交易日调度骨架")
    parser.add_argument("--run-once", action="store_true",
                       help="手动触发一次完整流程")
    parser.add_argument("--dry-run", action="store_true",
                       help="预览模式: 打印步骤但不实际执行")
    parser.add_argument("--date", type=str, default=None,
                       help="指定交易日期 YYYY-MM-DD")
    parser.add_argument("--serve", action="store_true",
                       help="启动定时调度服务")
    parser.add_argument("--run-time", type=str, default="16:30",
                       help="每日执行时间 HH:MM (默认16:30)")

    args = parser.parse_args()

    if args.dry_run or args.run_once or args.date:
        run_daily_pipeline(date_str=args.date, dry_run=args.dry_run)
    elif args.serve:
        start_scheduler(run_time=args.run_time)
    else:
        # 默认: 显示帮助
        parser.print_help()
        print(f"\n当前时间: {datetime.now()}")
        today = pd.Timestamp.now().normalize()
        print(f"今天是交易日: {'✅ 是' if is_trading_day(today) else '❌ 否'}")
        if not is_trading_day(today):
            nt = next_trading_day(today)
            print(f"下一个交易日: {nt.date()}")
