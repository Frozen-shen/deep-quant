"""
daily_pipeline.py — 每日盘后自动化管道

整合数据更新 → 质量校验 → 信号生成 → 执行记录，
使用 gate.py 门禁 + logger.py 日志 + experiment_tracker.py 追踪。

用法:
  py scripts/daily_pipeline.py                # 最近交易日全流程
  py scripts/daily_pipeline.py --date 2026-08-01
  py scripts/daily_pipeline.py --dry-run      # 预览步骤
  py scripts/daily_pipeline.py --skip-fetch   # 跳过数据拉取 (已有数据时)
  py scripts/daily_pipeline.py --signal-only  # 仅生成信号 (不拉数据)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "active"))

import storage  # noqa: E402  (持仓/成交, 用于分钟增量标的集合)
from gate import load_config, GateViolation
from logger import get_logger
from experiment_tracker import log_experiment
from alerter import AlertManager, check_and_alert

log = get_logger("daily_pipeline")
_alert_manager = AlertManager()


# ═══════════════════════════════════════════════════════════
#  步骤结果
# ═══════════════════════════════════════════════════════════

@dataclass
class StepResult:
    name: str
    success: bool
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class PipelineRun:
    date: str
    started_at: str
    finished_at: str = ""
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def add(self, name: str, success: bool, detail: str = "", duration: float = 0.0):
        self.steps.append(StepResult(name, success, detail, duration))
        if not success:
            self.success = False

    def summary(self) -> dict:
        return {
            "date": self.date,
            "success": self.success,
            "steps_total": len(self.steps),
            "steps_ok": sum(1 for s in self.steps if s.success),
            "steps_fail": sum(1 for s in self.steps if not s.success),
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════
#  步骤实现
# ═══════════════════════════════════════════════════════════

def step_trading_day_check(date_str: str) -> str:
    """检查是否为交易日，返回实际使用的交易日。"""
    import pandas as pd
    from data.calendar import is_trading_day, prev_trading_day

    dt = pd.Timestamp(date_str)
    if is_trading_day(dt):
        return date_str

    actual = prev_trading_day(dt)
    log.info("%s 非交易日, 回退到 %s", date_str, actual.date())
    return str(actual.date())


def _minute_symbols(date_str: str) -> list:
    """分钟增量更新标的: 持仓 ∪ 近期成交 ∪ 上次执行报告买卖信号。"""
    syms = set()
    try:
        for p in storage.get_all_positions():
            if p.get("symbol"):
                syms.add(p["symbol"])
    except Exception:
        pass
    try:
        for t in storage.get_trades(limit=2000):
            if t.get("symbol"):
                syms.add(t["symbol"])
    except Exception:
        pass
    report_path = os.path.join(BASE_DIR, "data", "paper_executions.jsonl")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                last = f.readlines()[-1].strip()
            if last:
                rep = json.loads(last)
                for key in ("buy_signals", "sell_signals"):
                    for s in rep.get(key, []):
                        if isinstance(s, str) and s:
                            syms.add(s)
        except Exception:
            pass
    return sorted(syms)


def _minute_refresh(date_str: str, since: int = 5, timeout_s: int = 1800) -> str:
    """管线分钟增量: 只补交易标的 (东财 em 源, 走 eastmoney_proxy 网关)。

    5m + 15m 两档, --since 增量合并去重。失败不阻塞管线 (返回文案说明),
    分钟数据陈旧度由信号步骤守卫告警兜底; 子进程超时会杀掉但已落盘部分保留。
    """
    syms = _minute_symbols(date_str)
    if not syms:
        return "分钟增量: 无交易标的, 跳过"
    script = os.path.join(BASE_DIR, "scripts", "active",
                          "fetch_baostock_minute.py")
    try:
        for freq in ("5", "15"):
            subprocess.run(
                [sys.executable, script, "--freq", freq,
                 "--since", str(since), "--symbols", ",".join(syms)],
                cwd=BASE_DIR, timeout=timeout_s, check=False)
        return f"分钟增量: {len(syms)} 只 (5m/15m, 最近 {since} 交易日) 完成"
    except subprocess.TimeoutExpired:
        return f"分钟增量: 超时 (>{timeout_s}s), 已落盘部分保留, 下次继续补"
    except Exception as e:
        return f"分钟增量失败: {e} (不阻塞, 信号步骤有陈旧度告警兜底)"


def step_data_fetch(date_str: str, resume: bool = True) -> str:
    """增量数据更新 — 日线 (东财 via eastmoney_proxy, 腾讯兜底) + 分钟增量。"""
    import pandas as pd

    from update_daily_data import run as update_run

    end_date = pd.Timestamp(date_str).strftime("%Y%m%d")
    meta = update_run(
        start_date="20180101",
        end_date=end_date,
        force=False,
        resume=resume,
        check_only_mode=False,
        limit=None,
    )

    if meta is None:
        daily_note = "fetch returned None (可能无新数据)"
    else:
        stats = meta.get("stats", {})
        count = meta.get("count", 0)
        failed = stats.get("failed", [])
        daily_note = f"更新 {count} 只, 失败 {len(failed)} 只"

    # 分钟增量 (交易标的): 失败不阻塞, 见 _minute_refresh
    minute_note = _minute_refresh(date_str)
    return f"{daily_note}; {minute_note}"


def step_data_quality(date_str: str) -> str:
    """基础数据质量校验 — 日线陈旧检查 + 各数据目录最新日期同步性检查。"""
    import pandas as pd
    import glob

    data_store = os.path.join(BASE_DIR, "data_store")
    files = glob.glob(os.path.join(data_store, "*.parquet"))

    if not files:
        raise RuntimeError("data_store/ 为空, 无数据可校验")

    target_date = pd.Timestamp(date_str)
    stale_count = 0
    empty_count = 0
    sample_size = min(50, len(files))  # 抽样检查

    import random
    sampled = random.sample(files, sample_size)

    for f in sampled:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                empty_count += 1
                continue
            df["date"] = pd.to_datetime(df["date"])
            max_date = df["date"].max()
            # 允许最多 5 天落后 (周末+节假日)
            if (target_date - max_date).days > 7:
                stale_count += 1
        except Exception:
            empty_count += 1

    detail = f"抽样 {sample_size} 只: 过期 {stale_count}, 空/异常 {empty_count}"
    if stale_count > sample_size * 0.3:
        raise RuntimeError(f"数据过期率过高: {stale_count}/{sample_size}")

    # ── 各数据目录最新日期同步性检查 (2026-08-09 新增, 防"静默滞后") ──
    # 背景: update_daily_data len(f)==11 bug + 腾讯源被封曾导致日线/指数静默滞后 1 周。
    sync_checks = [
        ("日线主库", os.path.join(data_store, "*.parquet"), "date", None),
        ("两融 aux_margin", os.path.join(data_store, "aux_margin", "*.parquet"), "file", "date"),
        ("龙虎榜 aux_lhb", os.path.join(data_store, "aux_lhb", "*.parquet"), "file", "date"),
        ("大宗交易 aux_dzjy", os.path.join(data_store, "aux_dzjy", "*.parquet"), "file", "date"),
        ("分钟15m", os.path.join(data_store, "minute_15m", "*.parquet"), "datetime", None),
        ("分钟5m", os.path.join(data_store, "minute_5m", "*.parquet"), "datetime", None),
        ("指数基准", os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet"), "date", None),
    ]
    sync_msgs = []
    for name, pattern, col, file_col in sync_checks:
        try:
            fs = glob.glob(pattern)
            if not fs:
                sync_msgs.append(f"{name}: 无文件")
                continue
            if col == "file":
                # 按日文件: 从文件名取日期
                latest = max(pd.to_datetime(
                    os.path.basename(f).replace(".parquet", "")) for f in fs)
            else:
                latest = pd.to_datetime(pd.read_parquet(fs[0])[col]).max()
            lag = (target_date - latest).days
            mark = "✓" if lag <= 7 else "⚠ 滞后"
            sync_msgs.append(f"{name}: {latest.date()} [{mark} 差{lag}天]")
        except Exception as e:
            sync_msgs.append(f"{name}: 检查失败({str(e)[:30]})")
    sync_str = " | ".join(sync_msgs)
    log.info("数据同步检查: %s", sync_str)

    # 任一日线关键数据滞后 > 7 天视为质量异常 (指数/aux 滞后只记录不中断)
    daily_latest = None
    try:
        fs = glob.glob(os.path.join(data_store, "*.parquet"))
        daily_latest = max(pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).max()
                           for f in fs[:50])
    except Exception:
        pass
    if daily_latest is not None and (target_date - daily_latest).days > 7:
        raise RuntimeError(
            f"日线数据滞后 {target_date.date()} vs {daily_latest.date()}, 请检查数据源/代理")

    return detail + " | " + sync_str


def step_signal_generation(date_str: str, dry_run: bool = False) -> str:
    """生成交易信号 — 调用 run_paper_signal 的核心逻辑。"""
    from run_paper_signal import generate_signal_v3

    result = generate_signal_v3(date_str=date_str, dry_run=dry_run)
    if result is None:
        return "无信号生成 (可能因子配置缺失)"

    buy_count = len(result.get("buy", []))
    sell_count = len(result.get("sell", []))
    return f"买入 {buy_count} 只, 卖出 {sell_count} 只"


def step_ic_monitor() -> str:
    """IC 衰减监控 (每周运行一次即可, 这里仅检查)。"""
    try:
        from run_ic_monitor import run_ic_monitor
        result = run_ic_monitor(lookback=60)
        decayed = result.get("decayed_factors", [])
        if decayed:
            log.warning("IC 衰减因子: %s", decayed)
            return f"衰减因子: {decayed}"
        return "IC 正常"
    except ImportError:
        return "run_ic_monitor 不可用, 跳过"
    except Exception as e:
        return f"IC 监控异常: {e}"


def step_circuit_breaker() -> str:
    """检查回撤熔断状态。"""
    try:
        from execution.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        state = cb.check()
        report = cb.get_status_report()
        dd = report.get("drawdown_pct", 0)
        if state in ("halted", "halted_manual"):
            log.warning("熔断状态: %s (回撤 %.2f%%)", state, dd)
            _alert_manager.send("circuit_breaker", {
                "state": state,
                "drawdown": dd,
                "equity": report.get("current_equity", 0),
            }, level="ERROR")
            return f"熔断: {state} (回撤 {dd:.2f}%)"
        elif state == "warning":
            log.warning("回撤警告: %.2f%%", dd)
            return f"警告: 回撤 {dd:.2f}%"
        return f"正常 (回撤 {dd:.2f}%)"
    except ImportError:
        return "circuit_breaker 不可用, 跳过"
    except Exception as e:
        return f"熔断检查异常: {e}"


# ═══════════════════════════════════════════════════════════
#  主管道
# ═══════════════════════════════════════════════════════════

def run_pipeline(
    date_str: Optional[str] = None,
    dry_run: bool = False,
    skip_fetch: bool = False,
    signal_only: bool = False,
) -> PipelineRun:
    """
    执行每日完整管道。

    Args:
        date_str: 交易日期 YYYY-MM-DD, None=今天
        dry_run: 仅预览步骤
        skip_fetch: 跳过数据拉取
        signal_only: 仅信号生成 (隐含 skip_fetch)
    """
    # ── Gate 门禁 ──
    # daily_pipeline 是生产脚本 (paper trading), 允许在 blind 期运行。
    # gate.check 仅拦截研究脚本对 blind 数据的回测, 不拦截生产信号。
    # 这里仅验证 config 完整性, 不做 partition 拦截。
    config = load_config(os.path.join(BASE_DIR, "config.yaml"))

    today = date_str or datetime.now().strftime("%Y-%m-%d")
    run = PipelineRun(date=today, started_at=datetime.now().isoformat())

    log.info("=" * 60)
    log.info("每日管道启动: %s %s", today, "[DRY-RUN]" if dry_run else "")
    log.info("=" * 60)

    def _run_step(name: str, fn, *args, **kwargs) -> Optional[object]:
        """执行步骤并记录。"""
        if dry_run:
            log.info("  [DRY-RUN] %s", name)
            run.add(name, True, "dry-run")
            return None

        t0 = time.time()
        try:
            log.info("  ▶ %s ...", name)
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            detail = str(result) if result else ""
            log.info("  ✅ %s (%.1fs) %s", name, elapsed, detail)
            run.add(name, True, detail, elapsed)
            return result
        except Exception as e:
            elapsed = time.time() - t0
            log.error("  ❌ %s (%.1fs): %s", name, elapsed, e)
            run.add(name, False, str(e), elapsed)
            # 关键步骤失败时推送告警
            if name in ("数据更新", "信号生成"):
                try:
                    _alert_manager.send("data_fetch_failed" if name == "数据更新"
                                       else "signal_empty",
                                       {"step": name, "error": str(e),
                                        "date": today},
                                       level="ERROR")
                except Exception:
                    pass
            raise

    try:
        # Step 1: 交易日检查
        actual_date = _run_step("交易日检查", step_trading_day_check, today)
        if actual_date:
            run.date = actual_date
            today = actual_date

        if signal_only:
            # 仅信号模式
            _run_step("信号生成", step_signal_generation, today, dry_run)
        else:
            # Step 2: 数据更新
            if not skip_fetch:
                _run_step("数据更新", step_data_fetch, today)
            else:
                log.info("  ⏭ 跳过数据拉取 (--skip-fetch)")
                run.add("数据更新", True, "skipped")

            # Step 3: 数据质量
            _run_step("数据质量校验", step_data_quality, today)

            # Step 4: 信号生成
            _run_step("信号生成", step_signal_generation, today, dry_run)

            # Step 5: IC 监控 (非关键, 失败不中断)
            try:
                _run_step("IC监控", step_ic_monitor)
            except Exception:
                pass  # 非关键步骤

            # Step 6: 回撤熔断检查 (非关键, 失败不中断)
            try:
                _run_step("熔断检查", step_circuit_breaker)
            except Exception:
                pass  # 非关键步骤

        log.info("管道完成: %s", "✅" if run.success else "❌")

    except Exception as e:
        run.error = str(e)
        run.success = False
        log.error("管道中断: %s", e)

    run.finished_at = datetime.now().isoformat()

    # 记录实验
    if not dry_run:
        try:
            log_experiment(
                script_name="daily_pipeline",
                partition="blind",
                config={"date": today, "skip_fetch": skip_fetch},
                results=run.summary(),
                notes="每日管道自动运行",
                experiments_dir=os.path.join(BASE_DIR, "experiments"),
            )
        except Exception as e:
            log.warning("实验记录失败: %s", e)

        # 最终告警扫描 (检查数据/信号/净值/熔断/IC 各项异常)
        try:
            check_and_alert()
        except Exception as e:
            log.warning("告警扫描失败: %s", e)

    return run


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="每日盘后自动化管道 (数据→校验→信号→记录)")
    parser.add_argument("--date", type=str, default=None,
                        help="交易日期 YYYY-MM-DD (默认今天)")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式: 打印步骤但不执行")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="跳过数据拉取 (使用已有缓存)")
    parser.add_argument("--signal-only", action="store_true",
                        help="仅生成信号 (跳过数据拉取和质量校验)")

    args = parser.parse_args()

    try:
        run = run_pipeline(
            date_str=args.date,
            dry_run=args.dry_run,
            skip_fetch=args.skip_fetch,
            signal_only=args.signal_only,
        )
    except GateViolation as e:
        log.error("门禁拦截: %s", e)
        sys.exit(2)

    sys.exit(0 if run.success else 1)


if __name__ == "__main__":
    main()
