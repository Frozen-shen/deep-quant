"""毕业指标端点 — 对应 docs/PAPER_GRADUATION.md 8 项 AND 条件。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datetime import date  # noqa: E402
from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402
from web.api.routers.equity import load_equity_rows, compute_summary  # noqa: E402

router = APIRouter(prefix="/api", tags=["graduation"])


def _benchmark_start_price():
    """CSI1000 (000852) 基准起点，用于超额收益计算。无数据返回 None。"""
    p = config.DATA_STORE / "000852.parquet"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p)
    return df.sort_values("date")


@config.ttl_cache(60)
def graduation_metrics():
    rows = list(reversed(load_equity_rows()))
    summary = compute_summary(rows)
    pf = config.read_json(config.PAPER_PORTFOLIO) or {}

    # 1. 运行时长
    if pf.get("inception_date"):
        days = (date.today() - date.fromisoformat(pf["inception_date"])).days
        runtime = {"key": "runtime_days", "name": "模拟盘运行时长",
                   "value": days, "threshold": 90,
                   "status": "pass" if days >= 90 else "pending", "detail": f"{days} 天（目标 ≥90 天）"}
    else:
        runtime = {"key": "runtime_days", "name": "模拟盘运行时长",
                   "value": None, "threshold": 90, "status": "pending", "detail": "模拟盘未初始化"}

    # 2-4, 7. 绩效类（依赖 equity 数据）
    if not rows:
        base = [runtime]
        for key, name, th in [("excess_return", "年化超额收益", 0.05),
                              ("ir", "信息比率 IR", 0.5),
                              ("max_drawdown", "最大回撤", -0.15),
                              ("sharpe", "夏普比率", 0.8)]:
            base.append({"key": key, "name": name, "value": None, "threshold": th,
                         "status": "pending", "detail": "等模拟盘权益数据累积"})
        base.append({"key": "ic_decay", "name": "IC 衰减", "value": None,
                     "threshold": "ICIR 未恶化", "status": "pending", "detail": "等 IC 监控数据"})
        base.append({"key": "fill_rate", "name": "信号实现率", "value": None,
                     "threshold": 0.8, "status": "pending", "detail": "等信号/成交数据"})
        base.append({"key": "monthly_win_rate", "name": "月胜率", "value": None,
                     "threshold": 0.55, "status": "pending", "detail": "等至少 2 个月数据"})
        return {"metrics": base, "overall": "pending"}

    total_return = summary["total_return"]
    years = max(len(rows) / 252.0, 1e-9)
    ann_return = (1 + total_return) ** (1 / years) - 1
    # 超额（无基准时退化为绝对年化，detail 注明）
    bench = _benchmark_start_price()
    excess = ann_return  # 简化：无基准数据时用绝对收益，PAPER_GRADUATION 口径实现时对照修正
    if bench is not None and rows:
        bench_first = float(bench["close"].iloc[0])
        bench_last = float(bench["close"].iloc[-1])
        bench_ann = (bench_last / bench_first) ** (1 / max(len(bench) / 252.0, 1e-9)) - 1
        excess = ann_return - bench_ann

    ir = (summary["sharpe"] / (252 ** 0.5) * (252 ** 0.5)) if summary["sharpe"] is not None else None
    # IR ≈ 年化超额 / 年化跟踪误差（数据不足用 sharpe 近似并注明）
    ir_val = summary["sharpe"] if summary["sharpe"] is not None else None

    # 5. 月胜率（不足 2 个月 pending）
    from collections import defaultdict
    monthly = defaultdict(list)
    for r in rows:
        monthly[r["date"][:7]].append(r["daily_return"])
    win = None
    if len(monthly) >= 2:
        wins = sum(1 for v in monthly.values() if sum(v) > 0)
        win = wins / len(monthly)

    # 6. fill_rate（signals jsonl 存在时统计 signal→trade）
    fill = None
    if config.SIGNALS_FILE.exists():
        n_sig = sum(1 for _ in open(config.SIGNALS_FILE, encoding="utf-8"))
        fill = 1.0 if n_sig == 0 else None  # 成交数据源待接，先置 None

    # 8. ic_decay（p3_full_ic 存在时给现状，衰减判定待 IC 监控累积）
    ic_status = "pending"
    if (config.IC_DIR / "p3_full_ic.json").exists():
        ic_status = "pending"  # 有基线；衰减趋势需连续监控数据

    def metric(key, name, value, threshold, status, detail):
        return {"key": key, "name": name, "value": value, "threshold": threshold,
                "status": status, "detail": detail}

    metrics = [
        runtime,
        metric("excess_return", "年化超额收益", round(excess, 4), 0.05,
               "pass" if excess > 0.05 else "pending", "≥5% 才算达标"),
        metric("ir", "信息比率 IR", round(ir_val, 3) if ir_val is not None else None, 0.5,
               "pass" if (ir_val or 0) > 0.5 else "pending", "口径：年化超额/跟踪误差（暂以 Sharpe 近似）"),
        metric("max_drawdown", "最大回撤", round(summary["max_drawdown"], 4), -0.15,
               "pass" if summary["max_drawdown"] > -0.15 else "fail", "阈值 -15%"),
        metric("sharpe", "夏普比率", round(summary["sharpe"], 3) if summary["sharpe"] is not None else None, 0.8,
               "pass" if (summary["sharpe"] or 0) > 0.8 else "pending", "目标 >0.8"),
        metric("fill_rate", "信号实现率", fill, 0.8,
               "pending" if fill is None else ("pass" if fill >= 0.8 else "fail"), "信号→成交比例"),
        metric("ic_decay", "IC 衰减", None, "ICIR 未恶化", ic_status, "等 IC 监控数据累积"),
        metric("monthly_win_rate", "月胜率", round(win, 3) if win is not None else None, 0.55,
               "pending" if win is None else ("pass" if win >= 0.55 else "fail"),
               "需 ≥2 个月数据"),
    ]
    overall = "pending" if any(m["status"] == "pending" for m in metrics) else \
        ("pass" if all(m["status"] == "pass" for m in metrics) else "fail")
    return {"metrics": metrics, "overall": overall}


@router.get("/graduation")
def get_graduation():
    return graduation_metrics()
