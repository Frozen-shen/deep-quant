"""券商执行状态端点（默认 PaperAdapter 降级）。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from execution.broker import get_adapter  # noqa: E402
import storage  # noqa: E402

router = APIRouter(prefix="/api/broker", tags=["broker"])
_adapter = get_adapter("paper")  # config.yaml broker.adapter 可切换 qmt

# v24b 最优实验的 EXTEND 模拟考结果 (2026-08-12 部署为生产权重)
V24B_RESULT = (Path(__file__).resolve().parents[3] / "data" / "ic_validation"
               / "walkforward_results_v24b_vwap.json")


@router.get("/status")
def broker_status():
    return {
        "adapter": "paper",
        "connected": _adapter.connect(),
        "balance": _adapter.get_balance(),
        "positions": _adapter.get_positions(),
        "orders": _adapter.get_orders(""),
        "trades": _adapter.get_trades(""),
    }


@router.get("/trades")
def broker_trades(year: int | None = None, limit: int = 2000):
    """按年份查询成交记录（2021-2024 回测成交 / 2026 模拟盘实盘）。"""
    rows = storage.get_trades(year=year, limit=limit)
    return {"year": year, "count": len(rows), "trades": rows}


def _load_v24b_extend() -> dict | None:
    """加载 v24b EXTEND 模拟考数据: {positions_history, equity_curve, weights_evolution}。"""
    if not V24B_RESULT.exists():
        return None
    try:
        d = json.loads(V24B_RESULT.read_text(encoding="utf-8"))
        return d.get("results", {}).get("extend_val", {})
    except Exception:
        return None


def _reconstruct_v24b_trades(ev: dict) -> list[dict]:
    """从相邻调仓点持仓差异还原买卖记录 (EXTEND 2025-01~2026-06, 18 次调仓)。

    首调仓日 (前一次持仓为空) = 全部买入; 之后逐期 diff:
      新增持仓 = BUY, 退出持仓 = SELL。
    qty/price 置 0 (回测 JSON 无逐笔价格, 前端标注"换入/换出"即可)。
    """
    pts = [p for p in ev.get("positions_history", []) if p.get("positions")]
    trades = []
    prev: set[str] = set()
    for pt in pts:
        cur = set(pt["positions"])
        date = pt["date"]
        for s in sorted(cur - prev):      # 新增持仓 → 买入
            trades.append({"date": date, "symbol": s, "action": "BUY",
                           "qty": 0, "price": 0.0, "commission": 0.0,
                           "reason": "v24b实验"})
        for s in sorted(prev - cur):      # 退出持仓 → 卖出
            trades.append({"date": date, "symbol": s, "action": "SELL",
                           "qty": 0, "price": 0.0, "commission": 0.0,
                           "reason": "v24b实验"})
        prev = cur
    return trades


@router.get("/backtest-trades")
def backtest_trades():
    """v24b 最优实验 (EXTEND 模拟考) 的调仓记录 — 还原自回测 JSON。"""
    ev = _load_v24b_extend()
    if not ev:
        return {"available": False, "count": 0, "trades": [],
                "note": "walkforward_results_v24b_vwap.json 不存在"}
    trades = _reconstruct_v24b_trades(ev)
    return {
        "available": True,
        "version": "v24b (VWAP执行+10bps残差)",
        "period": ev.get("period"),
        "excess_annual": ev.get("excess_annual"),
        "sharpe": ev.get("sharpe"),
        "max_drawdown": ev.get("max_drawdown"),
        "n_rebalances": ev.get("n_rebalances"),
        "count": len(trades),
        "trades": trades,
    }
