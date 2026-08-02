"""券商执行状态端点（默认 PaperAdapter 降级）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from execution.broker import get_adapter  # noqa: E402

router = APIRouter(prefix="/api/broker", tags=["broker"])
_adapter = get_adapter("paper")  # config.yaml broker.adapter 可切换 qmt


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
