"""组合端点：读 paper_trade/portfolio.json + storage。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["portfolio"])


@config.ttl_cache(30)
def load_portfolio():
    return config.read_json(config.PAPER_PORTFOLIO)


@router.get("/portfolio")
def get_portfolio():
    pf = load_portfolio() or {}
    positions = []
    for sym, p in (pf.get("positions") or {}).items():
        positions.append({"symbol": sym, "qty": p.get("qty"), "avg_cost": p.get("avg_cost"),
                          "market_value": p.get("market_value"), "entry_date": p.get("entry_date")})
    trades = storage.get_trades(limit=50)
    return {"cash": pf.get("cash"), "initial_capital": pf.get("initial_capital"),
            "inception_date": pf.get("inception_date"), "positions": positions,
            "recent_trades": trades}
