"""组合端点：读 paper_trade/portfolio.json + storage。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["portfolio"])


@config.ttl_cache(30)
def load_portfolio():
    return config.read_json(config.PAPER_PORTFOLIO)


def _current_price(symbol: str) -> float | None:
    """data_store 日线最后一根 close (与模拟盘 market_value 同日口径)。"""
    path = config.DATA_STORE / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date", "close"])
        if len(df) == 0:
            return None
        df = df.sort_values("date")
        return float(df["close"].iloc[-1])
    except Exception:
        return None


@router.get("/portfolio")
def get_portfolio():
    pf = load_portfolio() or {}
    positions = []
    for sym, p in (pf.get("positions") or {}).items():
        cur = _current_price(sym)
        qty = p.get("qty") or 0
        avg = p.get("avg_cost") or 0
        mv = p.get("market_value") or 0
        cost = avg * qty
        pnl = mv - cost
        positions.append({
            "symbol": sym, "qty": qty, "avg_cost": avg,
            "market_value": mv, "entry_date": p.get("entry_date"),
            "current_price": cur,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / cost, 4) if cost else None,
        })
    trades = storage.get_trades(limit=50)
    return {"cash": pf.get("cash"), "initial_capital": pf.get("initial_capital"),
            "inception_date": pf.get("inception_date"), "positions": positions,
            "recent_trades": trades}
