"""模拟盘个股盈亏端点 (SQLite trades FIFO + 持仓浮盈亏)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402
from web.api.aggregators import aggregate_stock_pnl  # noqa: E402

router = APIRouter(prefix="/api/paper", tags=["paper"])


@router.get("/stock-pnl")
def paper_stock_pnl():
    rows = storage.get_trades(limit=10000)
    trades = [{"date": r["date"], "symbol": r["symbol"], "action": r["action"],
               "price": r["price"], "qty": r["qty"], "commission": r["commission"]}
              for r in rows]
    agg = {r["symbol"]: r for r in aggregate_stock_pnl(trades)}
    items = []
    for sym, a in agg.items():
        pos = storage.get_position(sym)
        open_qty = (pos or {}).get("qty") or a.get("open_qty") or 0
        avg_cost = (pos or {}).get("avg_cost") or 0
        path = config.DATA_STORE / f"{sym}.parquet"
        cur_price = None
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["close"])
                cur_price = float(df["close"].iloc[-1]) if len(df) else None
            except Exception:
                pass
        unreal = (round((cur_price - avg_cost) * open_qty, 2)
                  if cur_price and avg_cost and open_qty else None)
        items.append({**a, "open_qty": open_qty, "avg_cost": avg_cost,
                      "current_price": cur_price, "unrealized_pnl": unreal})
    items.sort(key=lambda x: -(x["total_pnl"] + (x["unrealized_pnl"] or 0)))
    return {"items": items, "count": len(items)}
