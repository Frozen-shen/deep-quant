"""净值曲线端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
import storage  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["equity"])


def compute_summary(rows):
    """从 equity_log 行计算绩效摘要。rows: [{date,total_equity,daily_return}] 按日期升序。"""
    if not rows:
        return None
    prices = [r["total_equity"] for r in rows]
    rets = [r["daily_return"] for r in rows if r["daily_return"] is not None]
    total_return = prices[-1] / prices[0] - 1
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        max_dd = min(max_dd, p / peak - 1)
    vol = (sum(r * r for r in rets) / len(rets)) ** 0.5 if rets else 0.0
    sharpe = (sum(rets) / len(rets) / vol * (252 ** 0.5)) if vol > 0 else None
    return {"total_return": total_return, "max_drawdown": max_dd,
            "volatility": vol, "sharpe": sharpe}


@config.ttl_cache(60)
def load_equity_rows():
    return storage.get_equity_log(limit=252 * 3)


@router.get("/equity")
def get_equity():
    rows = load_equity_rows()
    rows_sorted = list(reversed(rows))
    curve = [{"date": r["date"], "total_equity": r["total_equity"],
              "daily_return": r["daily_return"]} for r in rows_sorted]
    return {"curve": curve, "summary": compute_summary(rows_sorted)}
