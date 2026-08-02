"""因子 IC 端点（多来源：p3 价量 / p6 基本面 / p7 相对 / p8 北向 / p9 分钟）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["factors"])
SOURCES = ["p3_full_ic", "p6_fundamental_ic", "p7_relative_ic",
           "p8_northbound_ic", "p9_minute_ic"]


@config.ttl_cache(60)
def load_ic():
    results = {}
    for name in SOURCES:
        p = config.IC_DIR / f"{name}.json"
        if p.exists():
            results[name] = config.read_json(p)
    return results


@router.get("/factors/ic")
def get_ic():
    data = load_ic()
    return {"sources": list(data.keys()), "results": data}
