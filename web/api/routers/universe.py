"""股票池/名称/板块端点。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["universe"])


@config.ttl_cache(300)
def load_universe():
    names = {}
    sectors = {}
    for fname, target in (("stock_names.json", names), ("a_sectors.json", sectors)):
        p = config.DATA_CACHE / fname
        if p.exists():
            try:
                target.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    syms = sorted(set(list(names.keys()) + list(sectors.keys())))
    return names, sectors, syms


@router.get("/universe")
def get_universe():
    names, sectors, syms = load_universe()
    return {"total": len(syms), "stocks": [
        {"symbol": s, "name": names.get(s, ""), "sector": sectors.get(s, "")} for s in syms]}


@router.get("/universe/search")
def search_universe(q: str):
    names, sectors, syms = load_universe()
    q = q.strip()
    hits = []
    for s in syms:
        if q in s or q in names.get(s, ""):
            hits.append({"symbol": s, "name": names.get(s, ""), "sector": sectors.get(s, "")})
            if len(hits) >= 50:
                break
    return {"stocks": hits}
