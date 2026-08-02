"""每日信号端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["signals"])


@config.ttl_cache(30)
def load_signals():
    if not config.SIGNALS_FILE.exists():
        return []
    out = []
    with open(config.SIGNALS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                out.append(json.loads(line))
    return out[-200:]


@router.get("/signals")
def get_signals():
    sigs = load_signals()
    return {"count": len(sigs), "signals": sigs, "fill_rate": None}
