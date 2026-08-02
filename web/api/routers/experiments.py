"""实验记录端点。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter  # noqa: E402
from web.api import config  # noqa: E402

router = APIRouter(prefix="/api", tags=["experiments"])


@config.ttl_cache(60)
def load_experiments():
    exps = []
    if config.EXPERIMENTS_DIR.exists():
        for p in sorted(config.EXPERIMENTS_DIR.glob("exp_*.json")):
            try:
                exps.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return exps


@router.get("/experiments")
def get_experiments():
    exps = load_experiments()
    by_script = {}
    by_partition = {}
    for e in exps:
        by_script[e.get("script", "?")] = by_script.get(e.get("script", "?"), 0) + 1
        by_partition[e.get("partition", "?")] = by_partition.get(e.get("partition", "?"), 0) + 1
    return {"count": len(exps), "experiments": exps[-100:],
            "by_script": by_script, "by_partition": by_partition}
