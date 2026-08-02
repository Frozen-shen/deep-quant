"""后端路径常量与缓存工具。"""
import json
import time
from functools import wraps
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "quant.db"
PAPER_PORTFOLIO = BASE_DIR / "paper_trade" / "portfolio.json"
PAPER_RISK = BASE_DIR / "paper_trade" / "risk_report.json"
SIGNALS_FILE = BASE_DIR / "data" / "paper_signals_v3.jsonl"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
IC_DIR = BASE_DIR / "data" / "ic_validation"
DATA_STORE = BASE_DIR / "data_store"
DATA_CACHE = BASE_DIR / "data_cache"


def ttl_cache(seconds: int = 60):
    """简单 TTL 缓存装饰器（JSON 安全）。"""
    cache = {}

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            hit = cache.get(key)
            if hit and time.time() - hit[0] < seconds:
                return hit[1]
            val = fn(*args, **kwargs)
            cache[key] = (time.time(), val)
            return val
        return wrapper
    return deco


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
