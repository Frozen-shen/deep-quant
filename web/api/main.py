"""quant-starter 金融看板后端 API。"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # quant-starter 根
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI  # noqa: E402

app = FastAPI(title="quant-starter Dashboard API", version="0.1.0")


@app.get("/api/health")
def health():
    return {"status": "ok", "data_sources": {"equity_log": False}}
