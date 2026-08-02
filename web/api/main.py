"""quant-starter 金融看板后端 API。"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # quant-starter 根
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI  # noqa: E402

app = FastAPI(title="quant-starter Dashboard API", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from web.api.routers import equity, portfolio  # noqa: E402

app.include_router(equity.router)
app.include_router(portfolio.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "data_sources": {"equity_log": False}}
