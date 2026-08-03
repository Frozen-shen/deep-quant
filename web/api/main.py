"""quant-starter 金融看板后端 API。"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # quant-starter 根
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI  # noqa: E402

from web.api import config  # noqa: E402
import storage  # noqa: E402

app = FastAPI(title="quant-starter Dashboard API", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from web.api.routers import equity, portfolio, graduation  # noqa: E402
from web.api.routers import signals, experiments, factors, universe, stocks  # noqa: E402
from web.api.routers import broker, decisions  # noqa: E402

app.include_router(equity.router)
app.include_router(portfolio.router)
app.include_router(graduation.router)
app.include_router(signals.router)
app.include_router(experiments.router)
app.include_router(factors.router)
app.include_router(universe.router)
app.include_router(stocks.router)
app.include_router(broker.router)
app.include_router(decisions.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_sources": {
            "equity_log": len(storage.get_equity_log(limit=1)) > 0,
            "portfolio": config.PAPER_PORTFOLIO.exists(),
            "signals": config.SIGNALS_FILE.exists(),
            "experiments": config.EXPERIMENTS_DIR.exists() and any(config.EXPERIMENTS_DIR.glob("exp_*.json")),
            "ic_results": (config.IC_DIR / "p3_full_ic.json").exists(),
            "data_store": config.DATA_STORE.exists(),
        },
    }
