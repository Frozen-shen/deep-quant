"""个股 K 线端点。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException  # noqa: E402
import pandas as pd  # noqa: E402
from web.api import config  # noqa: E402
from web.api.routers.universe import load_universe  # noqa: E402

router = APIRouter(prefix="/api", tags=["stocks"])


@config.ttl_cache(120)
def load_ohlc(symbol: str):
    p = config.DATA_STORE / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    return df.tail(500)


@router.get("/stocks/{symbol}")
def get_stock(symbol: str):
    df = load_ohlc(symbol)
    if df is None:
        raise HTTPException(status_code=404, detail="symbol not found")
    names, _, _ = load_universe()
    ohlc = [{"date": str(r["date"])[:10], "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]),
             "volume": float(r["volume"]) if "volume" in df.columns else None}
            for _, r in df.iterrows()]
    return {"symbol": symbol, "name": names.get(symbol, ""), "ohlc": ohlc, "signals": []}
