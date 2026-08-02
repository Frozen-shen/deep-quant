"""PaperAdapter — 包装现有 PaperExecutor 的降级/回归实现。"""
from typing import Dict, List, Optional
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from execution.broker.base import BrokerAdapter  # noqa: E402
import storage  # noqa: E402


class PaperAdapter(BrokerAdapter):
    """模拟撮合：直接写 storage 的 positions/trades 表。"""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}

    def connect(self) -> bool:
        return True

    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        import uuid
        order_id = str(uuid.uuid4())[:8]
        # 简化：按最新收盘价成交（真实路径走 PaperExecutor.execute_orders）
        storage.record_trade(symbol=symbol, market="A", date="", action=side,
                             qty=qty, price=price or 0.0, commission=0.0,
                             reason="paper_adapter")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_balance(self) -> Dict:
        pf = {}
        p = os.path.join(BASE, "paper_trade", "portfolio.json")
        if os.path.exists(p):
            import json
            with open(p, encoding="utf-8") as f:
                pf = json.load(f)
        return {"cash": pf.get("cash"), "frozen": 0.0,
                "total_asset": pf.get("cash", 0) + sum(
                    x.get("market_value", 0) for x in pf.get("positions", {}).values())}

    def get_positions(self) -> List[Dict]:
        out = []
        for p in storage.get_all_positions():
            out.append({"symbol": p["symbol"], "qty": p["qty"],
                        "avg_cost": p["avg_cost"], "market_value": 0.0})
        return out

    def get_orders(self, date: str) -> List[Dict]:
        return []

    def get_trades(self, date: str) -> List[Dict]:
        return storage.get_trades(limit=100)

    def get_quotes(self, symbols: List[str]) -> Dict:
        import data_cache
        out = {}
        for s in symbols:
            df = data_cache.load(s)
            if df is not None and len(df):
                out[s] = {"last": float(df["close"].iloc[-1]), "bid": 0.0, "ask": 0.0}
        return out
