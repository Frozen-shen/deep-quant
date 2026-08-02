"""券商执行适配器抽象 — 可插拔执行层（仿真/实盘）。"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BrokerAdapter(ABC):
    """统一券商接口。side: BUY/SELL; price_type: limit/market。"""

    @abstractmethod
    def connect(self) -> bool:
        """建立连接/登录。返回是否可用。"""

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        """下单，返回 order_id。"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单。"""

    @abstractmethod
    def get_balance(self) -> Dict:
        """返回 {cash, frozen, total_asset}。"""

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """返回 [{symbol, qty, avg_cost, market_value}]。"""

    @abstractmethod
    def get_orders(self, date: str) -> List[Dict]:
        """返回当日订单 [{order_id, symbol, side, qty, filled_qty, status}]。"""

    @abstractmethod
    def get_trades(self, date: str) -> List[Dict]:
        """返回当日成交 [{order_id, symbol, side, qty, price, amount}]。"""

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict:
        """返回 {symbol: {"last": float, "bid": float, "ask": float}}。"""
