"""
订单执行模块 — 模拟下单 & 券商API预留接口

用法:
    executor = MockExecutor()
    order_id = executor.place("01810", "BUY", 200, 35.20)
    executor.list_orders()
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    action: str       # BUY / SELL
    qty: int
    price: float
    status: str       # PENDING / FILLED / CANCELLED
    created_at: str
    filled_at: str = ""
    reason: str = ""


class BaseExecutor(ABC):
    """执行器抽象基类"""
    @abstractmethod
    def place(self, symbol: str, action: str, qty: int,
              price: float, reason: str = "") -> str:
        """下单，返回 order_id"""
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """撤单"""
        ...

    @abstractmethod
    def list_orders(self) -> List[Order]:
        """当日委托列表"""
        ...


class MockExecutor(BaseExecutor):
    """
    模拟执行器 — 打印指令，不实际下单。

    用于开发测试和手动确认模式。
    """

    def __init__(self, log_file: Optional[str] = None):
        self._orders: Dict[str, Order] = {}
        self._counter = 0
        self.log_file = log_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "orders.log"
        )

    def place(self, symbol: str, action: str, qty: int,
              price: float, reason: str = "") -> str:
        self._counter += 1
        oid = f"ORD-{datetime.now():%Y%m%d}-{self._counter:04d}"
        order = Order(
            id=oid, symbol=symbol, action=action, qty=qty,
            price=price, status="FILLED",  # 模拟立即成交
            created_at=datetime.now().isoformat(),
            filled_at=datetime.now().isoformat(),
            reason=reason,
        )
        self._orders[oid] = order

        # 打印指令 (醒目)
        emoji = "🔴" if action == "BUY" else "🟢"
        total = qty * price
        line = (
            f"\n{'='*50}\n"
            f"  {emoji} 模拟下单: {action} {symbol}\n"
            f"  数量: {qty}股  价格: {price:.2f}  金额: {total:,.2f}\n"
            f"  订单号: {oid}\n"
            f"  理由: {reason}\n"
            f"{'='*50}\n"
        )
        print(line)

        # 记录到日志
        self._log(line)

        return oid

    def cancel(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].status == "PENDING":
            self._orders[order_id].status = "CANCELLED"
            return True
        return False

    def list_orders(self) -> List[Order]:
        return list(self._orders.values())

    def _log(self, text: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
