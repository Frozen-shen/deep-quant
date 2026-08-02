"""QmtAdapter — 迅投 miniQMT (xtquant) 仿真/实盘适配器。

依赖: 券商提供的 xtquant 包（非 pip），需 QMT 客户端本机登录。
未安装 xtquant 时 connect() 返回 False，系统自动降级 PaperAdapter。
"""
from typing import Dict, List, Optional
from execution.broker.base import BrokerAdapter


class QmtAdapter(BrokerAdapter):
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.xt = None
        self._trader = None
        try:
            from xtquant import xttrader  # type: ignore
            self.xt = xttrader
        except ImportError:
            self.xt = None

    def connect(self) -> bool:
        if self.xt is None:
            return False
        # TODO(用户): 按券商账号填充 config.yaml broker.qmt 段后启用
        # self._trader = self.xt.XtQuantTrader(path, session_id)
        # self._trader.start(); self._trader.connect()
        return False  # xtquant 可用但账号未配置前保持关闭

    def place_order(self, symbol: str, side: str, qty: int,
                    price_type: str = "limit", price: Optional[float] = None) -> str:
        raise NotImplementedError("QMT 未启用")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("QMT 未启用")

    def get_balance(self) -> Dict:
        return {"cash": 0.0, "frozen": 0.0, "total_asset": 0.0}

    def get_positions(self) -> List[Dict]:
        return []

    def get_orders(self, date: str) -> List[Dict]:
        return []

    def get_trades(self, date: str) -> List[Dict]:
        return []

    def get_quotes(self, symbols: List[str]) -> Dict:
        return {}
