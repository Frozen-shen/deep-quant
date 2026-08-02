"""执行适配器工厂。config: {"adapter": "paper"|"qmt", "qmt": {...}}"""
from typing import Optional

from execution.broker.paper import PaperAdapter


def get_adapter(name: str = "paper", cfg: Optional[dict] = None):
    if name == "qmt":
        from execution.broker.qmt import QmtAdapter
        return QmtAdapter(cfg or {})
    from execution.broker.paper import PaperAdapter
    return PaperAdapter(cfg or {})
