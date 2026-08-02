"""
Portfolio construction, risk management, and rebalancing.

This module converts model scores into tradeable portfolio weights
with proper risk constraints and turnover control.

Key design decisions (learned from old system failures):
    - Old system: excessive turnover (>1000%/year) destroyed all alpha
    - Equal-weight no-trade baseline achieved IR=0.377
    - New approach: quarterly rebalancing, max 300% annual turnover
"""

from .constructor import PortfolioConstructor
from .risk import RiskManager, RiskReport
from .rebalance import RebalanceController

__all__ = [
    "PortfolioConstructor",
    "RiskManager",
    "RiskReport",
    "RebalanceController",
]
