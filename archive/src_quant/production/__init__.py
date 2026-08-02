"""
Production infrastructure for the quant-starter trading system.

Modules:
    signal_generator - Daily signal generation from IC-weighted linear model
    paper_trader     - Paper trading simulation with realistic execution
    risk_monitor     - Real-time risk monitoring and alerting
"""

from quant.production.signal_generator import SignalGenerator
from quant.production.paper_trader import PaperTrader
from quant.production.risk_monitor import RiskMonitor, RiskReport

__all__ = ["SignalGenerator", "PaperTrader", "RiskMonitor", "RiskReport"]
