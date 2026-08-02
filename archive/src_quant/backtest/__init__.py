"""
Vectorized backtesting engine with realistic A-share market microstructure.

Key features:
    - Realistic transaction costs (~10.5bp round trip, not the old 30bp)
    - T+1 settlement handling
    - Limit-up/down detection
    - Proper turnover tracking
    - Benchmark comparison (CSI1000 default)
"""

from .engine import BacktestEngine, BacktestResult
from .costs import CostModel
from .metrics import compute_metrics, format_report, compare_strategies

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "compute_metrics",
    "format_report",
    "compare_strategies",
]
