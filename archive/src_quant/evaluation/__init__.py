"""
Strategy evaluation and validation toolkit.

Provides tools for:
    - IC (Information Coefficient) analysis: measure predictive power of factors
    - Return attribution: decompose returns into beta, alpha, and costs
    - Robustness checks: statistical validation of strategy performance
"""

from .ic_analysis import ICAnalyzer
from .attribution import ReturnAttribution, AttributionResult
from .robustness import RobustnessChecker

__all__ = [
    "ICAnalyzer",
    "ReturnAttribution",
    "AttributionResult",
    "RobustnessChecker",
]
