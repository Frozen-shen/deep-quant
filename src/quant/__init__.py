"""
quant-starter: A quantitative trading system for A-share markets.

Provides factor-based stock selection, portfolio construction,
and walk-forward backtesting with conservative defaults.
"""

__version__ = "0.1.0"
__author__ = "quant-starter contributors"

from quant.config import QuantConfig

__all__ = ["QuantConfig", "__version__"]
