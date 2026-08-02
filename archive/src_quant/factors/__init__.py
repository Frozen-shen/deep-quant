"""
Factor computation layer.

A clean, deduplicated factor module built around a Qlib-style expression DSL.

Public API
----------
- :class:`FactorEngine`   -- parse and evaluate factor expressions on OHLCV data
- :class:`FactorComputer` -- batch computation into factor panels + IC filtering
- :class:`FactorDef`      -- metadata for a single factor
- :func:`get_all_factor_defs`, :func:`get_expression_map`, :func:`validate_library`
- :class:`BuybackFactor`, :class:`PEADFactor` -- event-driven overlay factors

Quick start
-----------
>>> from quant.factors import FactorEngine, FactorComputer
>>> from quant.factors import get_all_factor_defs
>>> engine = FactorEngine()
>>> computer = FactorComputer(engine, get_all_factor_defs())
>>> panels = computer.compute_all(panel, symbols)   # panel: (date, symbol) MultiIndex
"""

from quant.factors.engine import FactorEngine, parse_factor
from quant.factors.library import (
    FactorDef,
    get_all_factor_defs,
    get_factor_defs_by_category,
    get_expression_map,
    validate_library,
    ALL_CATEGORIES,
)
from quant.factors.compute import FactorComputer, DataPanel
from quant.factors.events import BuybackFactor, PEADFactor

__all__ = [
    "FactorEngine",
    "parse_factor",
    "FactorComputer",
    "DataPanel",
    "FactorDef",
    "get_all_factor_defs",
    "get_factor_defs_by_category",
    "get_expression_map",
    "validate_library",
    "ALL_CATEGORIES",
    "BuybackFactor",
    "PEADFactor",
]
