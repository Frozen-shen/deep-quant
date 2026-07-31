"""
Unified data layer for the quant-starter system.

This package provides a clean, cohesive interface for all data operations:
  - DataFetcher: network fetching with caching, rate limiting, and retry
  - DataStore: persistent parquet storage with DataPanel construction
  - DataPanel: the central data structure (date x symbol access patterns)
  - Universe: stock universe management with point-in-time support
  - TradingCalendar: trading day lookup and offset computation
  - DataValidator: data quality checks and reporting

Quick start:
    from quant.data import DataStore, TradingCalendar, Universe, DataValidator

    # Build a data panel (fetch + cache + load)
    store = DataStore()
    panel = store.build_panel(["600519", "000858", "300750"], "20200101", "20260101")

    # Access data
    close_prices = panel.get_panel("close")           # Date x Symbol matrix
    cross_section = panel.get_cross_section("2023-06-15", "close")  # One date
    single_stock = panel.get("600519", "2023-01-01", "2023-12-31")  # One symbol

    # Trading calendar
    cal = TradingCalendar()
    days = cal.get_trading_days("2023-01-01", "2023-06-30")
    prev = cal.offset("2023-06-15", -5)

    # Universe
    uni = Universe(index="csi1000")
    symbols = uni.get_symbols(as_of_date="2023-06-01")

    # Validation
    validator = DataValidator(calendar=cal)
    report = validator.validate_panel(panel)
    print(report.summary())
"""

from quant.data.calendar import TradingCalendar
from quant.data.fetcher import DataFetcher
from quant.data.store import DataPanel, DataStore
from quant.data.universe import Universe
from quant.data.validator import DataValidator, ValidationReport

__all__ = [
    "TradingCalendar",
    "DataFetcher",
    "DataPanel",
    "DataStore",
    "Universe",
    "DataValidator",
    "ValidationReport",
]
