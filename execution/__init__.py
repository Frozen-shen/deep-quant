"""
模拟盘执行模块

包含:
  - PaperExecutor: 持久化版模拟盘执行引擎
  - CircuitBreaker: 回撤熔断器
  - ExecutionReport: 执行报告
"""

from execution.paper_executor import (
    PaperExecutor,
    PaperState,
    ExecutionReport,
    OrderResult,
    RejectReason,
)

from execution.circuit_breaker import CircuitBreaker

__all__ = [
    "PaperExecutor",
    "PaperState",
    "ExecutionReport",
    "OrderResult",
    "RejectReason",
    "CircuitBreaker",
]
