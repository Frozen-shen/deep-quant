"""
因子模块

包含:
  - pead_factor: PEAD 事件因子 (业绩预告惊喜度)
  - earnings_momentum: 盈利动量因子 (SUE, ROE加速度, 营收惊喜, 预告惊喜, 应计比率)
  - money_flow: 资金流因子 (机构/聪明钱行为信号)
  - analyst_revision: 分析师修正因子 (预期修正, 评级升级, 目标价空间, 覆盖变化)
  - event_signals: 事件信号因子 (解禁压力, 龙虎榜机构, 预告相对惊喜)
"""

from factors.pead_factor import PEADFactor
from factors.earnings_momentum import EarningsMomentum, EM_FACTOR_NAMES
from factors.money_flow import MoneyFlowFactor, FACTOR_NAMES as FLOW_FACTOR_NAMES
from factors.analyst_revision import AnalystRevision, FACTOR_NAMES as ANALYST_FACTOR_NAMES
from factors.event_signals import EventSignals, FACTOR_NAMES as EVENT_FACTOR_NAMES

__all__ = [
    "PEADFactor", "EarningsMomentum", "EM_FACTOR_NAMES",
    "MoneyFlowFactor", "FLOW_FACTOR_NAMES",
    "AnalystRevision", "ANALYST_FACTOR_NAMES",
    "EventSignals", "EVENT_FACTOR_NAMES",
]
