"""
factor_factory — 因子研究基础设施

提供因子全生命周期管理: 注册 → 计算 → 验证 → 中性化 → 报告

用法:
    from factor_factory import FactorRegistry, validate_factor, neutralize

    # 注册因子
    reg = FactorRegistry.load()
    reg.register("mom_20d", expr="Ref($close, 20) / $close - 1",
                 category="momentum", freq="daily")

    # 一键验证
    report = validate_factor("mom_20d", period=("2018-01-01", "2022-12-31"))

    # 中性化
    neutralized = neutralize(factor_values, industry_map, market_cap)

CLI:
    py -m factor_factory list
    py -m factor_factory validate mom_20d
    py -m factor_factory validate --all --period 2018-2022
    py -m factor_factory report mom_20d
"""

from factor_factory.registry import FactorRegistry, FactorMeta
from factor_factory.validation import validate_factor, validate_batch
from factor_factory.neutralize import neutralize_cross_section

__all__ = [
    "FactorRegistry", "FactorMeta",
    "validate_factor", "validate_batch",
    "neutralize_cross_section",
]
