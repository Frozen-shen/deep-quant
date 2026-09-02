"""
因子名单容器 — 供下游 IC 加权 / ML 学习真实权重

历史说明:
  本模块早期内置了一批手工/历史硬编码数值权重预设 (ic_auto / ic_top20 /
  ic_optimized / ic_optimized_v2 / trend_momentum / mean_reversion / a_share /
  balanced) 以及配套的加权打分方法 (score / normalize / cross_sectional_score /
  normalize_cross_sectional)。经全仓库 grep 确认: 所有活跃调用点
  (decision_engine / model.pipeline / run_walkforward_backtest / run_paper_signal /
  run_full_ic_validation 等) 只通过 FactorScorer.from_preset(...).factor_weights.keys()
  取"因子名单", 交由 FactorCache.compute_factors + 下游 IC 加权 / LightGBM 学习真实
  权重, 从未调用上述加权打分方法, 硬编码数值权重从未生效。故予以删除 (纪律第十条:
  模块准入 / 防平行代码库)。仅保留权重恒为 1.0 的因子名单容器预设
  (full_auto / full_auto_v5 / alpha158_full)。

用法:
    from factor_scorer import FactorScorer
    scorer = FactorScorer.from_preset("full_auto")
    factor_names = sorted(scorer.factor_weights.keys())   # 因子名单
"""

import pandas as pd
from factor_engine import FactorLibrary
from factor_library import get_all_factors, FUNDAMENTAL_FACTORS, AUX_FACTORS


# ================================================================
#  因子名单预设 (权重恒为 1.0, 真实权重由下游 IC/ML 学习)
# ================================================================

FACTOR_PRESETS: dict = {}

# ── 动态构建 full_auto 预设: 包含全部因子, 权重=1.0 (由LightGBM学习) ──
def _build_full_auto_factors():
    """全部 DSL 价量因子, 权重=1.0 (full_auto 与 full_auto_v5 共用)。"""
    from factor_library import get_all_factors
    lib = get_all_factors()
    names = list(lib.factors.keys()) if hasattr(lib, 'factors') else []
    if not names:
        # fallback: import dicts directly
        from factor_library import (PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS,
            CANDLESTICK_FACTORS, EXPANDED_FACTORS, PHASE2_FACTORS,
            P2_ENHANCED_FACTORS, MICROSTRUCTURE_FACTORS)
        merged = {}
        for d in [PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS, CANDLESTICK_FACTORS,
                  EXPANDED_FACTORS, PHASE2_FACTORS, P2_ENHANCED_FACTORS, MICROSTRUCTURE_FACTORS]:
            merged.update(d)
        names = list(merged.keys())
    return {name: 1.0 for name in names}

def _build_full_auto_preset():
    return {
        "name": "全因子(LightGBM自动学习权重)",
        "factors": _build_full_auto_factors(),
        "buy_threshold": 0.15,
        "sell_threshold": -0.10,
    }

FACTOR_PRESETS["full_auto"] = _build_full_auto_preset()

# ★ v5: 方案C — 全部价量因子 + 基本面因子 (fund_* 权重暂1.0, 由fold筛选决定实际使用)
# market_cap/liq_ratio 曾因缺 outstanding_share 字段被剔除;
# 2026-08-03 已通过 baostock volume/turnover 反推补全 outstanding_share → 恢复
_V5_EXCLUDE: set = set()

def _build_full_auto_v5_factors():
    return {k: 1.0 for k in _build_full_auto_factors()
            if k not in _V5_EXCLUDE}

FACTOR_PRESETS["full_auto_v5"] = {
    "name": "方案C v5: 全部价量因子 + 基本面因子 + 辅助数据因子 (fold筛选决定权重)",
    "factors": {**_build_full_auto_v5_factors(),
                **{k: 1.0 for k in FUNDAMENTAL_FACTORS},
                **{k: 1.0 for k in AUX_FACTORS}},
    "buy_threshold": 0.15,
    "sell_threshold": -0.10,
}


# ── 动态构建 alpha158_full 预设: Qlib Alpha158 价量因子全集, 权重=1.0 (由LightGBM/IC学习) ──
def _build_alpha158_full_preset():
    from factor_library import (NEW_KLINE_FACTORS, NEW_ROLLING_FACTORS,
        NEW_TURNOVER_FACTORS, NEW_BOLL_FACTORS, ALPHA158_FACTORS)
    merged = {}
    for d in [NEW_KLINE_FACTORS, NEW_ROLLING_FACTORS, NEW_TURNOVER_FACTORS,
              NEW_BOLL_FACTORS, ALPHA158_FACTORS]:
        merged.update(d)
    return {
        "name": "Alpha158价量因子全集(LightGBM/IC自动学习权重)",
        "factors": {name: 1.0 for name in merged},
        "buy_threshold": 0.15,
        "sell_threshold": -0.10,
    }

FACTOR_PRESETS["alpha158_full"] = _build_alpha158_full_preset()


# ================================================================
#  因子名单容器
# ================================================================

class FactorScorer:
    """
    因子名单容器。

    仅负责: 从预设解析出因子名单 (factor_weights.keys(), 权重恒为 1.0),
    并基于 FactorLibrary 计算因子原始值。真实的因子加权由下游
    (IC 加权 / LightGBM) 学习, 不在本类内进行。
    """

    def __init__(self, factor_weights: dict, buy_threshold: float = 0.3,
                 sell_threshold: float = -0.2, norm_window: int = 252):
        self.factor_weights = factor_weights
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.norm_window = norm_window

        # 构建 FactorLibrary
        config = {name: self._expr_for_factor(name)
                  for name in factor_weights.keys()}
        self.library = FactorLibrary.from_config(config)

    @classmethod
    def from_preset(cls, preset_name: str = "full_auto"):
        """从因子名单预设创建。"""
        preset = FACTOR_PRESETS.get(preset_name, FACTOR_PRESETS["full_auto"])
        return cls(
            factor_weights=preset["factors"],
            buy_threshold=preset["buy_threshold"],
            sell_threshold=preset["sell_threshold"],
        )

    def _expr_for_factor(self, name: str) -> str:
        """根据因子名反查表达式 (从 factor_library 的预定义)。"""
        all_config = {}
        from factor_library import (PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS,
            CANDLESTICK_FACTORS, NEW_KLINE_FACTORS, NEW_ROLLING_FACTORS,
            NEW_TURNOVER_FACTORS, NEW_BOLL_FACTORS, EXPANDED_FACTORS,
            PHASE2_FACTORS, P2_ENHANCED_FACTORS, MICROSTRUCTURE_FACTORS,
            ALPHA158_FACTORS, MOMENTUM_GROWTH_FACTORS)
        for d in [PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS, CANDLESTICK_FACTORS,
                  NEW_KLINE_FACTORS, NEW_ROLLING_FACTORS, NEW_TURNOVER_FACTORS, NEW_BOLL_FACTORS,
                  EXPANDED_FACTORS, PHASE2_FACTORS, P2_ENHANCED_FACTORS,
                  MICROSTRUCTURE_FACTORS, ALPHA158_FACTORS, MOMENTUM_GROWTH_FACTORS]:
            all_config.update(d)
        return all_config.get(name, f"${name}")

    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子原始值。"""
        return self.library.evaluate_all(df)
