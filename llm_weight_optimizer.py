"""
LLM 权重优化器 — 替代手调 preset

对每只股票，LLM 定制因子权重和阈值：
  - 高波动股 → 加大波动惩罚,降低动量权重
  - 互联网股 → 加大动量权重,快速进出
  - 金融股 → 均衡权重,控制交易频率

用法:
    opt = LLMWeightOptimizer(backend="mock")
    weights = opt.optimize(symbol, features)
    scorer = FactorScorer(weights, buy_threshold, sell_threshold)
"""

import os
import json
import numpy as np


WEIGHT_PROMPT = """你是量化策略优化专家。根据股票特征,给出该股最优的因子权重和交易阈值。

输出JSON:
{
  "factor_weights": {"return_5d": 0.15, "volatility_20d": -0.1, ...},
  "buy_threshold": 0.15,
  "sell_threshold": -0.1,
  "reason": "一句话理由"
}
可用因子: return_5d, return_20d, return_60d, ma5_ma20_spread, ma10_ma20_spread, ma20_ma60_spread, ma5_cross_ma20, ma_bullish, ma_bearish, vol_ratio, vol_up_price_up, vol_up_price_down, volatility_20d, position_20d"""


class LLMWeightOptimizer:
    """LLM 因子权重优化器。"""

    # 基础因子池
    ALL_FACTORS = [
        "return_5d", "return_20d", "return_60d",
        "ma5_ma20_spread", "ma10_ma20_spread", "ma20_ma60_spread",
        "ma5_cross_ma20", "ma_bullish", "ma_bearish",
        "vol_ratio", "vol_up_price_up", "vol_up_price_down",
        "volatility_20d", "position_20d",
    ]

    def __init__(self, backend: str = "mock", api_key: str = None,
                 model: str = None, base_url: str = None):
        self.backend = backend
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    def optimize(self, symbol: str, features: dict) -> dict:
        """
        为股票定制因子权重。

        features: {volatility, trend_adx, daily_range, ...}
        返回: {factor_weights: dict, buy_threshold: float, sell_threshold: float}
        """
        if self.backend != "mock":
            return self._llm_optimize(symbol, features)
        return self._rule_based(symbol, features)

    def _rule_based(self, symbol: str, features: dict) -> dict:
        """规则引擎: 根据股票特征分配权重。"""
        vol = features.get("volatility", 0.3)
        adx = features.get("trend_adx", 20)
        daily_range = features.get("daily_range", 0.03)
        sector = features.get("sector", "")

        weights = {f: 0.0 for f in self.ALL_FACTORS}

        # 基础动量权重
        weights["return_5d"] = 0.15
        weights["return_20d"] = 0.10
        weights["ma5_ma20_spread"] = 0.15
        weights["ma5_cross_ma20"] = 0.10

        # 趋势强 → 加动量, 减波动惩罚
        if adx > 25:
            weights["return_20d"] += 0.05
            weights["ma20_ma60_spread"] += 0.10
            weights["ma_bullish"] += 0.05
            weights["volatility_20d"] = -0.05  # 低惩罚
            buy_threshold = 0.20
            sell_threshold = -0.15

        # 波动高 → 加波动惩罚, 减动量
        elif daily_range > 0.04:
            weights["volatility_20d"] = -0.20
            weights["position_20d"] = 0.10
            weights["vol_up_price_down"] = -0.10
            weights["return_5d"] = 0.10
            buy_threshold = 0.25
            sell_threshold = -0.10

        # 中等趋势 → 均衡
        else:
            weights["vol_ratio"] = 0.10
            weights["vol_up_price_up"] = 0.05
            weights["ma10_ma20_spread"] = 0.05
            weights["volatility_20d"] = -0.10
            buy_threshold = 0.15
            sell_threshold = -0.10

        # 互联网/科技 → 更依赖动量
        if "互联网" in sector or "半导体" in sector or "科技" in sector:
            weights["return_5d"] += 0.05
            weights["return_20d"] += 0.05

        # 消费/医药 → 更依赖均线
        if "消费" in sector or "医药" in sector or "体育" in sector:
            weights["ma5_ma20_spread"] += 0.05
            weights["ma20_ma60_spread"] += 0.05

        return {
            "factor_weights": weights,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }

    def _llm_optimize(self, symbol: str, features: dict) -> dict:
        """LLM 模式 (预留)。"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            from llm_factor import OpenAIBackend
            llm = OpenAIBackend(api_key, os.environ.get("LLM_MODEL", "deepseek-chat"),
                               os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
            resp = llm.query(WEIGHT_PROMPT,
                           f"股票: {symbol}\n特征: {json.dumps(features, ensure_ascii=False)}")
            import re
            m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"[LLMWeight] LLM失败,用规则: {e}")
        return self._rule_based(symbol, features)
