"""
多因子打分引擎 — 替代二元 MA 信号

核心思路:
  不是: MA金叉? → buy (二元)
  而是: 动量(+0.3) + 波动(-0.1) + 量价(+0.2) + 趋势(+0.1) = 0.5 → buy (连续打分)

用法:
    from factor_scorer import FactorScorer
    scorer = FactorScorer.from_preset("trend_momentum")
    df_signal = scorer.score_and_signal(df)
"""

import pandas as pd
import numpy as np
from factor_engine import FactorLibrary, parse_factor
from factor_library import get_all_factors


# ================================================================
#  预定义因子权重配置
# ================================================================

FACTOR_PRESETS = {
    # 趋势动量型 (适合小米类波段股)
    "trend_momentum": {
        "name": "趋势动量",
        "factors": {
            # 动量因子 — 正权重: 涨势越好越买
            "return_5d":    0.15,
            "return_20d":   0.10,
            "ma5_ma20_spread": 0.20,
            "ma10_ma20_spread": 0.10,
            # 趋势因子
            "ma5_cross_ma20": 0.15,
            "ma_bullish":   0.10,
            # 量价确认
            "vol_ratio":    0.10,
            "vol_up_price_up": 0.05,
            # 风控因子 — 负权重: 波动大减分
            "volatility_20d": -0.15,
        },
        "buy_threshold": 0.15,
        "sell_threshold": -0.10,
    },

    # 均值回复型 (适合高波动震荡股)
    "mean_reversion": {
        "name": "均值回复",
        "factors": {
            "return_5d":   -0.25,  # 涨多了减分
            "return_20d":  -0.15,  # 涨多了减分
            "ma5_bias":    -0.10,  # 偏离均线减分
            "ma20_bias":   -0.10,
            "volatility_20d": 0.10,  # 波动大反而有机会
            "position_20d": -0.15,  # 高位减分, 低位加分(取负=低位加分)
            "vol_ratio":    0.10,
            "down_streak":  0.05,   # 连续下跌加分(抄底)
        },
        "buy_threshold": 0.2,
        "sell_threshold": -0.1,
    },

    # A股专属 — 高动量+政策驱动+T+1适应
    "a_share": {
        "name": "A股动量",
        "factors": {
            # 动量因子 — A股政策驱动,动量效应更强
            "return_5d":    0.25,
            "return_20d":   0.20,
            "return_60d":   0.10,
            "ma5_ma20_spread":  0.20,
            "ma10_ma20_spread": 0.10,
            "ma20_ma60_spread": 0.10,
            # 趋势确认
            "ma5_cross_ma20": 0.10,
            "ma_bullish":   0.10,
            "ma_bearish":  -0.10,
            # 量价 — A股量价配合更关键
            "vol_ratio":    0.15,
            "vol_up_price_up": 0.10,
            "vol_up_price_down": -0.10,
            # 风控 — A股波动大,降低波动惩罚
            "volatility_20d": -0.05,
            "position_20d":   0.05,
        },
        "buy_threshold": 0.25,    # 比通用低,更容易触发买入
        "sell_threshold": -0.15,  # 比通用宽松,减少频繁卖出
    },

    # 均衡型 (通用)
    "balanced": {
        "name": "均衡",
        "factors": {
            "return_5d":    0.10,
            "return_20d":   0.05,
            "ma5_ma20_spread": 0.15,
            "ma20_ma60_spread": 0.10,
            "ma_bullish":   0.10,
            "ma_bearish":  -0.10,
            "vol_ratio":    0.08,
            "volatility_20d": -0.08,
            "vol_up_price_up": 0.05,
            "position_20d":  0.05,
        },
        "buy_threshold": 0.2,
        "sell_threshold": -0.15,
    },
}


# ================================================================
#  因子打分引擎
# ================================================================

class FactorScorer:
    """
    多因子打分引擎。

    流程:
    1. 计算所有因子值 (通过 FactorLibrary)
    2. 滚动窗口 Z-score 标准化 (每列独立)
    3. 加权求和 → 综合分数
    4. 分数 > buy_threshold → signal=1
       分数 < sell_threshold → signal=-1
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
    def from_preset(cls, preset_name: str = "trend_momentum"):
        """从预定义配置创建。"""
        preset = FACTOR_PRESETS.get(preset_name, FACTOR_PRESETS["balanced"])
        return cls(
            factor_weights=preset["factors"],
            buy_threshold=preset["buy_threshold"],
            sell_threshold=preset["sell_threshold"],
        )

    def _expr_for_factor(self, name: str) -> str:
        """根据因子名反查表达式 (从 factor_library 的预定义)。"""
        all_config = {}
        from factor_library import PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS, CANDLESTICK_FACTORS
        for d in [PRICE_FACTORS, MA_FACTORS, VOLUME_FACTORS, CANDLESTICK_FACTORS]:
            all_config.update(d)
        return all_config.get(name, f"${name}")

    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子原始值。"""
        return self.library.evaluate_all(df)

    def normalize(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        滚动窗口 Z-score 标准化。

        每列: z = (x - rolling_mean) / rolling_std
        """
        factor_cols = [c for c in factors_df.columns if c != "date"]
        normalized = pd.DataFrame(index=factors_df.index)
        if "date" in factors_df.columns:
            normalized["date"] = factors_df["date"]

        for col in factor_cols:
            series = factors_df[col].astype(float)
            roll_mean = series.rolling(self.norm_window, min_periods=20).mean()
            roll_std = series.rolling(self.norm_window, min_periods=20).std()
            normalized[col] = ((series - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)

        return normalized

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整打分流程: 计算 → 标准化 → 加权 → 信号。

        返回: df + score + signal + position
        """
        # 1. 计算原始因子
        factors_raw = self.compute_factors(df)

        # 2. 标准化
        factors_norm = self.normalize(factors_raw)

        # 3. 加权求和
        score = pd.Series(0.0, index=df.index)
        for factor_name, weight in self.factor_weights.items():
            if factor_name in factors_norm.columns:
                score += factors_norm[factor_name].fillna(0) * weight

        # 4. 分数平滑 (减少噪音)
        score = score.rolling(3, min_periods=1).mean()

        # 5. 转为信号
        df = df.copy()
        df["factor_score"] = score
        df["signal"] = 0
        df.loc[score > self.buy_threshold, "signal"] = 1
        df.loc[score < self.sell_threshold, "signal"] = -1

        # 6. 持仓状态
        df["position"] = np.where(df["signal"] == 1, 1,
                          np.where(df["signal"] == -1, 0, np.nan))
        df["position"] = df["position"].ffill().fillna(0).astype(int)

        # 统计
        buys = (df["signal"] == 1).sum()
        sells = (df["signal"] == -1).sum()
        avg_score = score.mean()
        print(f"[FactorScorer] 综合分数均值={avg_score:+.3f}, "
              f"BUY={buys}, SELL={sells} "
              f"(阈值: 买>{self.buy_threshold}, 卖<{self.sell_threshold})")

        return df

    def normalize_cross_sectional(self, all_factors: dict) -> dict:
        """
        截面标准化: 同一天所有股票的因子值一起排名。

        参数:
          all_factors: {symbol: DataFrame(因子值)}
        返回:
          {symbol: DataFrame(标准化因子值)}
        """
        symbols = list(all_factors.keys())
        if len(symbols) < 2:
            return all_factors  # 只有1只股票,不需要截面

        # 找到公共因子列
        factor_cols = [c for c in all_factors[symbols[0]].columns
                       if c not in ("date", "symbol")]

        # 对每个因子列,截面排名
        for col in factor_cols:
            # 收集所有股票的当前值
            values = {}
            for sym in symbols:
                df = all_factors[sym]
                if col in df.columns and len(df) > 0:
                    v = df[col].iloc[-1]
                    if not np.isnan(v):
                        values[sym] = v

            if len(values) < 2:
                continue

            # 截面标准化: (x - mean) / std
            vals = np.array(list(values.values()))
            mean = np.mean(vals)
            std = np.std(vals) if np.std(vals) > 0 else 1.0

            for sym in symbols:
                if sym in values:
                    z = (values[sym] - mean) / std
                    idx = all_factors[sym].index[-1]
                    all_factors[sym].loc[idx, col] = z

        return all_factors

    def cross_sectional_score(self, stock_data: dict) -> dict:
        """
        截面评分: 所有股票同一天打分,分数可直接比较。

        参数:
          stock_data: {symbol: DataFrame(price data)}
        返回:
          {symbol: float(score)}
        """
        # 1. 每只股票独立计算因子
        all_factors = {}
        for sym, df in stock_data.items():
            factors = self.compute_factors(df)
            if len(factors) > 0:
                # 用最近值
                last = factors.iloc[-1:].copy()
                all_factors[sym] = last

        # 2. 截面标准化
        if len(all_factors) >= 2:
            all_factors = self.normalize_cross_sectional(all_factors)

        # 3. 加权求和
        scores = {}
        for sym, factors in all_factors.items():
            if len(factors) == 0:
                scores[sym] = 0.0
                continue
            score = 0.0
            row = factors.iloc[-1]
            for factor_name, weight in self.factor_weights.items():
                if factor_name in factors.columns:
                    v = row[factor_name]
                    score += (0 if np.isnan(v) else v) * weight
            scores[sym] = score

        return scores
        """生成交易信号 (兼容现有策略接口)。"""
        return self.score(df)
