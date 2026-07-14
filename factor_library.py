"""
预定义因子集 — 参考 Qlib Alpha158 / Alpha360 的因子配置

用法:
    from factor_library import get_alpha_factors, get_candlestick_factors
    lib = get_alpha_factors()
    df_factors = lib.evaluate_all(df)
"""

from factor_engine import FactorLibrary, parse_factor

# ================================================================
#  价格因子 (Price Factors)
# ================================================================

PRICE_FACTORS = {
    # 多周期收益率
    "return_1d":  "Ref($close, 1) / $close - 1",
    "return_5d":  "Ref($close, 5) / $close - 1",
    "return_10d": "Ref($close, 10) / $close - 1",
    "return_20d": "Ref($close, 20) / $close - 1",
    "return_60d": "Ref($close, 60) / $close - 1",

    # 多周期波动率
    "volatility_5d":  "Std(Ref($close, 1) / $close - 1, 5)",
    "volatility_20d": "Std(Ref($close, 1) / $close - 1, 20)",
    "volatility_60d": "Std(Ref($close, 1) / $close - 1, 60)",

    # 价格位置 (相对于N日高低点)
    "position_20d": "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 0.01)",
    "position_60d": "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 0.01)",
}


# ================================================================
#  均线因子 (MA Factors)
# ================================================================

MA_FACTORS = {
    # 均线偏离度
    "ma5_bias":   "Mean($close, 5) / $close - 1",
    "ma10_bias":  "Mean($close, 10) / $close - 1",
    "ma20_bias":  "Mean($close, 20) / $close - 1",
    "ma60_bias":  "Mean($close, 60) / $close - 1",

    # 均线间距
    "ma5_ma20_spread":    "Mean($close, 5) / Mean($close, 20) - 1",
    "ma10_ma20_spread":   "Mean($close, 10) / Mean($close, 20) - 1",
    "ma20_ma60_spread":   "Mean($close, 20) / Mean($close, 60) - 1",

    # 交叉信号
    "ma5_cross_ma20":     "Cross(Mean($close, 5), Mean($close, 20))",
    "ma10_cross_ma20":    "Cross(Mean($close, 10), Mean($close, 20))",
    "ma5_cross_ma60":     "Cross(Mean($close, 5), Mean($close, 60))",

    # 均线排列
    "ma_bullish":  "Mean($close, 5) > Mean($close, 10) > Mean($close, 20)",
    "ma_bearish":  "Mean($close, 5) < Mean($close, 10) < Mean($close, 20)",
}


# ================================================================
#  量价因子 (Volume Factors)
# ================================================================

VOLUME_FACTORS = {
    "vol_ratio":         "$volume / Mean($volume, 5)",
    "vol_ratio_20d":     "$volume / Mean($volume, 20)",
    "vol_change_5d":     "Mean($volume, 5) / Ref(Mean($volume, 5), 5) - 1",
    "amount_ratio":      "$amount / Mean($amount, 5)",

    # 量价配合
    "vol_up_price_up":   "($volume > Mean($volume, 5)) * ($close > Ref($close, 1))",
    "vol_up_price_down": "($volume > Mean($volume, 5)) * ($close < Ref($close, 1))",
}


# ================================================================
#  K线形态因子 (Candlestick Factors)
# ================================================================

CANDLESTICK_FACTORS = {
    # 实体 / 影线
    "body_ratio":    "($close - $open) / ($open + 0.01)",
    "upper_shadow":  "($high - Max($close, $open)) / (Max($close, $open) - Min($close, $open) + 0.01)",
    "lower_shadow":  "(Min($close, $open) - $low) / (Max($close, $open) - Min($close, $open) + 0.01)",
    "k_len":         "($high - $low) / ($open + 0.01)",
    "k_mid":         "(($close - $open) / 2 + Min($close, $open)) / Ref($close, 1)",

    # 大阳线/大阴线检测
    "big_bull":      "(($close - $open) / ($open + 0.01)) > 0.05",
    "big_bear":      "(($close - $open) / ($open + 0.01)) < -0.05",
    "doji":          "(($close - $open) / ($open + 0.01)) < 0.003",

    # 连续N日方向
    "up_streak":    "Mean($close > Ref($close, 1), 5)",
    "down_streak":  "Mean($close < Ref($close, 1), 5)",
}


# ================================================================
#  便捷工厂
# ================================================================

def get_price_factors() -> FactorLibrary:
    return FactorLibrary.from_config(PRICE_FACTORS)

def get_ma_factors() -> FactorLibrary:
    return FactorLibrary.from_config(MA_FACTORS)

def get_volume_factors() -> FactorLibrary:
    return FactorLibrary.from_config(VOLUME_FACTORS)

def get_candlestick_factors() -> FactorLibrary:
    return FactorLibrary.from_config(CANDLESTICK_FACTORS)

def get_all_factors() -> FactorLibrary:
    """合并所有预定义因子。"""
    all_config = {}
    all_config.update(PRICE_FACTORS)
    all_config.update(MA_FACTORS)
    all_config.update(VOLUME_FACTORS)
    all_config.update(CANDLESTICK_FACTORS)
    return FactorLibrary.from_config(all_config)
