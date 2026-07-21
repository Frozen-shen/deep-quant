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
#  日内因子 (Intraday Factors)
# ================================================================

INTRADAY_FACTORS = {
    "intraday_vol":        "日内波动率 (high/low-1)",
    "vwap_deviation":      "收盘价 vs VWAP 偏离度",
    "tail_return":         "尾盘效应 (最后30min收益)",
    "open_change":         "开盘跳空 (vs 昨日收盘)",
    "intraday_trend":      "日内趋势 (开盘→收盘)",
    "intraday_vol_ratio":  "日内量比",
}


# ================================================================
#  新增长江Alpha158因子 (Qlib参考)
# ================================================================

NEW_KLINE_FACTORS = {
    "kmid":     "($close - $open) / ($open + 0.01)",
    "klen":     "($high - $low) / ($open + 0.01)",
    "kmid2":    "($close - $open) / ($high - $low + 0.01)",
    "kup":      "($high - Max($close, $open)) / ($open + 0.01)",
    "klow":     "(Min($close, $open) - $low) / ($open + 0.01)",
    "ksft":     "(2 * $close - $high - $low) / ($open + 0.01)",
    "ksft2":    "(2 * $close - $high - $low) / ($high - $low + 0.01)",
}

NEW_ROLLING_FACTORS = {
    # RSV (KDJ前身)
    "rsv_9":  "RSV(9)",
    "rsv_14": "RSV(14)",

    # 动量方向
    "cntp_5":  "Mean($close > Ref($close, 1), 5)",
    "cntp_20": "Mean($close > Ref($close, 1), 20)",
    "cntd_5":  "Mean($close > Ref($close, 1), 5) - Mean($close < Ref($close, 1), 5)",
    "cntd_20": "Mean($close > Ref($close, 1), 20) - Mean($close < Ref($close, 1), 20)",

    # RSI-like
    "sump_14": "Mean(($close - Ref($close, 1)) * ($close > Ref($close, 1)), 14) / (Std($close, 14) + 0.01)",
    "ema_12":  "EMA($close, 12)",
    "ema_26":  "EMA($close, 26)",
    "rank_5":  "Rank($close, 5)",
    "rank_20": "Rank($close, 20)",
}

NEW_TURNOVER_FACTORS = {
    "turnover_ratio":   "$turnover / Mean($turnover, 5)",
    "turnover_ma5":     "Mean($turnover, 5) / ($turnover + 0.01)",
    "turnover_ma20":    "Mean($turnover, 20) / ($turnover + 0.01)",
    "turnover_change":  "Mean($turnover, 5) / Ref(Mean($turnover, 5), 5) - 1",
}

NEW_BOLL_FACTORS = {
    "boll_width":  "(Mean($close, 20) + 2 * Std($close, 20)) / (Mean($close, 20) - 2 * Std($close, 20) + 0.01) - 1",
    "boll_pct":   "($close - (Mean($close, 20) - 2 * Std($close, 20))) / (4 * Std($close, 20) + 0.01)",
    "macd_dif":   "EMA($close, 12) - EMA($close, 26)",
    "macd_ratio": "(EMA($close, 12) - EMA($close, 26)) / ($close + 0.01)",
}

# ================================================================
#  Phase 1扩展: 多窗口 + 量价组合 + 通道突破 (35+因子)
# ================================================================

EXPANDED_FACTORS = {
    # 多周期收益
    "return_2d":  "Ref($close, 2) / $close - 1",
    "return_3d":  "Ref($close, 3) / $close - 1",
    "return_7d":  "Ref($close, 7) / $close - 1",
    "return_15d": "Ref($close, 15) / $close - 1",
    "return_30d": "Ref($close, 30) / $close - 1",
    "return_90d": "Ref($close, 90) / $close - 1",

    # 多周期波动率
    "volatility_2d":  "Std(Ref($close, 1) / $close - 1, 2)",
    "volatility_10d": "Std(Ref($close, 1) / $close - 1, 10)",
    "volatility_30d": "Std(Ref($close, 1) / $close - 1, 30)",
    "volatility_90d": "Std(Ref($close, 1) / $close - 1, 90)",

    # 更多MA配对
    "ma3_ma10_spread":  "Mean($close, 3) / Mean($close, 10) - 1",
    "ma3_ma20_spread":  "Mean($close, 3) / Mean($close, 20) - 1",
    "ma5_ma10_spread":  "Mean($close, 5) / Mean($close, 10) - 1",
    "ma5_ma30_spread":  "Mean($close, 5) / Mean($close, 30) - 1",
    "ma10_ma30_spread": "Mean($close, 10) / Mean($close, 30) - 1",
    "ma10_ma60_spread": "Mean($close, 10) / Mean($close, 60) - 1",
    "ma30_ma60_spread": "Mean($close, 30) / Mean($close, 60) - 1",

    # Sharpe比
    "sharpe_5d":  "Mean(Ref($close, 1) / $close - 1, 5) / (Std(Ref($close, 1) / $close - 1, 5) + 0.001)",
    "sharpe_20d": "Mean(Ref($close, 1) / $close - 1, 20) / (Std(Ref($close, 1) / $close - 1, 20) + 0.001)",

    # 通道突破
    "channel_high_20": "($close - Max($high, 20)) / ($close + 0.01)",
    "channel_low_20":  "($close - Min($low, 20)) / ($close + 0.01)",
    "channel_high_60": "($close - Max($high, 60)) / ($close + 0.01)",

    # 振幅
    "amplitude_5d":  "Mean(($high - $low) / Ref($close, 1), 5)",
    "amplitude_20d": "Mean(($high - $low) / Ref($close, 1), 20)",

    # 偏度/峰度(已有Skew/Kurt算子)
    "skew_20d":  "Skew($close, 20)",
    "skew_60d":  "Skew($close, 60)",

    # 换手率全系列
    "turnover_vol":   "Std($turnover, 20) / (Mean($turnover, 20) + 0.01)",
    "turnover_max5":  "Max($turnover, 5) / (Mean($turnover, 20) + 0.01)",
    "turnover_trend": "Mean($turnover, 5) / Mean($turnover, 20) - 1",

    # 市值因子
    "market_cap":     "$close * $outstanding_share",
    "liq_ratio":      "$volume / ($outstanding_share + 1)",
    "amt_ratio_5d":   "Mean($amount, 5) / (Mean($amount, 20) + 0.01)",

    # Boll变体
    "boll_width_10":  "(Mean($close, 10) + 2 * Std($close, 10)) / (Mean($close, 10) - 2 * Std($close, 10) + 0.01) - 1",
    "boll_width_30":  "(Mean($close, 30) + 2 * Std($close, 30)) / (Mean($close, 30) - 2 * Std($close, 30) + 0.01) - 1",
    "macd_hist":      "EMA($close, 12) - EMA($close, 26) - EMA(EMA($close, 12) - EMA($close, 26), 9)",
}

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
