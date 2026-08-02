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

    # 均线排列 (用乘法代替链式比较, 因为 A>B>C 会被解析为 (A>B)>C 产生bug)
    "ma_bullish":  "(Mean($close, 5) > Mean($close, 10)) * (Mean($close, 10) > Mean($close, 20))",
    "ma_bearish":  "(Mean($close, 5) < Mean($close, 10)) * (Mean($close, 10) < Mean($close, 20))",
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

# ================================================================
#  分钟频因子 (5分钟K线聚合为日频, 由 minute_factors.py 计算)
#  非DSL表达式 — 需要分钟级数据, 通过 compute_minute_factors_batch() 获取
# ================================================================

MINUTE_FACTORS = {
    "min_realized_vol":      "已实现波动率 (5min returns std × √(48×252))",
    "min_realized_skew":     "已实现偏度 (5min returns skewness)",
    "min_vwap_dev":          "收盘 vs VWAP 偏离度",
    "min_tail_return":       "尾盘效应 (最后30min收益)",
    "min_open_gap":          "开盘跳空 (vs 昨日收盘)",
    "min_intraday_trend":    "日内趋势 (open→close)",
    "min_vol_concentration": "成交集中度 (Gini系数, 高=主力集中交易)",
    "min_large_order_flow":  "大单净流入代理 (高volume bar的净方向)",
    "min_am_pm_ratio":       "上下午量比 (AM/PM volume)",
    "min_close_strength":    "收盘强度 (close在日内range中的位置)",
}

# 保留旧名称兼容 (deprecated)
INTRADAY_FACTORS = MINUTE_FACTORS


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

# ================================================================
#  Phase 2扩展: 价量相关 + 趋势质量
# ================================================================

PHASE2_FACTORS = {
    # 价量相关性 (Qlib: CORR)
    "corr_pv_10":   "Corr($close, Log($volume + 1), 10)",
    "corr_pv_20":   "Corr($close, Log($volume + 1), 20)",
    # 趋势拟合度 (Qlib: RSQR)
    "rsqr_20":      "RSqr($close, 20)",
    "rsqr_60":      "RSqr($close, 60)",
}

# ================================================================
#  P2 增强因子: 动量反转 + 流动性 + 波动率regime
# ================================================================

P2_ENHANCED_FACTORS = {
    # ── 动量反转复合 ──
    # 短期反转 (1-3天): 昨日跌的今日可能涨 (A股散户效应)
    "reversal_1d":    "-(Ref($close, 1) / $close - 1)",
    "reversal_3d":    "-(Ref($close, 3) / $close - 1)",
    # 中期动量 (7-20天): 趋势延续
    "momentum_7d":    "Ref($close, 7) / $close - 1",
    "momentum_20d":   "Ref($close, 20) / $close - 1",
    # 反转-动量交叉: 短期反转+中期动量 = 高质量信号
    "rev_mom_spread": "-(Ref($close, 1) / $close - 1) + Ref($close, 20) / $close - 1",

    # ── 流动性 ──
    # Amihud illiquidity: |return| / amount — 值越大流动性越差
    "amihud_5d":   "Mean(Abs(Ref($close, 1) / $close - 1) / ($amount + 1), 5)",
    "amihud_20d":  "Mean(Abs(Ref($close, 1) / $close - 1) / ($amount + 1), 20)",
    # 换手率趋势: 近5日换手率 vs 20日均值 — 放量上涨/缩量下跌
    "turnover_trend": "Mean($turnover, 5) / (Mean($turnover, 20) + 0.01)",
    # 量价同步性: 价涨量增=健康, 价涨量缩=虚弱
    "vol_price_sync": "($close / Ref($close, 5) - 1) * ($volume / Mean($volume, 20) - 1)",

    # ── 波动率regime ──
    # 波动率变化率: 近期波动 vs 长期波动
    "vol_regime":  "Std(Ref($close, 1) / $close - 1, 10) / (Std(Ref($close, 1) / $close - 1, 60) + 0.001)",
    # 波动压缩: 布林带宽度变化
    "vol_compress": "Std($close, 5) / (Std($close, 20) + 0.01)",
    # 高低价范围: 近期高低差占比
    "range_20d":   "(Max($high, 20) - Min($low, 20)) / $close",

    # ── 均线排列质量 (已修复链式比较bug) ──
    "ma_bullish":  "(Mean($close, 5) > Mean($close, 10)) * (Mean($close, 10) > Mean($close, 20))",
    "ma_bearish":  "(Mean($close, 5) < Mean($close, 10)) * (Mean($close, 10) < Mean($close, 20))",
}

# ================================================================
#  微观结构因子 (Microstructure Factors)
# ================================================================

MICROSTRUCTURE_FACTORS = {
    "amihud_illiq":          "Mean(Abs($close/Ref($close,1)-1) / ($amount+1), 20)",
    "intraday_range_20d":    "Mean(($high-$low)/($close+0.001), 20)",
    "volume_price_corr_20d": "Corr($volume, $close, 20)",
    "turnover_spike":        "$volume / Mean($volume, 20)",
    "high_low_ratio_5d":     "Max($high, 5) / (Min($low, 5) + 0.001)",
    "close_position_20d":    "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 0.001)",
    "volume_cv_20d":         "Std($volume, 20) / (Mean($volume, 20) + 1)",
    "return_skew_20d":       "Skew($close/Ref($close,1)-1, 20)",
    "overnight_return_20d":  "Mean($open/Ref($close,1)-1, 20)",
    "realized_vol_ratio":    "Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 0.0001)",
}


# ================================================================
#  市场相对因子 (Market-Relative Factors)
#  需要指数数据, 由 relative_factors.py 计算 (非表达式因子)
# ================================================================

RELATIVE_FACTORS = {
    "rel_mom_20d":  "stock 20d return - index 20d return (relative momentum)",
    "rel_mom_60d":  "stock 60d return - index 60d return (relative momentum)",
    "true_beta":    "CAPM beta (OLS slope of stock_ret vs index_ret, 60d)",
    "idio_vol":     "idiosyncratic volatility (std of OLS residuals, 60d)",
    "rel_strength": "stock return_20d / index return_20d (ratio)",
    "max_dd_60d":   "maximum drawdown over trailing 60 days",
    "downside_vol": "downside deviation (std of negative returns, 20d)",
    "sortino_20d":  "mean(daily_ret) / downside_vol over 20 days",
}


# ================================================================
#  Alpha158 补全因子 (Qlib 价量因子, 仅用 OHLCV)
#  BETA/RSQR/RESI/IMAX/IMIN/IMXD/WVMA/CORD/SUMP/SUMN/SUMD/VMA/VSTD
#  每个家族覆盖窗口 d ∈ {5, 10, 20, 30, 60}
# ================================================================

_ALPHA158_WINDOWS = [5, 10, 20, 30, 60]

ALPHA158_FACTORS = {}

# BETA — 滚动线性回归斜率 (趋势方向与速率, 除以价格做尺度归一)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"beta_{_d}"] = f"Slope($close, {_d}) / $close"

# RSQR — 趋势拟合优度 R² (越接近1趋势越线性)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"rsqr_{_d}"] = f"RSqr($close, {_d})"

# RESI — 回归残差 (价格偏离趋势的程度, 除以价格归一)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"resi_{_d}"] = f"Resi($close, {_d}) / $close"

# IMAX — 距N日最高点的回看天数 (归一到 [0,1])
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"imax_{_d}"] = f"IdxMax($high, {_d}) / {_d}"

# IMIN — 距N日最低点的回看天数 (归一到 [0,1])
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"imin_{_d}"] = f"IdxMin($low, {_d}) / {_d}"

# IMXD — 最高点与最低点位置之差 (衡量高低点先后顺序)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"imxd_{_d}"] = f"(IdxMax($high, {_d}) - IdxMin($low, {_d})) / {_d}"

# WVMA — 成交量加权波动率 (放量波动 vs 平均放量波动)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"wvma_{_d}"] = (
        f"Std(Abs($close/Ref($close,1)-1)*$volume, {_d}) / "
        f"(Mean(Abs($close/Ref($close,1)-1)*$volume, {_d})+0.01)"
    )

# CORD — 收益率与成交量变化的相关性 (量价联动)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"cord_{_d}"] = (
        f"Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), {_d})"
    )

# SUMP / SUMN / SUMD — RSI 类 (上涨/下跌动量占比及其差)
for _d in _ALPHA158_WINDOWS:
    _denom = f"(Sum(Abs($close-Ref($close,1)), {_d})+0.01)"
    _up = f"Sum(($close-Ref($close,1))*($close>Ref($close,1)), {_d})"
    _dn = f"Sum((Ref($close,1)-$close)*($close<Ref($close,1)), {_d})"
    ALPHA158_FACTORS[f"sump_{_d}"] = f"{_up} / {_denom}"
    ALPHA158_FACTORS[f"sumn_{_d}"] = f"{_dn} / {_denom}"
    ALPHA158_FACTORS[f"sumd_{_d}"] = f"({_up} - {_dn}) / {_denom}"

# VMA — 成交量均值比 (当前量相对N日均量)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"vma_{_d}"] = f"Mean($volume, {_d}) / ($volume+1)"

# VSTD — 成交量波动比 (N日量的波动相对当前量)
for _d in _ALPHA158_WINDOWS:
    ALPHA158_FACTORS[f"vstd_{_d}"] = f"Std($volume, {_d}) / ($volume+1)"


def get_price_factors() -> FactorLibrary:
    return FactorLibrary.from_config(PRICE_FACTORS)

def get_ma_factors() -> FactorLibrary:
    return FactorLibrary.from_config(MA_FACTORS)

def get_volume_factors() -> FactorLibrary:
    return FactorLibrary.from_config(VOLUME_FACTORS)

def get_candlestick_factors() -> FactorLibrary:
    return FactorLibrary.from_config(CANDLESTICK_FACTORS)

def get_alpha158_factors() -> FactorLibrary:
    """Alpha158 补全因子 (BETA/RSQR/RESI/IMAX/IMIN/IMXD/WVMA/CORD/SUMP/SUMN/SUMD/VMA/VSTD)。"""
    return FactorLibrary.from_config(ALPHA158_FACTORS)

def get_relative_factor_names() -> list:
    """Return names of market-relative factors (computed via relative_factors.py)."""
    return list(RELATIVE_FACTORS.keys())

def get_all_factors() -> FactorLibrary:
    """合并所有预定义因子 (含 Phase2 + P2增强 + 微观结构 + Alpha158补全)。"""
    all_config = {}
    all_config.update(PRICE_FACTORS)
    all_config.update(MA_FACTORS)
    all_config.update(VOLUME_FACTORS)
    all_config.update(CANDLESTICK_FACTORS)
    all_config.update(EXPANDED_FACTORS)
    all_config.update(PHASE2_FACTORS)
    all_config.update(P2_ENHANCED_FACTORS)
    all_config.update(MICROSTRUCTURE_FACTORS)
    all_config.update(ALPHA158_FACTORS)
    return FactorLibrary.from_config(all_config)
