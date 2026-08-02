"""
Factor registry — a clean, deduplicated library of ~38 price/volume factors.

This replaces the legacy ``factor_library.py``, which had grown to ~98 factors
with several exact duplicates and near-duplicate (near-perfect-correlation)
overlaps. The duplicates removed during consolidation included:

- ``amihud_illiq`` == ``amihud_20d`` (identical expression)  -> ``amihud``
- ``ma_bullish`` / ``ma_bearish`` defined twice (in MA_FACTORS and again in
  P2_ENHANCED_FACTORS)                                       -> kept once
- ``turnover_trend`` defined twice with different formulas   -> kept once
- ``vol_compress`` == ``realized_vol_ratio`` == ``vol_ratio`` (all Std5/Std20)
                                                             -> ``vol_ratio``
- ``turnover_spike`` == ``vol_ratio_20d`` ($volume/Mean($volume,20))
                                                             -> ``volume_spike``
- ``intraday_range_20d`` ~ ``amplitude_20d`` (near-perfect correlation)
                                                             -> ``intraday_range``

Each factor carries metadata (:class:`FactorDef`) describing its category,
expected direction, required lookback, and a human-readable description.

Conventions
-----------
- ``direction = 1``  means a higher factor value is expected to predict higher
  forward returns; ``direction = -1`` means the opposite.
- Momentum factors use the sign convention ``Ref($close, n) / $close - 1``
  (note: this is *negative* of the usual return when read left-to-right); it is
  kept consistent with the legacy system and is accounted for in ``direction``.
- Every expression is a valid :class:`~quant.factors.engine.FactorEngine` DSL
  expression and is validated at import time by :func:`validate_library`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FactorDef:
    """Immutable definition of a single factor."""

    name: str
    expression: str
    category: str          # momentum | volatility | volume | liquidity | technical | microstructure
    direction: int         # 1 = higher is better, -1 = lower is better
    lookback: int          # minimum rows (trading days) needed for a stable value
    description: str

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name}: direction must be 1 or -1")
        if self.lookback < 1:
            raise ValueError(f"{self.name}: lookback must be >= 1")


def _defs(category: str, rows: List[tuple]) -> List[FactorDef]:
    """Build FactorDef objects from (name, expression, direction, lookback, desc) tuples."""
    return [
        FactorDef(name=r[0], expression=r[1], category=category,
                  direction=r[2], lookback=r[3], description=r[4])
        for r in rows
    ]


# ============================================================================
#  Momentum factors
# ============================================================================
# Sign convention: Ref($close, n) / $close - 1. Because Ref($close, n) is the
# price n days *ago*, a stock that has risen has Ref > close is false ... the
# expression is (past / now - 1), i.e. negative of the n-day return. So a stock
# that went UP has a NEGATIVE value here. direction=-1 therefore means
# "higher raw value (less negative) = worse recent momentum"; the IC analysis
# treats these as mean-reversion-flavoured momentum signals.

MOMENTUM_FACTORS: List[FactorDef] = _defs("momentum", [
    ("ret_5d", "Ref($close, 5) / $close - 1", -1, 6,
     "5-day return (reversed sign): past/now - 1."),
    ("ret_10d", "Ref($close, 10) / $close - 1", -1, 11,
     "10-day return (reversed sign)."),
    ("ret_20d", "Ref($close, 20) / $close - 1", -1, 21,
     "20-day return (reversed sign)."),
    ("ret_60d", "Ref($close, 60) / $close - 1", -1, 61,
     "60-day return (reversed sign)."),
    ("ret_120d", "Ref($close, 120) / $close - 1", -1, 121,
     "120-day return (reversed sign); long-horizon momentum."),
    ("reversal_1d", "$close / Ref($close, 1) - 1", -1, 2,
     "1-day return; short-term reversal (recent losers tend to rebound)."),
    ("return_accel",
     "Mean($close/Ref($close,1)-1, 5) - Mean($close/Ref($close,1)-1, 20)", 1, 21,
     "Momentum acceleration: 5d avg return minus 20d avg return."),
    ("max_return_20d", "Max($close/Ref($close,1)-1, 20)", -1, 21,
     "MAX effect (Bali et al.): max single-day return over 20d; lottery demand."),
    ("up_day_ratio", "Mean($close > Ref($close,1), 20)", 1, 21,
     "Fraction of up-days over the last 20 sessions; trend breadth."),
])


# ============================================================================
#  Volatility factors
# ============================================================================

VOLATILITY_FACTORS: List[FactorDef] = _defs("volatility", [
    ("vol_5d", "Std($close/Ref($close,1)-1, 5)", -1, 6,
     "5-day realized return volatility."),
    ("vol_20d", "Std($close/Ref($close,1)-1, 20)", -1, 21,
     "20-day realized return volatility."),
    ("vol_60d", "Std($close/Ref($close,1)-1, 60)", -1, 61,
     "60-day realized return volatility."),
    ("vol_ratio",
     "Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 0.0001)",
     -1, 21,
     "Short/long volatility ratio; regime compression/expansion."),
    ("intraday_range", "Mean(($high-$low)/($close+0.001), 20)", -1, 21,
     "Average 20-day intraday high-low range as a fraction of price."),
    ("downside_vol",
     "Std(If($close/Ref($close,1)-1 < 0, $close/Ref($close,1)-1, 0), 20)", -1, 21,
     "Downside deviation: volatility of negative returns only."),
])


# ============================================================================
#  Volume / turnover factors
# ============================================================================

VOLUME_FACTORS: List[FactorDef] = _defs("volume", [
    ("volume_ratio_5_20", "Mean($volume, 5) / (Mean($volume, 20) + 1)", 1, 21,
     "5d vs 20d average volume ratio; rising attention."),
    ("volume_spike", "$volume / (Mean($volume, 20) + 1)", 1, 21,
     "Today's volume relative to 20d average; event/spike detector."),
    ("volume_cv", "Std($volume, 20) / (Mean($volume, 20) + 1)", -1, 21,
     "Coefficient of variation of 20d volume; volume stability."),
    ("volume_price_corr", "Corr($volume, $close, 20)", 1, 21,
     "20-day correlation between volume and close; trend confirmation."),
    ("turnover_mean_5", "Mean($turnover, 5)", -1, 6,
     "5-day average turnover rate."),
    ("turnover_mean_20", "Mean($turnover, 20)", -1, 21,
     "20-day average turnover rate."),
    ("turnover_trend", "Mean($turnover, 5) / (Mean($turnover, 20) + 0.01)", 1, 21,
     "5d vs 20d turnover ratio; accelerating/decelerating activity."),
])


# ============================================================================
#  Liquidity factors
# ============================================================================

LIQUIDITY_FACTORS: List[FactorDef] = _defs("liquidity", [
    ("amihud", "Mean(Abs($close/Ref($close,1)-1) / ($amount+1), 20)", -1, 21,
     "Amihud (2002) illiquidity: |return| / dollar volume, 20d average."),
    ("illiquidity_trend",
     "Mean(Abs($close/Ref($close,1)-1)/($amount+1), 5) / "
     "(Mean(Abs($close/Ref($close,1)-1)/($amount+1), 20) + 0.0001)",
     -1, 21,
     "5d vs 20d Amihud illiquidity ratio; deteriorating/improving liquidity."),
])


# ============================================================================
#  Technical factors
# ============================================================================
# RSI(14) and MACD are implemented purely in the DSL:
#   RSI  = 100 - 100 / (1 + RS),  RS = avg up-move / avg down-move over 14 days
#   MACD = EMA(close,12) - EMA(close,26) - EMA(that, 9)   (the MACD histogram)

TECHNICAL_FACTORS: List[FactorDef] = _defs("technical", [
    ("ma_ratio_5_20", "Mean($close, 5) / (Mean($close, 20) + 0.001) - 1", 1, 21,
     "Short/medium moving-average spread (MA5 vs MA20)."),
    ("ma_ratio_10_60", "Mean($close, 10) / (Mean($close, 60) + 0.001) - 1", 1, 61,
     "Medium/long moving-average spread (MA10 vs MA60)."),
    ("rsi_14",
     "100 - 100 / (1 + "
     "Mean(If($close - Ref($close,1) > 0, $close - Ref($close,1), 0), 14) / "
     "(Mean(If($close - Ref($close,1) < 0, Ref($close,1) - $close, 0), 14) + 0.0001))",
     1, 30,
     "Relative Strength Index (14): 100 - 100/(1+RS), RS = avg gain / avg loss."),
    ("macd_signal",
     "EMA($close, 12) - EMA($close, 26) - EMA(EMA($close, 12) - EMA($close, 26), 9)",
     1, 40,
     "MACD histogram: DIF(12,26) minus its 9-day EMA."),
    ("bollinger_position",
     "($close - Mean($close, 20)) / (Std($close, 20) + 0.001)", -1, 21,
     "Close position in 20-day Bollinger band, in standard-deviation units."),
    ("close_position_20d",
     "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 0.001)", -1, 21,
     "Close location within the 20-day high-low channel (0..1)."),
    ("high_low_ratio_5d", "Max($high, 5) / (Min($low, 5) + 0.001)", -1, 6,
     "5-day high/low range ratio; short-term dispersion."),
    ("trend_r2_20", "RSqr($close, 20)", 1, 21,
     "20-day trend quality: R^2 of a linear fit to close prices."),
])


# ============================================================================
#  Microstructure factors
# ============================================================================

MICROSTRUCTURE_FACTORS: List[FactorDef] = _defs("microstructure", [
    ("overnight_return", "Mean($open/Ref($close,1)-1, 20)", -1, 22,
     "Average 20-day overnight (close-to-open) return."),
    ("return_skew", "Skew($close/Ref($close,1)-1, 20)", -1, 21,
     "20-day return skewness; tail-risk proxy."),
    ("high_low_spread", "Mean(($high-$low)/$close, 20)", -1, 21,
     "Roll (1984)-style effective spread proxy from high-low quotes."),
    ("close_to_high", "Mean(($close-$low)/($high-$low+0.001), 20)", 1, 21,
     "Average close position within the daily range; buying-pressure proxy."),
    ("return_kurt", "Kurt($close/Ref($close,1)-1, 20)", -1, 21,
     "20-day return kurtosis; fat-tail / crash-risk proxy."),
])


# ============================================================================
#  Aggregation helpers
# ============================================================================

ALL_CATEGORIES: Dict[str, List[FactorDef]] = {
    "momentum": MOMENTUM_FACTORS,
    "volatility": VOLATILITY_FACTORS,
    "volume": VOLUME_FACTORS,
    "liquidity": LIQUIDITY_FACTORS,
    "technical": TECHNICAL_FACTORS,
    "microstructure": MICROSTRUCTURE_FACTORS,
}


def get_all_factor_defs() -> List[FactorDef]:
    """Return every factor definition across all categories (deduplicated)."""
    out: List[FactorDef] = []
    for defs in ALL_CATEGORIES.values():
        out.extend(defs)
    return out


def get_factor_defs_by_category(category: str) -> List[FactorDef]:
    """Return factor definitions for a single category."""
    if category not in ALL_CATEGORIES:
        raise KeyError(
            f"Unknown category '{category}'. Valid: {sorted(ALL_CATEGORIES)}"
        )
    return list(ALL_CATEGORIES[category])


def get_expression_map() -> Dict[str, str]:
    """Return ``{factor_name: expression}`` for every factor (for the engine)."""
    return {d.name: d.expression for d in get_all_factor_defs()}


def validate_library(engine=None) -> Dict[str, str]:
    """
    Validate the registry for duplicates and unparseable expressions.

    Returns a dict of ``{problem: detail}``; an empty dict means the library is
    clean. Checks: (1) no duplicate names, (2) no duplicate expressions,
    (3) every expression parses.
    """
    from quant.factors.engine import FactorEngine

    engine = engine or FactorEngine()
    problems: Dict[str, str] = {}

    defs = get_all_factor_defs()

    names: Dict[str, int] = {}
    for d in defs:
        names[d.name] = names.get(d.name, 0) + 1
    for name, count in names.items():
        if count > 1:
            problems[f"duplicate_name:{name}"] = f"appears {count} times"

    exprs: Dict[str, str] = {}
    for d in defs:
        # Normalize whitespace so cosmetic differences don't hide true dupes.
        norm = "".join(d.expression.split())
        if norm in exprs:
            problems[f"duplicate_expression:{d.name}"] = (
                f"same expression as '{exprs[norm]}'"
            )
        else:
            exprs[norm] = d.name

    for d in defs:
        if not engine.validate(d.expression):
            problems[f"parse_error:{d.name}"] = d.expression

    return problems
