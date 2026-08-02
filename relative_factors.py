"""
Market-relative factors requiring index data.
Computed separately from per-stock expression factors.

Factors:
  rel_mom_20d   - stock 20d return minus index 20d return
  rel_mom_60d   - stock 60d return minus index 60d return
  true_beta     - CAPM beta (OLS slope of stock_ret vs index_ret, 60d)
  idio_vol      - idiosyncratic volatility (std of OLS residuals, 60d)
  rel_strength  - stock return_20d / index return_20d (ratio)
  max_dd_60d    - maximum drawdown over trailing 60 days
  downside_vol  - downside deviation (std of negative returns, 20d)
  sortino_20d   - mean(daily_ret) / downside_vol over 20 days

Usage:
  from relative_factors import compute_relative_factors, compute_relative_factors_batch
  result = compute_relative_factors(stock_df, pd.Timestamp('2025-06-30'))
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "cache", "index_csi1000.parquet")

_index_cache = None


def _load_index() -> pd.DataFrame:
    """Load and cache index data (CSI1000 by default)."""
    global _index_cache
    if _index_cache is None:
        df = pd.read_parquet(INDEX_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        _index_cache = df
    return _index_cache


def compute_relative_factors(stock_df: pd.DataFrame, as_of_date, lookback: int = 60) -> dict:
    """
    Compute market-relative factors for a single stock.

    Args:
        stock_df: DataFrame with columns [date, open, high, low, close, volume]
                  date column can be datetime or will be converted.
        as_of_date: pd.Timestamp - the date to compute factors for
        lookback: rolling window for beta/vol (default 60)

    Returns:
        dict of {factor_name: float_value}, or empty dict if insufficient data.
    """
    as_of = pd.Timestamp(as_of_date)
    index_df = _load_index()

    # Prepare stock data
    sdf = stock_df.copy()
    sdf["date"] = pd.to_datetime(sdf["date"])
    sdf = sdf.set_index("date").sort_index()

    # Truncate to as_of_date
    sdf = sdf.loc[:as_of]
    idf = index_df.loc[:as_of]

    if len(sdf) < lookback + 1 or len(idf) < lookback + 1:
        return {}

    # Daily returns
    stock_close = sdf["close"]
    index_close = idf["close"]

    stock_ret = stock_close.pct_change().dropna()
    index_ret = index_close.pct_change().dropna()

    # Align by date (inner join on trading days)
    aligned = pd.DataFrame({
        "stock_ret": stock_ret,
        "index_ret": index_ret,
    }).dropna()

    if len(aligned) < lookback:
        return {}

    # Use trailing `lookback` days for regression-based factors
    window = aligned.iloc[-lookback:]
    s_ret = window["stock_ret"].values
    i_ret = window["index_ret"].values

    results = {}

    # --- Relative Momentum ---
    # rel_mom_20d: stock 20d return - index 20d return
    if len(stock_close) > 20 and len(index_close) > 20:
        s_ret_20 = stock_close.iloc[-1] / stock_close.iloc[-21] - 1
        i_ret_20 = index_close.iloc[-1] / index_close.iloc[-21] - 1
        results["rel_mom_20d"] = float(s_ret_20 - i_ret_20)

    # rel_mom_60d: stock 60d return - index 60d return
    if len(stock_close) > 60 and len(index_close) > 60:
        s_ret_60 = stock_close.iloc[-1] / stock_close.iloc[-61] - 1
        i_ret_60 = index_close.iloc[-1] / index_close.iloc[-61] - 1
        results["rel_mom_60d"] = float(s_ret_60 - i_ret_60)

    # --- CAPM Beta & Idiosyncratic Vol ---
    # OLS: stock_ret = alpha + beta * index_ret + epsilon
    if len(s_ret) >= 30:  # need minimum observations
        x = i_ret
        y = s_ret
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx > 1e-12:
            beta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
            alpha = y_mean - beta * x_mean
            residuals = y - (alpha + beta * x)
            results["true_beta"] = float(beta)
            results["idio_vol"] = float(np.std(residuals, ddof=1))

    # --- Relative Strength ---
    # rel_strength = stock_ret_20d / index_ret_20d
    if "rel_mom_20d" in results and len(stock_close) > 20 and len(index_close) > 20:
        s_ret_20 = stock_close.iloc[-1] / stock_close.iloc[-21] - 1
        i_ret_20 = index_close.iloc[-1] / index_close.iloc[-21] - 1
        if abs(i_ret_20) > 1e-8:
            results["rel_strength"] = float(s_ret_20 / i_ret_20)

    # --- Maximum Drawdown (60d) ---
    if len(stock_close) >= 60:
        prices = stock_close.iloc[-60:].values
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices / running_max) - 1.0
        results["max_dd_60d"] = float(np.min(drawdown))

    # --- Downside Vol & Sortino (20d) ---
    # Use trailing 20 days of daily returns
    if len(aligned) >= 20:
        recent_ret = window["stock_ret"].iloc[-20:].values
        neg_ret = recent_ret[recent_ret < 0]
        if len(neg_ret) >= 3:
            downside_vol = float(np.std(neg_ret, ddof=1))
            results["downside_vol"] = downside_vol
            mean_ret = float(np.mean(recent_ret))
            if downside_vol > 1e-10:
                results["sortino_20d"] = mean_ret / downside_vol

    return results


def compute_relative_factors_batch(all_data: dict, as_of_date, lookback: int = 60) -> dict:
    """
    Compute relative factors for all stocks.

    Args:
        all_data: {symbol: DataFrame} - stock price data
        as_of_date: pd.Timestamp or date-like
        lookback: rolling window

    Returns:
        {symbol: {factor_name: value}} for stocks with sufficient data.
    """
    results = {}
    for symbol, df in all_data.items():
        factors = compute_relative_factors(df, as_of_date, lookback=lookback)
        if factors:
            results[symbol] = factors
    return results


def get_relative_factor_names() -> list:
    """Return list of all relative factor names."""
    return [
        "rel_mom_20d",
        "rel_mom_60d",
        "true_beta",
        "idio_vol",
        "rel_strength",
        "max_dd_60d",
        "downside_vol",
        "sortino_20d",
    ]
