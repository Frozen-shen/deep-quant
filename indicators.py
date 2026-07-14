"""
技术指标库 — 参考 Backtrader 122 个指标的标准实现

用法:
    from indicators import RSI, MACD, BollingerBands, ATR
    df["rsi"] = RSI(df["close"], period=14)
    macd, signal, hist = MACD(df["close"])
    upper, middle, lower = BollingerBands(df["close"])
"""

import pandas as pd
import numpy as np


def RSI(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指标 → (macd_line, signal_line, histogram)。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def BollingerBands(close: pd.Series, period: int = 20, stddev: float = 2.0):
    """布林带 → (upper, middle, lower)。"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + stddev * std
    lower = middle - stddev * std
    return upper, middle, lower


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅 (ATR)。"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均趋向指数 (ADX) — 衡量趋势强度。"""
    prev_close = close.shift(1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.Series(adx, index=close.index)


def KDJ(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 9, m1: int = 3, m2: int = 3):
    """KDJ 指标 → (K, D, J)。"""
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 (OBV)。"""
    direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    return (volume * direction).cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    一键添加所有常用指标到 DataFrame。

    需要列: close, high, low, volume
    """
    df = df.copy()

    # RSI
    df["rsi_6"] = RSI(df["close"], 6)
    df["rsi_14"] = RSI(df["close"], 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = MACD(df["close"])

    # Bollinger
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = BollingerBands(df["close"])

    # ATR
    if all(c in df.columns for c in ["high", "low"]):
        df["atr_14"] = ATR(df["high"], df["low"], df["close"], 14)

        # ADX
        df["adx_14"] = ADX(df["high"], df["low"], df["close"], 14)

        # KDJ
        df["kdj_k"], df["kdj_d"], df["kdj_j"] = KDJ(df["high"], df["low"], df["close"])

    # OBV
    if "volume" in df.columns:
        df["obv"] = OBV(df["close"], df["volume"])

    # 均线
    for p in [5, 10, 20, 60]:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()
        df[f"ma{p}_bias"] = df[f"ma_{p}"] / df["close"] - 1

    return df
