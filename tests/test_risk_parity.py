"""tests/test_risk_parity.py"""
import numpy as np
import pandas as pd
import pytest
import scripts.active.run_walkforward_backtest as rw


def _mk_data(n_dates=120, n_stocks=4):
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    all_data = {}
    rng = np.random.default_rng(5)
    # 低波动股票 (S0) 与高波动 (S3), 部分相关
    base = rng.normal(0, 0.01, n_dates)
    vols = [0.008, 0.015, 0.022, 0.030]
    for i, v in enumerate(vols):
        rets = base * (0.3 if i < 2 else 0.6) + rng.normal(0, v, n_dates)
        px = 10 * np.exp(np.cumsum(rets))
        all_data[f"S{i}"] = pd.DataFrame(
            {"date": dates, "open": px, "close": px, "high": px * 1.01,
             "low": px * 0.99, "volume": np.full(n_dates, 1e6),
             "amount": np.full(n_dates, 1e7)})
    return all_data, dates[-1]


def test_risk_parity_weights_normalized():
    all_data, today = _mk_data()
    w = rw._risk_parity_weights(all_data, ["S0", "S1", "S2", "S3"], today)
    assert w is not None
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # 高波动股票权重低于低波动
    assert w["S0"] > w["S3"]


def test_risk_parity_insufficient_data_returns_none():
    all_data, today = _mk_data(n_dates=10)
    w = rw._risk_parity_weights(all_data, ["S0", "S1", "S2", "S3"], today)
    assert w is None or all(v == 0 for v in w.values())
