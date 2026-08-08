"""tests/test_pool_filter.py — 波动率分层 × regime 乘数"""
import numpy as np
import pandas as pd
import pytest

from pool_filter import vol_bucket, apply_pool_filter


def _mk_data(n_dates=80, n_stocks=9):
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    all_data = {}
    vols = [0.005] * 3 + [0.01] * 3 + [0.02] * 3  # 低/中/高各3只 (分组)
    for i, v in enumerate(vols):
        rng = np.random.default_rng(i)
        rets = rng.normal(0, v, n_dates)
        px = 10 * np.exp(np.cumsum(rets))
        all_data[f"S{i}"] = pd.DataFrame(
            {"date": dates, "open": px, "close": px, "high": px * 1.01,
             "low": px * 0.99, "volume": np.full(n_dates, 1e6),
             "amount": np.full(n_dates, 1e7)})
    return all_data, dates[-1]


def test_vol_bucket_three_tiers():
    all_data, today = _mk_data()
    scores = {f"S{i}": 1.0 for i in range(9)}
    buckets = vol_bucket(scores, all_data, today)
    assert buckets["S0"] == "low" and buckets["S1"] == "low" and buckets["S2"] == "low"
    assert buckets["S3"] == "mid" and buckets["S4"] == "mid" and buckets["S5"] == "mid"
    assert buckets["S6"] == "high" and buckets["S7"] == "high" and buckets["S8"] == "high"


def test_vol_bucket_insufficient_data_defaults_mid():
    all_data, today = _mk_data(n_dates=5)  # 波动率数据不足
    scores = {f"S{i}": 1.0 for i in range(9)}
    buckets = vol_bucket(scores, all_data, today)
    assert all(b == "mid" for b in buckets.values())


def test_apply_pool_filter_high_vol_regime():
    scores = {"a": 1.0, "b": 2.0, "c": 3.0}
    buckets = {"a": "low", "b": "mid", "c": "high"}
    mults = {"low": 1.5, "mid": 1.0, "high": 0.5}
    out = apply_pool_filter(scores, buckets, vol_pct=0.8, mults=mults)
    assert out["a"] == pytest.approx(1.5)
    assert out["b"] == pytest.approx(2.0)
    assert out["c"] == pytest.approx(1.5)
    assert out is not scores  # 不原地修改


def test_apply_pool_filter_does_not_mutate_input():
    scores = {"a": 1.0}
    buckets = {"a": "low"}
    apply_pool_filter(scores, buckets, 0.8, {"low": 1.5, "mid": 1.0, "high": 0.5})
    assert scores["a"] == 1.0
