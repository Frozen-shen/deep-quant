"""tests/test_orthogonalize.py"""
import numpy as np
import pandas as pd
import pytest
from orthogonalize import orthogonalize_panels


def test_orthogonalize_removes_correlation():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i}" for i in range(20)]
    base = rng.normal(0, 1, (len(dates), 1))
    f1 = pd.DataFrame(np.tile(base, (1, 20)) + rng.normal(0, 0.1, (len(dates), 20)),
                      index=dates, columns=syms).astype(np.float32)
    f2 = pd.DataFrame(np.tile(base * 2, (1, 20)) + rng.normal(0, 0.1, (len(dates), 20)),
                      index=dates, columns=syms).astype(np.float32)
    panels = {"f1": f1, "f2": f2}
    out = orthogonalize_panels(panels, ["f1", "f2"], method="gs")
    # f2 正交化后与 f1 相关性 ≈ 0
    r = np.corrcoef(out["f1"].to_numpy().ravel(), out["f2"].to_numpy().ravel())[0, 1]
    assert abs(r) < 0.05


def test_orthogonalize_keeps_first_factor():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i}" for i in range(20)]
    f1 = pd.DataFrame(rng.normal(0, 1, (len(dates), 20)), index=dates, columns=syms).astype(np.float32)
    panels = {"f1": f1}
    out = orthogonalize_panels(panels, ["f1"], method="gs")
    assert np.allclose(out["f1"].to_numpy(), f1.to_numpy(), atol=1e-6)
