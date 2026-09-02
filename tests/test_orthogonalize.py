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


def test_orthogonalize_no_cross_date_leakage():
    """逐日正交化: 第 1 天的输出只由第 1 天数据决定。

    构造 2 天面板, 两天的横截面相关符号相反 (第 1 天正相关, 第 2 天负相关)。
    若系数用跨日展平数据一次性估计 (旧实现的展平回归), 第 2 天数据会污染
    第 1 天的投影系数; 逐日估计则修改第 2 天输入后第 1 天输出完全不变。
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=2)
    syms = [f"S{i}" for i in range(30)]
    n = len(syms)

    f1_vals = rng.normal(0, 1, (2, n))
    f2_vals = np.empty((2, n))
    f2_vals[0] = f1_vals[0] + rng.normal(0, 0.1, n)   # 第 1 天: 正相关
    f2_vals[1] = -f1_vals[1] + rng.normal(0, 0.1, n)  # 第 2 天: 负相关

    f1 = pd.DataFrame(f1_vals, index=dates, columns=syms).astype(np.float32)
    f2 = pd.DataFrame(f2_vals, index=dates, columns=syms).astype(np.float32)

    out = orthogonalize_panels({"f1": f1, "f2": f2}, ["f1", "f2"])
    day1_f2 = out["f2"].to_numpy()[0].copy()
    day1_f1 = out["f1"].to_numpy()[0].copy()

    # 大幅篡改第 2 天输入: 第 1 天输出必须完全不变
    f2_mod = f2.copy()
    f2_mod.iloc[1] = f2_mod.iloc[1] * 100.0 + 5.0
    out_mod = orthogonalize_panels({"f1": f1, "f2": f2_mod}, ["f1", "f2"])

    np.testing.assert_allclose(
        out_mod["f2"].to_numpy()[0], day1_f2, rtol=0, atol=1e-6,
        err_msg="第 1 天正交化结果受第 2 天数据影响 → 存在跨日期泄露")
    np.testing.assert_allclose(out_mod["f1"].to_numpy()[0], day1_f1, rtol=0, atol=1e-6)

    # 第 1 天残差应与第 1 天 f1 横截面近似正交 (逐日回归的直接效果)
    r1 = np.corrcoef(day1_f1, day1_f2)[0, 1]
    assert abs(r1) < 1e-3, f"第 1 天残差与 f1 相关性 {r1:.4f} 未去除"


def test_orthogonalize_keeps_first_factor():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=60)
    syms = [f"S{i}" for i in range(20)]
    f1 = pd.DataFrame(rng.normal(0, 1, (len(dates), 20)), index=dates, columns=syms).astype(np.float32)
    panels = {"f1": f1}
    out = orthogonalize_panels(panels, ["f1"], method="gs")
    assert np.allclose(out["f1"].to_numpy(), f1.to_numpy(), atol=1e-6)
