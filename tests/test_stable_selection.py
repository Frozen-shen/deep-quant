"""稳定因子双门槛筛选 (select_stable_factors) 单元测试。

2026-09-02: 多重检验校正接入后, 稳定因子 = (原门槛: ≥3/5 折显著+方向一致)
∩ (BH 命中折数 ≥3/5), 按 |中位数ICIR| 截断到上限。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "active"))


def _mk(hits, icirs, bh_hits=None):
    """构造 core_names/factor_hits/factor_icirs/factor_bh_hits。"""
    names = sorted(hits.keys())
    return (names, hits, icirs, bh_hits)


def test_original_gate_only():
    # 无 BH 信息 (factor_bh_hits=None) → 行为与 2026-09-02 前完全一致
    from run_walkforward_backtest import select_stable_factors
    names, hits, icirs, _ = _mk(
        {"f_good": 4, "f_weak": 2, "f_mixed": 3},
        {"f_good": [0.3, 0.28, 0.32, 0.25, 0.1],
         "f_weak": [0.2, 0.1, 0.0, 0.0, 0.0],
         "f_mixed": [0.3, -0.3, 0.2, -0.2, 0.0]})  # 方向不一致 (2正2负)
    stable, stable_icir, cand = select_stable_factors(
        names, hits, icirs, factor_bh_hits=None,
        min_hits=3, icir_min=0.05, max_factors=50)
    assert stable == ["f_good"]  # f_weak 命中不足, f_mixed 方向不一致


def test_bh_gate_removes_factor():
    # f_lucky 过原门槛但 BH 命中不足 → 双门槛剔除
    from run_walkforward_backtest import select_stable_factors
    names, hits, icirs, bh = _mk(
        {"f_lucky": 3, "f_solid": 5},
        {"f_lucky": [0.2, 0.15, 0.18, 0.0, 0.0],
         "f_solid": [0.4, 0.35, 0.38, 0.3, 0.33]},
        bh_hits={"f_lucky": 1, "f_solid": 5})
    stable, _, cand = select_stable_factors(
        names, hits, icirs, factor_bh_hits=bh,
        min_hits=3, icir_min=0.05, max_factors=50)
    assert stable == ["f_solid"]
    assert "f_lucky" not in cand  # 双门槛候选集已剔除
    # 无 BH 时的完整候选集仍含 f_lucky (主流程用它做前后对比)
    _, _, cand_pre = select_stable_factors(
        names, hits, icirs, factor_bh_hits=None,
        min_hits=3, icir_min=0.05, max_factors=50)
    assert "f_lucky" in cand_pre


def test_bh_none_keeps_all_original_candidates():
    from run_walkforward_backtest import select_stable_factors
    names, hits, icirs, bh = _mk(
        {"a": 3, "b": 3},
        {"a": [0.2, 0.1, 0.15, 0.0, 0.0],
         "b": [0.3, 0.25, 0.2, 0.0, 0.0]},
        bh_hits={"a": 0, "b": 3})
    stable_none, _, _ = select_stable_factors(
        names, hits, icirs, factor_bh_hits=None,
        min_hits=3, icir_min=0.05, max_factors=50)
    stable_bh, _, _ = select_stable_factors(
        names, hits, icirs, factor_bh_hits=bh,
        min_hits=3, icir_min=0.05, max_factors=50)
    assert sorted(stable_none) == ["a", "b"]
    assert stable_bh == ["b"]


def test_max_factors_truncation_by_abs_median():
    from run_walkforward_backtest import select_stable_factors
    names, hits, icirs, _ = _mk(
        {"a": 5, "b": 5, "c": 5},
        {"a": [0.1, 0.1, 0.1, 0.1, 0.1],
         "b": [-0.5, -0.5, -0.5, -0.5, -0.5],
         "c": [0.3, 0.3, 0.3, 0.3, 0.3]})
    stable, stable_icir, _ = select_stable_factors(
        names, hits, icirs, factor_bh_hits=None,
        min_hits=3, icir_min=0.05, max_factors=2)
    # 按 |中位数| 排序: b(0.5) > c(0.3) > a(0.1) → 取前2
    assert stable == ["b", "c"]
    assert stable_icir["b"] < 0  # 符号保留


def test_null_override_icir_min():
    from run_walkforward_backtest import select_stable_factors
    names, hits, icirs, _ = _mk(
        {"low": 5, "high": 5},
        {"low": [0.08, 0.07, 0.09, 0.06, 0.08],
         "high": [0.4, 0.4, 0.4, 0.4, 0.4]})
    stable_base, _, _ = select_stable_factors(
        names, hits, icirs, min_hits=3, icir_min=0.05, max_factors=50)
    stable_strict, _, _ = select_stable_factors(
        names, hits, icirs, min_hits=3, icir_min=0.05, max_factors=50,
        null_override={"icir_min": 0.2})
    assert sorted(stable_base) == ["high", "low"]
    assert stable_strict == ["high"]


def test_empty_inputs():
    from run_walkforward_backtest import select_stable_factors
    stable, stable_icir, cand = select_stable_factors(
        [], {}, {}, factor_bh_hits=None)
    assert stable == [] and stable_icir == {} and cand == {}


def test_permute_seed_changes_labels_but_not_shape():
    # compute_icir_weights 的置换路径: 相同种子可复现, 不同种子结果不同,
    # 置换不改变观测数 (n_obs) 与因子集
    import numpy as np
    import pandas as pd
    from run_walkforward_backtest import (build_calendar, build_close_panel,
                                          compute_icir_weights)
    rng = np.random.RandomState(0)
    dates = pd.bdate_range("2020-01-01", periods=200)
    syms = [f"s{i:03d}" for i in range(60)]
    all_data = {}
    px = 10 + np.cumsum(rng.normal(0, 0.2, (200, 60)), axis=0)
    for j, s in enumerate(syms):
        all_data[s] = pd.DataFrame({
            "date": dates, "open": px[:, j], "high": px[:, j] * 1.01,
            "low": px[:, j] * 0.99, "close": px[:, j],
            "volume": 1e6, "amount": 1e7})
    calendar = build_calendar(all_data, min_coverage=30)
    close_panel = build_close_panel(all_data, calendar)
    cal_idx = {d: i for i, d in enumerate(calendar)}
    # 两个因子: 一个与未来收益强相关 (加噪, 避免 IC 恒=1 → std=0), 一个纯噪音
    from run_walkforward_backtest import LABEL_HORIZON
    factor_panels = {}
    fvals = np.zeros((len(calendar), 60))
    for i in range(len(calendar) - LABEL_HORIZON):
        fvals[i] = (px[i + LABEL_HORIZON] / px[i] - 1
                    + rng.normal(0, 0.3, 60))
    factor_panels["perfect"] = pd.DataFrame(
        fvals, index=calendar, columns=syms)
    factor_panels["noise"] = pd.DataFrame(
        rng.normal(size=(len(calendar), 60)), index=calendar, columns=syms)
    t_date = calendar[-5]
    tr_end = str(calendar[-10].date())
    tr_start = str(calendar[0].date())

    _, stats_real = compute_icir_weights(
        factor_panels, close_panel, calendar, cal_idx, t_date,
        ["perfect", "noise"], train_start=tr_start, train_end=tr_end)
    _, stats_p1 = compute_icir_weights(
        factor_panels, close_panel, calendar, cal_idx, t_date,
        ["perfect", "noise"], train_start=tr_start, train_end=tr_end,
        permute_seed=1)
    _, stats_p1b = compute_icir_weights(
        factor_panels, close_panel, calendar, cal_idx, t_date,
        ["perfect", "noise"], train_start=tr_start, train_end=tr_end,
        permute_seed=1)
    _, stats_p2 = compute_icir_weights(
        factor_panels, close_panel, calendar, cal_idx, t_date,
        ["perfect", "noise"], train_start=tr_start, train_end=tr_end,
        permute_seed=2)

    # 真实: perfect 因子 ICIR 很高
    assert abs(stats_real["perfect"]["icir"]) > 0.5
    # 置换后观测数不变
    assert stats_p1["perfect"]["n_obs"] == stats_real["perfect"]["n_obs"]
    # 同种子可复现
    assert stats_p1["perfect"]["icir"] == stats_p1b["perfect"]["icir"]
    # 不同种子结果不同 (概率 1)
    assert stats_p1["perfect"]["icir"] != stats_p2["perfect"]["icir"]
    # 置换破坏了 perfect 因子的预测力: |ICIR| 大幅下降
    assert abs(stats_p1["perfect"]["icir"]) < abs(stats_real["perfect"]["icir"]) * 0.5
