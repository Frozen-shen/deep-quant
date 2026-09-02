"""
tests/test_stats_correction.py — 多重检验校正 / 有效检验数 / 试验计数

覆盖 2026-09-02 自 archive/legacy/research_rigor.py 迁移到
stats_correction.py 的函数, 及新增的有效独立检验数/试验计数逻辑。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from stats_correction import (
    apply_fdr_correction,
    bonferroni_correction,
    count_trials,
    effective_num_tests,
    factor_correlation_matrix,
    fdr_correction,
    ic_to_pvalue,
    load_fdr_correction,
    t_summary_to_pvalue,
)


# ─────────────────────────────────────────────────────────────
# 1. 基础校正函数
# ─────────────────────────────────────────────────────────────


def test_bonferroni_basic():
    # alpha=0.05, m=4 → 阈值 0.0125
    res = bonferroni_correction([0.01, 0.02, 0.05, 0.5], alpha=0.05)
    assert res == [True, False, False, False]


def test_bonferroni_empty():
    assert bonferroni_correction([]) == []


def test_bonferroni_with_effective_m():
    # m=2 时阈值 0.025 → 0.02 也能通过
    res = bonferroni_correction([0.01, 0.02, 0.05], alpha=0.05, m=2)
    assert res == [True, True, False]


def test_fdr_classic_example():
    # m=5, alpha=0.05, p=[0.001,0.008,0.039,0.041,0.06]
    # 阈值: [0.01, 0.02, 0.03, 0.04, 0.05]
    # p_(2)=0.008<=0.02 ✓, p_(3)=0.039>0.03 ✗, ..., p_(5)=0.06>0.05 ✗
    # step-up 最大 k=2 → 前两个拒绝
    res = fdr_correction([0.001, 0.008, 0.039, 0.041, 0.06], alpha=0.05)
    assert res == [True, True, False, False, False]


def test_fdr_stepup_covers_tail():
    # step-up 特性: p_(5)=0.042<=0.05 通过 → 所有排名<=5 的都拒绝,
    # 即使 p_(3)/p_(4) 单独看超过各自阈值
    res = fdr_correction([0.001, 0.008, 0.039, 0.041, 0.042], alpha=0.05)
    assert res == [True, True, True, True, True]


def test_fdr_all_rejected():
    res = fdr_correction([0.001, 0.002, 0.003], alpha=0.05)
    assert res == [True, True, True]


def test_fdr_none_rejected():
    res = fdr_correction([0.5, 0.6, 0.7], alpha=0.05)
    assert res == [False, False, False]


def test_fdr_preserves_input_order():
    # 显著的因子在中间位置, 结果须按原顺序返回
    res = fdr_correction([0.9, 0.001, 0.8, 0.002], alpha=0.05)
    assert res == [False, True, False, True]


def test_fdr_empty():
    assert fdr_correction([]) == []


def test_fdr_stepup_property():
    # BH step-up: 若 p_(k) 通过, 所有排名 <=k 的都通过 (即使个别更小的
    # p 值数值上超过其自身阈值, 也被最大 k 覆盖)
    # p=[0.009, 0.011, 0.029], alpha=0.05, m=3 → 阈值 [0.0167,0.0333,0.05]
    # p_(1)=0.009<=0.0167 ✓, p_(2)=0.011<=0.0333 ✓, p_(3)=0.029<=0.05 ✓ → 全过
    res = fdr_correction([0.009, 0.011, 0.029], alpha=0.05)
    assert res == [True, True, True]


def test_fdr_with_effective_m_less_conservative():
    # 同样 p 值: 用 m=4 (完整) 不显著, 用 m_eff=2 显著
    pvals = [0.02, 0.03, 0.4, 0.6]
    assert fdr_correction(pvals, alpha=0.05) == [False, False, False, False]
    # m_eff=2: 阈值 [0.025, 0.05, ...] → p_(1)=0.02<=0.025 ✓ → 第一个过
    res = fdr_correction(pvals, alpha=0.05, m=2)
    assert res[0] is True


# ─────────────────────────────────────────────────────────────
# 2. IC → p 值
# ─────────────────────────────────────────────────────────────


def test_ic_to_pvalue_strong_signal():
    rng = np.random.RandomState(0)
    ic = rng.normal(0.05, 0.1, size=100)  # 均值显著非零
    p = ic_to_pvalue(ic)
    assert p < 0.001


def test_ic_to_pvalue_null():
    rng = np.random.RandomState(1)
    ic = rng.normal(0.0, 0.1, size=100)
    p = ic_to_pvalue(ic)
    assert 0.0 <= p <= 1.0
    # 零假设下不要求一定大于0.05 (随机性), 只验证合法输出


def test_ic_to_pvalue_degenerate():
    assert ic_to_pvalue(np.array([])) == 1.0
    assert ic_to_pvalue(np.array([0.1])) == 1.0          # n<2
    assert ic_to_pvalue(np.array([0.1, 0.1, 0.1])) == 1.0  # 方差=0
    # NaN 全被剔除后不足
    assert ic_to_pvalue(np.array([np.nan, 0.1, np.nan])) == 1.0


def test_t_summary_matches_ic_to_pvalue():
    # 摘要版 (ddof=0, 主流程 ic_stats 口径) 与序列版 (ddof=1) 的 t 统计量
    # 相差仅 sqrt((n-1)/n) (<1% @ n=57), p 值在常规区间内相对差异 <5%,
    # 且显著性判断一致
    rng = np.random.RandomState(2)
    ic = rng.normal(0.04, 0.12, size=57)
    p_series = ic_to_pvalue(ic)
    p_summary = t_summary_to_pvalue(np.mean(ic), np.std(ic), 57)  # ddof=0
    assert abs(p_series - p_summary) / max(p_series, 1e-12) < 0.05
    assert (p_series < 0.05) == (p_summary < 0.05)


def test_t_summary_degenerate():
    assert t_summary_to_pvalue(0.05, 0.0, 100) == 1.0
    assert t_summary_to_pvalue(0.05, 0.1, 1) == 1.0
    assert t_summary_to_pvalue(0.05, 0.1, None) == 1.0


# ─────────────────────────────────────────────────────────────
# 3. 有效独立检验数
# ─────────────────────────────────────────────────────────────


def test_effective_num_tests_independent():
    # 单位阵: 每个因子独立 → m_eff = k
    assert effective_num_tests(np.eye(10)) == 10


def test_effective_num_tests_fully_correlated():
    # 全1矩阵: 所有因子完全相关 → m_eff = 1
    corr = np.ones((8, 8))
    assert effective_num_tests(corr) == 1


def test_effective_num_tests_block_structure():
    # 两个完全独立的因子块 (块内相关0.9, 块间0): 特征值 [4.6, 4.6, 0.1×8],
    # 累计方差 92% < 95% → m_eff=3; 块间弱相关 (0.1) 时略高 (<=5)。
    # 两种情形都远小于原始因子数 10 → 相关性校正确实降低了有效检验数
    k = 10
    corr0 = np.zeros((k, k))
    corr0[:5, :5] = 0.9
    corr0[5:, 5:] = 0.9
    np.fill_diagonal(corr0, 1.0)
    m_eff0 = effective_num_tests(corr0, variance_explained=0.95)
    assert 2 <= m_eff0 <= 5  # 接近块数 2, 远小于 10

    corr1 = np.eye(k) * 0.1
    corr1[:5, :5] = 0.9
    corr1[5:, 5:] = 0.9
    np.fill_diagonal(corr1, 1.0)
    m_eff1 = effective_num_tests(corr1, variance_explained=0.95)
    assert 3 <= m_eff1 <= 5
    assert max(m_eff0, m_eff1) < k  # 相关性校正确实降低了有效检验数


def test_effective_num_tests_bounds():
    assert effective_num_tests(np.zeros((0, 0))) == 0
    assert effective_num_tests(np.array([[1.0]])) == 1
    rng = np.random.RandomState(3)
    x = rng.normal(size=(200, 6))
    corr = np.corrcoef(x.T)
    m_eff = effective_num_tests(corr)
    assert 1 <= m_eff <= 6


def test_effective_num_tests_nonfinite_defense():
    # NaN/Inf 输入会导致特征分解静默塌缩 (实测 NaN → m_eff=1, 最不保守);
    # 防御: 一律回退完整检验数 k (保守方向)
    c = np.eye(10)
    c[0, 1] = np.nan
    assert effective_num_tests(c) == 10
    c2 = np.eye(10)
    c2[1, 2] = np.inf
    assert effective_num_tests(c2) == 10


def test_factor_correlation_matrix_structure():
    # 构造两组因子: 组内高度相关, 组间独立
    rng = np.random.RandomState(4)
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    stocks = [f"s{i:03d}" for i in range(100)]
    base_a = rng.normal(size=(5, 100))
    base_b = rng.normal(size=(5, 100))
    panels = {}
    for j in range(3):  # A 组: base_a + 少量噪音
        panels[f"a{j}"] = pd.DataFrame(base_a + rng.normal(0, 0.1, (5, 100)),
                                       index=dates, columns=stocks)
    for j in range(3):  # B 组
        panels[f"b{j}"] = pd.DataFrame(base_b + rng.normal(0, 0.1, (5, 100)),
                                       index=dates, columns=stocks)

    corr, names = factor_correlation_matrix(
        panels, list(dates), ["a0", "a1", "a2", "b0", "b1", "b2"])
    assert names == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert corr.shape == (6, 6)
    # 组内高相关, 组间接近 0
    assert corr[0, 1] > 0.9
    assert abs(corr[0, 3]) < 0.3
    # 有效检验数应远小于 6
    assert effective_num_tests(corr) <= 3


def test_factor_correlation_matrix_missing_panel():
    # 缺面板/缺日期的因子不参与但保持位置 (单位阵回退)
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    stocks = [f"s{i:03d}" for i in range(50)]
    rng = np.random.RandomState(5)
    panels = {
        "x": pd.DataFrame(rng.normal(size=(3, 50)), index=dates, columns=stocks),
        "y": pd.DataFrame(rng.normal(size=(3, 50)), index=dates, columns=stocks),
    }
    corr, names = factor_correlation_matrix(
        panels, list(dates), ["x", "y", "missing"])
    assert names == ["x", "y"]
    assert corr.shape == (2, 2)


def test_factor_correlation_empty():
    corr, names = factor_correlation_matrix({}, [], [])
    assert names == []
    assert corr.shape == (0, 0)


# ─────────────────────────────────────────────────────────────
# 4. 试验计数 + 校正结果产物
# ─────────────────────────────────────────────────────────────


def test_count_trials(tmp_path):
    d = tmp_path / "experiments"
    d.mkdir()
    assert count_trials(str(d)) == 0
    for i in range(3):
        (d / f"exp_2026010{i+1}_000000_000{i}.json").write_text("{}", encoding="utf-8")
    (d / "not_an_experiment.txt").write_text("x", encoding="utf-8")
    assert count_trials(str(d)) == 3
    # 目录不存在 → 0
    assert count_trials(str(tmp_path / "nope")) == 0


def test_load_and_apply_fdr_correction(tmp_path):
    path = tmp_path / "fdr_correction_report.json"
    report = {
        "generated_at": "2026-09-02 12:00:00",
        "stable_factors": ["f1", "f2"],
        "rejected_factors": ["f3"],
        "meta": {},
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_fdr_correction(str(path))
    assert loaded["stable_factors"] == ["f1", "f2"]

    factors = [{"name": "f1"}, {"name": "f2"}, {"name": "f3"},
               {"name": "f_unknown"}]  # 不在报告中的因子: 保留 (只剔除显式拒绝的)
    kept = apply_fdr_correction(factors, loaded)
    assert sorted(f["name"] for f in kept) == ["f1", "f2", "f_unknown"]


def test_apply_fdr_correction_missing_report():
    # 报告缺失 → 不剔除任何因子 (降级为原列表)
    factors = [{"name": "f1"}, {"name": "f2"}]
    assert apply_fdr_correction(factors, None) == factors


def test_load_fdr_correction_corrupt(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid json", encoding="utf-8")
    assert load_fdr_correction(str(path)) is None
    assert load_fdr_correction(str(tmp_path / "missing.json")) is None


# ─────────────────────────────────────────────────────────────
# 5. 统计一致性 (蒙特卡洛): 空假设下 BH 控制假发现比例
# ─────────────────────────────────────────────────────────────


def test_bh_controls_fdr_under_null():
    # 100 个纯噪音检验, alpha=0.1: BH 拒绝的期望数 ~0; 即使有拒绝,
    # 重复 200 轮的平均拒绝数应远低于 10 (alpha×m)
    rng = np.random.RandomState(6)
    total_rejections = 0
    for _ in range(200):
        pvals = rng.uniform(0, 1, size=100)  # H0 下 p ~ U(0,1)
        total_rejections += sum(fdr_correction(pvals.tolist(), alpha=0.1))
    avg = total_rejections / 200
    assert avg < 1.0, f"BH 在空假设下平均拒绝 {avg} 个, 期望接近 0"
