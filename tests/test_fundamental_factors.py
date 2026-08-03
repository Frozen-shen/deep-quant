import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_library import FUNDAMENTAL_FACTORS

def test_fundamental_factors_exist():
    assert len(FUNDAMENTAL_FACTORS) >= 10
    assert "fund_bp" in FUNDAMENTAL_FACTORS
    assert "fund_ep" in FUNDAMENTAL_FACTORS

def test_full_auto_v5_preset():
    from factor_scorer import FactorScorer
    sc = FactorScorer.from_preset("full_auto_v5")
    assert len(sc.factor_weights) > 170
    assert any(k.startswith("fund_") for k in sc.factor_weights)


def test_neutralize_winsorize():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))
    from run_walkforward_backtest import neutralize_factor
    # >=10 个非 NaN 才参与中性化 (简报代码 m.sum() < 10 直接跳过)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0, 7.0,
                             8.0, 9.0, 10.0, 11.0]})
    out = neutralize_factor(df)
    # 100 被去极值到 MAD 范围内
    assert out.iloc[3, 0] < 10
    # z-score 后均值≈0
    assert abs(out.mean().iloc[0]) < 1e-6


def test_neutralize_nan_preserved():
    """NaN 保留且不参与统计 (去极值/z-score 均忽略 NaN)。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))
    from run_walkforward_backtest import neutralize_factor
    df = pd.DataFrame({"a": [1.0, 2.0, float("nan"), 3.0, 100.0, 4.0,
                             5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    out = neutralize_factor(df)
    assert out.iloc[2, 0] != out.iloc[2, 0]  # NaN 保留
    assert out.iloc[0, 0] == out.iloc[0, 0]  # 非 NaN 已填充为 z-score
    assert abs(out["a"].mean()) < 1e-6


def test_fundamental_panel_merge():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))
    from run_walkforward_backtest import precompute_factor_panels
    from data_cache import load
    df = load("000001")
    if df is None:
        import pytest; pytest.skip("no data")
    all_data = {"000001": df}
    needed = [pd.Timestamp("2025-06-30")]
    from factor_scorer import FactorScorer
    factor_names = list(FactorScorer.from_preset("full_auto_v5").factor_weights.keys())
    panels = precompute_factor_panels(all_data, factor_names, needed, include_fundamental=True)
    assert any(k.startswith("fund_") for k in panels)


def test_portfolio_constraints():
    """组合后置约束: 等权分仓目标权重, 单票超过 max_single_pct 时缩放到上限。

    语义 (修正简报 Step1 测试与 Step3 实现的矛盾): 函数返回调整后的目标
    权重 dict (非原始 scores) — 等权 1/n 天然满足上限时返回等权, 超过上限
    时每只缩放到 max_single (剩余仓位留现金)。测试钉住真实约束行为。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))
    from run_walkforward_backtest import apply_portfolio_constraints
    # 30只: 等权 1/30 ≈ 3.3% <= 5% 上限 → 等权天然满足
    scores = {f"s{i}": float(i) for i in range(30)}
    c = {"max_single_pct": 0.05}
    out = apply_portfolio_constraints(scores, c)
    assert len(out) == 30
    assert all(v <= 0.05 + 1e-9 for v in out.values())
    assert all(abs(v - 1.0 / 30) < 1e-9 for v in out.values())
    # 10只: 等权 10% > 5% 上限 → 每只缩放到 5% (约束触发分支)
    scores2 = {f"t{i}": float(i) for i in range(10)}
    out2 = apply_portfolio_constraints(scores2, c)
    assert len(out2) == 10
    assert all(abs(v - 0.05) < 1e-9 for v in out2.values())
    # 空输入: 原样返回空 dict
    assert apply_portfolio_constraints({}, c) == {}
