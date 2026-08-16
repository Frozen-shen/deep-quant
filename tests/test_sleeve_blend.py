"""多风格 sleeve 配置解析测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))


def _base_styles():
    return {
        "styles": {
            "enabled": True,
            "budgets": {"momentum": 0.25, "growth": 0.15},
            "sleeves": {
                "momentum": {"min_hits": 3,
                             "factors": ["mom_60d", "return_30d"]},
                "growth": {"min_hits": 0, "fallback_weight": 0.1,
                           "factors": ["fund_profit_growth"]},
            },
        }
    }


def test_load_styles_config_disabled_returns_none():
    from run_walkforward_backtest import load_styles_config
    assert load_styles_config({}) is None
    assert load_styles_config({"styles": {"enabled": False}}) is None


def test_load_styles_config_valid():
    from run_walkforward_backtest import load_styles_config
    assert load_styles_config(_base_styles()) is not None


def test_load_styles_config_budget_over_1_raises():
    from run_walkforward_backtest import load_styles_config
    import pytest
    cfg = _base_styles()
    cfg["styles"]["budgets"] = {"momentum": 0.8, "growth": 0.5}
    with pytest.raises(ValueError):
        load_styles_config(cfg)


def test_split_sleeve_factors():
    from run_walkforward_backtest import split_sleeve_factors
    core, sleeves = split_sleeve_factors(
        ["mom_60d", "return_30d", "fund_bp", "amihud_5d", "fund_profit_growth"],
        _base_styles())
    assert core == ["fund_bp", "amihud_5d"]
    assert set(sleeves["momentum"]) == {"mom_60d", "return_30d"}
    assert sleeves["growth"] == ["fund_profit_growth"]


def test_split_sleeve_factors_unknown_skipped():
    from run_walkforward_backtest import split_sleeve_factors
    cfg = _base_styles()
    cfg["styles"]["sleeves"]["momentum"]["factors"] = ["mom_60d", "not_a_factor"]
    core, sleeves = split_sleeve_factors(["mom_60d", "fund_bp"], cfg)
    assert sleeves["momentum"] == ["mom_60d"]
    assert core == ["fund_bp"]


def test_parse_budget_combos_valid():
    from run_walkforward_backtest import parse_budget_combos
    combos = parse_budget_combos("0.25/0.15,0.2/0.2")
    assert combos == [[("momentum", 0.25), ("growth", 0.15)],
                      [("momentum", 0.2), ("growth", 0.2)]]


def test_parse_budget_combos_errors():
    from run_walkforward_backtest import parse_budget_combos
    import pytest
    with pytest.raises(ValueError):
        parse_budget_combos("0.25")            # 元素数不足
    with pytest.raises(ValueError):
        parse_budget_combos("0.8/0.5")         # 合计 > 1
    with pytest.raises(ValueError):
        parse_budget_combos("")                # 空


# ── Task 2: score_stocks 多通道重构 ──


def _panels():
    import pandas as pd
    import numpy as np
    cal = pd.date_range("2024-01-01", periods=3, freq="B")
    syms = [f"s{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.02, (3, 10))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=cal, columns=syms)
    panels = {}
    for name in ["f1", "f2", "m1", "g1"]:
        vals = close / close.shift(1).fillna(1.0) - 1.0
        panels[name] = pd.DataFrame(
            vals.to_numpy(), index=cal, columns=syms, dtype=np.float32)
    return panels, cal


def test_score_stocks_no_sleeve_formula_unchanged():
    """无 sleeve 时与单通道旧实现数值一致。"""
    from run_walkforward_backtest import score_stocks
    panels, cal = _panels()
    t = cal[1]
    s1 = score_stocks(panels, {"f1": 1.0, "f2": -0.5}, t)
    s2 = score_stocks(panels, {"f1": 1.0, "f2": -0.5}, t, sleeve_weights=None)
    assert s1 == s2
    # 手工复算: composite = (1*z1 - 0.5*z2)/1.5
    # (np.nanstd: ddof=0, 与实现 _weighted_z_composite 一致;
    #  astype(float64): 与实现的 float64 计算路径一致, 避免 float32 舍入差)
    import numpy as np
    cross = panels["f1"].loc[t].astype(np.float64)
    z1 = (cross - cross.mean()) / np.nanstd(cross)
    cross2 = panels["f2"].loc[t].astype(np.float64)
    z2 = (cross2 - cross2.mean()) / np.nanstd(cross2)
    expect = (1.0 * z1 - 0.5 * z2) / 1.5
    for k in s1:
        assert abs(s1[k] - expect[k]) < 1e-9


def test_score_stocks_sleeve_blend_math():
    """composite = (1-Σbudget)×core + Σ budget×sleeve (各通道内部归一)。"""
    from run_walkforward_backtest import score_stocks
    import numpy as np
    panels, cal = _panels()
    t = cal[1]
    sw = [
        {"name": "momentum", "weights": {"m1": 1.0}, "budget": 0.25},
        {"name": "growth", "weights": {"g1": 1.0}, "budget": 0.15},
    ]
    s = score_stocks(panels, {"f1": 1.0}, t, sleeve_weights=sw)
    # (np.nanstd: ddof=0, 与实现 _weighted_z_composite 一致;
    #  astype(float64): 与实现的 float64 计算路径一致, 避免 float32 舍入差)
    c = panels["f1"].loc[t].astype(np.float64)
    z1 = (c - c.mean()) / np.nanstd(c)
    m = panels["m1"].loc[t].astype(np.float64)
    zm = (m - m.mean()) / np.nanstd(m)
    g = panels["g1"].loc[t].astype(np.float64)
    zg = (g - g.mean()) / np.nanstd(g)
    expect = 0.6 * z1 + 0.25 * zm + 0.15 * zg
    for k in s:
        assert abs(s[k] - expect[k]) < 1e-9, f"{k}: {s[k]} vs {expect[k]}"


# ── Task 3: sleeve_median_weights 纯函数 ──


def test_sleeve_median_weights_fallback():
    """|median|<0.02 用 fallback_weight 正号; 否则用 median ICIR。"""
    from run_walkforward_backtest import sleeve_median_weights
    icirs = {
        "momentum": {"m1": [0.05, 0.06, 0.04, 0.05, 0.05],
                     "m2": [0.01, -0.01, 0.0, 0.0, 0.0]},
        "growth": {"g1": [0.0, 0.0, 0.0, 0.0, 0.0]},
    }
    cfg = {"momentum": {"fallback_weight": 0.1},
           "growth": {"fallback_weight": 0.1}}
    out = sleeve_median_weights(icirs, cfg)
    assert abs(out["momentum"]["m1"] - 0.05) < 1e-9
    assert out["momentum"]["m2"] == 0.1   # median≈0 → fallback
    assert out["growth"]["g1"] == 0.1     # 全 0 → fallback


def test_load_styles_config_missing_budget_raises():
    """sleeve 存在但 budgets 缺键 → ValueError (提前失败, 防 fold 循环 KeyError)。"""
    from run_walkforward_backtest import load_styles_config
    import pytest
    cfg = _base_styles()
    cfg["styles"]["budgets"] = {"momentum": 0.25}  # growth sleeve 缺预算
    with pytest.raises(ValueError):
        load_styles_config(cfg)


# ── Task 4: build_extend_sleeve_weights 纯函数 ──


def test_build_extend_sleeve_weights():
    """fold_out + styles_cfg → extend 用 sleeve 列表 (空预算跳过)。"""
    from run_walkforward_backtest import build_extend_sleeve_weights
    fold_out = {"sleeve_median_weights": {
        "momentum": {"m1": 0.05, "m2": 0.1},
        "growth": {"g1": 0.1}}}
    cfg = {"budgets": {"momentum": 0.25, "growth": 0.0},
           "sleeves": {"momentum": {}, "growth": {}}}
    out = build_extend_sleeve_weights(fold_out, cfg)
    assert len(out) == 1
    assert out[0]["name"] == "momentum" and out[0]["budget"] == 0.25
    assert out[0]["weights"] == {"m1": 0.05, "m2": 0.1}


def test_build_extend_sleeve_weights_no_styles():
    from run_walkforward_backtest import build_extend_sleeve_weights
    assert build_extend_sleeve_weights({}, None) is None


# ── F2: sleeve 模式与一次性 TEST 互斥守卫 ──


def test_assert_sleeve_mode_allowed():
    from run_walkforward_backtest import assert_sleeve_mode_allowed
    import pytest
    assert_sleeve_mode_allowed(None, True, False)  # 未启用: 不拦
    assert_sleeve_mode_allowed({"sleeves": {}}, True, True)  # folds-only: 不拦
    with pytest.raises(RuntimeError):
        assert_sleeve_mode_allowed({"sleeves": {}}, True, False)


# ── F4: load_styles_config 预算校验覆盖全部键 ──


def test_load_styles_config_extra_budget_key_counted():
    from run_walkforward_backtest import load_styles_config
    import pytest
    cfg = _base_styles()
    cfg["styles"]["budgets"] = {"momentum": 0.8, "growth": 0.1, "value": 0.3}
    with pytest.raises(ValueError):
        load_styles_config(cfg)
