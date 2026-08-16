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
