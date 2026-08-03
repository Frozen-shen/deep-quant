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
