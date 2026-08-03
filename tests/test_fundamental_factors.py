import sys, os
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
