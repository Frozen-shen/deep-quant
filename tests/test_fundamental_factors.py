import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_library import FUNDAMENTAL_FACTORS

def test_fundamental_factors_exist():
    assert len(FUNDAMENTAL_FACTORS) >= 10
    assert "fund_bp" in FUNDAMENTAL_FACTORS
    assert "fund_ep" in FUNDAMENTAL_FACTORS
