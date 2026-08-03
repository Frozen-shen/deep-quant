import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from regime_detector import RegimeDetector, Regime

def test_detect_v2_basic():
    det = RegimeDetector.from_benchmark_parquet("data/cache/index_csi1000.parquet")
    regime, vol_pct = det.detect_v2("2024-06-30")
    assert regime in list(Regime)
    assert 0.0 <= vol_pct <= 1.0

def test_momentum_crash_protection():
    det = RegimeDetector.from_benchmark_parquet("data/cache/index_csi1000.parquet")
    mults = det.get_weight_multipliers("2024-06-30")
    assert "momentum" in mults
    assert "reversal" in mults
    assert 0.0 <= mults["momentum"] <= 3.0
