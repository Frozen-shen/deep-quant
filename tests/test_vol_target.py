"""tests/test_vol_target.py — 波动率目标仓位缩放 (P0, Moreira & Muir)"""
import pytest
import scripts.active.run_walkforward_backtest as rw


def test_vol_target_scale_low_vol_full():
    cfg = {"target_pct": 0.70, "max_scale": 1.0, "min_scale": 0.4}
    s = rw._vol_target_scale(0.20, cfg)  # 市场低波动 → 满仓
    assert s == pytest.approx(1.0)


def test_vol_target_scale_high_vol_reduced():
    # 注: 简报原断言 s < 0.6 与公式不符 (0.70/0.95=0.737, 简报算术笔误),
    # 按推荐改为 s < 0.8 — 语义不变: 高波动 → 降仓
    cfg = {"target_pct": 0.70, "max_scale": 1.0, "min_scale": 0.4}
    s = rw._vol_target_scale(0.95, cfg)  # 市场高波动 → 降仓
    assert s < 0.8
    assert s >= 0.4


def test_vol_target_disabled_when_none():
    assert rw._vol_target_scale(0.95, None) == 1.0
