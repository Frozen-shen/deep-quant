"""冻结权重 (extend_val) vs 滚动重训 (级联折) 发散度监控单测 (2026-09-03, P2)

覆盖 check_frozen_vs_rolling_divergence():
- 真实 v29 数据复现: extend +44.2pp vs fold_6/7 滚动均值 -21.1pp → 应触发警示
- "两者接近"合成场景 → 不误报 (含阈值边界 14.9/15.1pp)
- extend 区间不覆盖任何级联折验证窗 → rolling_folds 为空、warning=False
"""
import copy
import json
import os

import pytest

import scripts.active.run_walkforward_backtest as rw


def _rec(val_s, val_e, excess):
    """构造一条折结果记录 (val 区间 + 超额年化, pp)。"""
    return {"val": f"{val_s} ~ {val_e}", "excess_annual": excess}


def _extend(s, e, excess):
    return {"period": f"{s} ~ {e}", "excess_annual": excess}


@pytest.fixture()
def real_results():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "ic_validation", "walkforward_results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def test_real_v29_data_triggers_warning(real_results):
    """v29 实况: extend +44.2pp vs fold_6/7 (-12.4/-29.9, 均值 -21.15pp) → 必警示。"""
    out = rw.check_frozen_vs_rolling_divergence(
        {k: v for k, v in real_results.items() if k.startswith("fold_")},
        real_results["extend_val"], rw.FOLDS, threshold_pp=15.0)
    assert out["warning"] is True
    assert out["rolling_folds"] == ["fold_6", "fold_7"]
    assert out["extend_excess_annual"] == pytest.approx(44.2)
    assert out["rolling_excess_annual"] == pytest.approx(-21.15, abs=1e-6)
    assert out["abs_gap_pp"] == pytest.approx(65.35, abs=1e-6)


def test_close_case_no_warning():
    """滚动与冻结接近 → 不误报。"""
    fold_results = {
        "fold_6": _rec("2025-01-01", "2025-12-31", 4.0),
        "fold_7": _rec("2026-01-01", "2026-06-30", 6.0),
    }
    out = rw.check_frozen_vs_rolling_divergence(
        fold_results, _extend("2025-01-02", "2026-06-30", 5.0),
        rw.FOLDS, threshold_pp=15.0)
    assert out["warning"] is False
    assert out["rolling_folds"] == ["fold_6", "fold_7"]
    assert out["abs_gap_pp"] == pytest.approx(0.0)


def test_threshold_boundary():
    """阈值边界: 差值 14.9pp 不警示, 15.1pp 警示。"""
    fold_results = {
        "fold_6": _rec("2025-01-01", "2025-12-31", -10.0),
        "fold_7": _rec("2026-01-01", "2026-06-30", -10.0),
    }
    below = rw.check_frozen_vs_rolling_divergence(
        fold_results, _extend("2025-01-02", "2026-06-30", 4.9),
        rw.FOLDS, threshold_pp=15.0)
    above = rw.check_frozen_vs_rolling_divergence(
        fold_results, _extend("2025-01-02", "2026-06-30", 5.1),
        rw.FOLDS, threshold_pp=15.0)
    assert below["warning"] is False and below["abs_gap_pp"] == pytest.approx(14.9)
    assert above["warning"] is True and above["abs_gap_pp"] == pytest.approx(15.1)


def test_no_overlapping_cascade_fold():
    """extend 区间 (2019) 不覆盖任何级联折验证窗 (2020+) → 无警示、无匹配。"""
    fold_results = {f"fold_{i}": _rec(*rw.FOLDS[i]["val"], 3.0)
                    for i in range(len(rw.FOLDS))}
    out = rw.check_frozen_vs_rolling_divergence(
        fold_results, _extend("2019-01-01", "2019-12-31", 20.0),
        rw.FOLDS, threshold_pp=15.0)
    assert out["warning"] is False
    assert out["rolling_folds"] == []
    assert out["rolling_excess_annual"] is None


def test_boundary_day_slack_matches_real_periods():
    """年份边界错位 (fold val 起 01-01 vs extend 起 01-02) 不丢匹配折。"""
    fold_results = {
        "fold_6": _rec("2025-01-01", "2025-12-31", 2.0),
        "fold_7": _rec("2026-01-01", "2026-06-30", 2.0),
    }
    out = rw.check_frozen_vs_rolling_divergence(
        fold_results, _extend("2025-01-02", "2026-06-30", 2.0),
        rw.FOLDS, threshold_pp=15.0)
    assert out["rolling_folds"] == ["fold_6", "fold_7"]
    assert out["warning"] is False
