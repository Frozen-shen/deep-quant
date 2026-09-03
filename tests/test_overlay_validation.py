"""overlay_validation.py — Overlay 参数 5 折验证框架单测 (2026-09-03, P4)

覆盖:
- 跨折一致改进 (≥3/5) + t 检验显著 → accept
- 方向一致但统计不显著 → reject_not_significant (不误 accept)
- 跨折方向不一致 (单折大涨不算数) → reject_not_consistent
- 多候选 BH-FDR 多重比较 (沿用 stats_correction.fdr_correction)
- validate_from_matrix / 缺基线行报错 / 无公共折 insufficient
"""
import numpy as np
import pytest

from overlay_validation import (paired_fold_deltas,
                                validate_from_matrix,
                                validate_overlay_candidates)

FOLDS = [f"fold_{i}" for i in range(1, 6)]


def _flat(v):
    return {f: float(x) for f, x in zip(FOLDS, v)}


def test_consistent_improvement_accept():
    base = _flat([5.0] * 5)
    cand = _flat([7.2, 6.9, 7.5, 6.8, 7.0])  # 每折 +1.8~+2.5pp
    out = validate_overlay_candidates("test", base, {"cand": cand})
    v = out["candidates"]["cand"]
    assert v["consistent_folds"] == 5
    assert out["verdicts"]["cand"] == "accept"
    assert v["p_value"] < 0.05


def test_consistent_but_not_significant_reject():
    base = _flat([0.0] * 5)
    # 3/5 折微正, 2 折负: 方向一致(勉强)但远不显著 → 不得 accept
    cand = _flat([0.3, -0.4, 0.5, 0.6, -0.2])
    out = validate_overlay_candidates("test", base, {"cand": cand})
    v = out["candidates"]["cand"]
    assert v["consistent_folds"] == 3
    assert v["mean_delta_pp"] > 0
    assert out["verdicts"]["cand"] == "reject_not_significant"


def test_inconsistent_single_fold_win_reject():
    base = _flat([0.0] * 5)
    cand = _flat([10.0, -2.0, -2.0, -2.0, -2.0])  # 单折大涨, 4 折更差
    out = validate_overlay_candidates("test", base, {"cand": cand})
    v = out["candidates"]["cand"]
    assert v["consistent_folds"] == 1
    assert out["verdicts"]["cand"] == "reject_not_consistent"


def test_min_consistent_threshold_4():
    base = _flat([0.0] * 5)
    cand = _flat([1.0, 1.0, 1.0, 1.0, -0.1])
    out = validate_overlay_candidates("test", base, {"cand": cand},
                                      min_consistent_folds=4)
    assert out["verdicts"]["cand"] == "accept"  # 4/5 一致 + 显著


def test_bh_across_candidates():
    """2 强候选 + 2 噪音候选: 只有强候选 accept。"""
    base = _flat([0.0] * 5)
    strong_a = _flat([2.0, 2.1, 1.9, 2.2, 2.0])
    strong_b = _flat([1.5, 1.6, 1.4, 1.7, 1.5])
    noise_c = _flat([0.2, -0.1, 0.1, 0.3, -0.2])
    noise_d = _flat([0.0, 0.2, -0.3, 0.1, 0.2])
    out = validate_overlay_candidates(
        "test", base,
        {"strong_a": strong_a, "strong_b": strong_b,
         "noise_c": noise_c, "noise_d": noise_d})
    assert out["verdicts"]["strong_a"] == "accept"
    assert out["verdicts"]["strong_b"] == "accept"
    assert out["verdicts"]["noise_c"].startswith("reject")
    assert out["verdicts"]["noise_d"].startswith("reject")


def test_from_matrix_and_missing_base():
    matrix = {"baseline": _flat([0.0] * 5),
              "cand": _flat([1.0, 1.0, 1.0, 1.0, 1.0])}
    out = validate_from_matrix("test", matrix)
    assert out["verdicts"]["cand"] == "accept"
    with pytest.raises(ValueError, match="基线行"):
        validate_from_matrix("test", {"cand": matrix["cand"]})


def test_no_common_folds_insufficient():
    out = validate_overlay_candidates("test", {"fold_x": 1.0},
                                      {"cand": {"fold_y": 2.0}})
    assert out["verdicts"]["cand"] == "insufficient"


def test_paired_fold_deltas_only_common():
    pd_ = paired_fold_deltas({"fold_1": 1.0, "fold_2": 2.0, "fold_9": 9.0},
                             {"fold_1": 3.0, "fold_2": 4.0})
    assert pd_["folds"] == ["fold_1", "fold_2"]
    assert pd_["deltas"] == {"fold_1": 2.0, "fold_2": 2.0}
