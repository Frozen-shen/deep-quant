"""regime 五折验证的输入门禁测试。"""

import pytest

import scripts.active.run_regime_robustness as regime_robustness


def test_folds_require_current_fdr_report(tmp_path, monkeypatch):
    """缺少当前 FDR 产物时，不能静默使用旧 p5 因子集继续正式验证。"""
    missing = tmp_path / "fdr_correction_report.json"
    monkeypatch.setattr(regime_robustness, "FDR_REPORT_PATH", str(missing))

    with pytest.raises(RuntimeError, match="fdr_correction_report"):
        regime_robustness.require_fdr_report_for_folds()
