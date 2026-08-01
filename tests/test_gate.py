"""Tests for gate.py — hard enforcement gate."""
import pytest

from gate import GateViolation, check, load_config, validate_config_integrity


def _minimal_config():
    """Return a valid config dict that passes all gate checks."""
    return {
        "model": {
            "n_estimators": 50,
            "max_depth": 2,
        },
        "execution": {
            "slippage_bps": 30,
            "commission_buy": 0.00025,
            "commission_sell": 0.00075,
        },
    }


class TestPartitionChecks:
    def test_research_partition_passes(self):
        config = _minimal_config()
        # Should not raise
        check(partition="research", script_name="test.py", config=config)

    def test_development_partition_passes(self):
        config = _minimal_config()
        check(partition="development", script_name="test.py", config=config)

    def test_test_partition_passes(self):
        config = _minimal_config()
        check(partition="test", script_name="test.py", config=config)

    def test_blind_partition_raises(self):
        config = _minimal_config()
        with pytest.raises(GateViolation, match="盲测集"):
            check(partition="blind", script_name="test.py", config=config)

    def test_invalid_partition_raises(self):
        config = _minimal_config()
        with pytest.raises(GateViolation, match="非法分区"):
            check(partition="production", script_name="test.py", config=config)


class TestModelParamChecks:
    def test_compliant_model_passes(self):
        config = _minimal_config()
        config["model"]["n_estimators"] = 50
        config["model"]["max_depth"] = 2
        check(partition="research", script_name="test.py", config=config)

    def test_too_many_trees_raises(self):
        config = _minimal_config()
        config["model"]["n_estimators"] = 600
        with pytest.raises(GateViolation, match="n_estimators"):
            check(partition="research", script_name="test.py", config=config)

    def test_too_deep_raises(self):
        config = _minimal_config()
        config["model"]["max_depth"] = 5
        with pytest.raises(GateViolation, match="max_depth"):
            check(partition="research", script_name="test.py", config=config)


class TestCostFloorChecks:
    def test_adequate_cost_passes(self):
        config = _minimal_config()
        config["execution"]["slippage_bps"] = 30
        config["execution"]["commission_buy"] = 0.00025
        config["execution"]["commission_sell"] = 0.00075
        # total = 30 + (0.00025 + 0.00075) * 10000 = 40 >= 15
        check(partition="research", script_name="test.py", config=config)

    def test_zero_cost_raises(self):
        config = _minimal_config()
        config["execution"]["slippage_bps"] = 0
        config["execution"]["commission_buy"] = 0
        config["execution"]["commission_sell"] = 0
        with pytest.raises(GateViolation, match="成本"):
            check(partition="research", script_name="test.py", config=config)


class TestConfigIntegrity:
    def test_valid_config_no_errors(self):
        config = _minimal_config()
        raw_yaml = "model:\n  n_estimators: 50\n  max_depth: 2\n"
        errors = validate_config_integrity(config, raw_yaml=raw_yaml)
        assert errors == []

    def test_duplicate_key_detected(self):
        config = _minimal_config()
        raw_yaml = (
            "model:\n"
            "  n_estimators: 50\n"
            "  max_depth: 2\n"
            "  n_estimators: 100\n"
        )
        errors = validate_config_integrity(config, raw_yaml=raw_yaml)
        assert len(errors) > 0
        assert any("n_estimators" in e for e in errors)
