"""Tests for experiment_tracker.py"""

import json
from pathlib import Path

from experiment_tracker import log_experiment, list_experiments


class TestLogExperiment:
    def test_creates_json_file(self, tmp_path):
        config = {"lookback": 20, "threshold": 0.5}
        results = {"sharpe": 1.85, "max_drawdown": -0.12}

        exp_id = log_experiment(
            script_name="run_backtest.py",
            partition="train",
            config=config,
            results=results,
            notes="test run",
            experiments_dir=str(tmp_path),
        )

        # exp_id format check
        assert exp_id.startswith("exp_")

        # Exactly 1 JSON file created
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

        # Load and verify fields
        record = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert record["script"] == "run_backtest.py"
        assert record["partition"] == "train"
        assert record["results"]["sharpe"] == 1.85
        assert "timestamp" in record
        assert "config_hash" in record
        assert record["notes"] == "test run"
        assert record["experiment_id"] == exp_id

    def test_config_hash_deterministic(self, tmp_path):
        config = {"lookback": 20, "threshold": 0.5}
        results = {"sharpe": 1.0}

        exp_id_1 = log_experiment(
            script_name="a.py",
            partition="train",
            config=config,
            results=results,
            experiments_dir=str(tmp_path),
        )
        exp_id_2 = log_experiment(
            script_name="b.py",
            partition="test",
            config=config,
            results=results,
            experiments_dir=str(tmp_path),
        )

        files = sorted(tmp_path.glob("*.json"))
        record1 = json.loads(files[0].read_text(encoding="utf-8"))
        record2 = json.loads(files[1].read_text(encoding="utf-8"))

        assert record1["config_hash"] == record2["config_hash"]
        assert len(record1["config_hash"]) == 12


class TestListExperiments:
    def test_empty_dir(self, tmp_path):
        result = list_experiments(experiments_dir=str(tmp_path))
        assert result == []

    def test_nonexistent_dir(self):
        result = list_experiments(experiments_dir="/nonexistent/path/xyz")
        assert result == []

    def test_lists_all(self, tmp_path):
        config = {"a": 1}
        results = {"sharpe": 1.0}

        log_experiment("s1.py", "train", config, results, experiments_dir=str(tmp_path))
        log_experiment("s2.py", "test", config, results, experiments_dir=str(tmp_path))

        records = list_experiments(experiments_dir=str(tmp_path))
        assert len(records) == 2
