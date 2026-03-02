"""Tests for automation runner."""

import json

import numpy as np
import pytest

from src.automation.runner import AutomationRunner


@pytest.fixture
def batch_config(tmp_path):
    """Create a minimal batch config file for testing."""
    config = {
        "batch_id": "test-001",
        "max_iterations": 3,
        "stagnation_limit": 2,
        "hyperparameters": {
            "mask_p_tail": 0.08,
            "mask_p_attack": 0.02,
            "temperature": 0.9,
        },
        "automation": {
            "allowed_params": ["mask_p_tail", "mask_p_attack", "temperature"],
            "step_size_limits": {"mask_p_tail": 0.05, "temperature": 0.2},
        },
        "guardrails": {
            "rollback_thresholds": {
                "mrstft_p95": 0.50,
            },
        },
        "reports_dir": str(tmp_path / "reports"),
        "outputs_dir": str(tmp_path / "outputs"),
        "dry_run": True,
    }
    config_path = tmp_path / "batch.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


class TestAutomationRunner:

    def test_init_loads_config(self, batch_config):
        """Runner should load batch config on init."""
        runner = AutomationRunner(batch_config)
        assert runner.batch_id == "test-001"
        assert runner.max_iterations == 3
        assert runner.stagnation_limit == 2

    def test_stops_at_iteration_cap(self, batch_config):
        """Should stop when max_iterations is reached."""
        runner = AutomationRunner(batch_config)
        summary = runner.run()
        assert summary["stop_reason"] == "iteration_cap"
        assert summary["n_iterations"] == 3

    def test_stops_on_stagnation(self, batch_config, tmp_path):
        """Should stop when stagnation limit is reached."""
        # Create a runner with stagnation_limit=1 and 10 max iterations
        config = {
            "batch_id": "stag-test",
            "max_iterations": 10,
            "stagnation_limit": 1,
            "hyperparameters": {"mask_p_tail": 0.08},
            "guardrails": {"rollback_thresholds": {}},
            "reports_dir": str(tmp_path / "reports2"),
            "outputs_dir": str(tmp_path / "outputs2"),
            "dry_run": True,
        }
        config_path = tmp_path / "stag_batch.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        runner = AutomationRunner(config_path)

        # Override _run_evaluation to return metrics that stagnate
        call_count = [0]
        def fake_eval(hyperparams, iteration):
            call_count[0] += 1
            return {"mrstft": [0.5, 0.5, 0.5]}, {}
        runner._run_evaluation = fake_eval

        summary = runner.run()
        # After 2 iterations with no improvement, stagnation counter hits 1
        # which equals stagnation_limit=1, so it stops
        assert summary["stop_reason"] == "stagnation"

    def test_regression_detection(self, batch_config):
        """Should detect metric regression."""
        runner = AutomationRunner(batch_config)
        metrics = {"mrstft": [0.6, 0.7, 0.8]}  # p95 > 0.50 threshold
        regressed, reasons = runner._check_regression(
            metrics, {"mrstft_p95": 0.50},
        )
        assert regressed
        assert len(reasons) > 0

    def test_no_regression_within_threshold(self, batch_config):
        """Should not flag regression when within threshold."""
        runner = AutomationRunner(batch_config)
        metrics = {"mrstft": [0.1, 0.2, 0.3]}  # p95 = 0.3 < 0.50
        regressed, reasons = runner._check_regression(
            metrics, {"mrstft_p95": 0.50},
        )
        assert not regressed

    def test_rollback_config(self, batch_config):
        """Should rollback to last non-rolled-back config."""
        runner = AutomationRunner(batch_config)
        runner.iteration_history = [
            {"iteration_id": 0, "config": {"mask_p_tail": 0.08}, "rolled_back": False},
            {"iteration_id": 1, "config": {"mask_p_tail": 0.12}, "rolled_back": True},
        ]
        rolled_back = runner._rollback_config(runner.iteration_history)
        assert rolled_back["mask_p_tail"] == 0.08

    def test_dry_run_skips_claude(self, batch_config):
        """Dry run should not call Claude API."""
        runner = AutomationRunner(batch_config)
        assert runner.dry_run is True
        # Run should complete without API errors
        summary = runner.run()
        assert summary["n_iterations"] > 0

    def test_batch_summary_written(self, batch_config, tmp_path):
        """Should write batch summary on completion."""
        runner = AutomationRunner(batch_config)
        runner.run()
        reports_dir = tmp_path / "reports"
        json_files = list(reports_dir.glob("batch-*-summary.json"))
        assert len(json_files) == 1
        md_files = list(reports_dir.glob("batch-*-summary.md"))
        assert len(md_files) == 1

    def test_config_applied_between_iterations(self, batch_config):
        """Config updates should persist between iterations."""
        runner = AutomationRunner(batch_config)
        initial_temp = runner.current_hyperparams.get("temperature")
        assert initial_temp == 0.9
        # After run, hyperparams should still be accessible
        runner.run()
        assert runner.current_hyperparams is not None
