"""Tests for iteration report generator."""

import json

import numpy as np
import pytest

from src.automation.report import (
    generate_iteration_report,
    save_iteration_report,
    _summarise_metric,
    _compute_trend,
)
from src.eval.baselines import MetricDistribution


@pytest.fixture
def sample_config():
    """Minimal config dict for testing."""
    return {
        "masking": {"p_tail": 0.08, "p_attack": 0.02},
        "sampling": {"temperature": 0.9, "top_p": 0.95},
    }


@pytest.fixture
def sample_eval_metrics():
    """Sample evaluation metrics with 5 test samples."""
    rng = np.random.default_rng(42)
    return {
        "mrstft": rng.uniform(0.1, 0.5, 5).tolist(),
        "mfcc": rng.uniform(2.0, 8.0, 5).tolist(),
        "attack_smear": rng.uniform(0.85, 1.0, 5).tolist(),
        "hf_energy_delta_db": rng.uniform(-2.0, 2.0, 5).tolist(),
        "accepted": [1.0, 1.0, 0.0, 1.0, 1.0],
    }


@pytest.fixture
def sample_audio_paths():
    """Sample audio paths dict."""
    return {
        f"sample_{i}": {
            "source": f"data/source_{i}.wav",
            "variations": [f"outputs/var_{i}_{j}.wav" for j in range(4)],
        }
        for i in range(5)
    }


class TestGenerateIterationReport:

    def test_has_required_keys(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """Report should contain all required top-level keys."""
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        required_keys = {
            "iteration_id", "timestamp", "config", "metrics",
            "baseline_comparison", "acceptance_rate", "trends",
            "best_samples", "worst_samples", "audio_paths", "n_samples",
        }
        assert required_keys.issubset(report.keys())

    def test_metrics_correctly_summarised(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """Each metric should have mean, p5, p50, p95, std."""
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        for metric_name in ["mrstft", "mfcc", "attack_smear"]:
            summary = report["metrics"][metric_name]
            assert "mean" in summary
            assert "p5" in summary
            assert "p50" in summary
            assert "p95" in summary
            assert "std" in summary
            assert summary["n"] == 5

    def test_acceptance_rate_computed(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """Acceptance rate should be computed from 'accepted' metric."""
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        # 4 out of 5 accepted
        assert report["acceptance_rate"] == pytest.approx(0.8, abs=0.01)

    def test_trend_first_iteration(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """First iteration should have 'first' trend for all metrics."""
        report = generate_iteration_report(
            iteration_id=0,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        for trend in report["trends"].values():
            assert trend == "first"

    def test_trend_improving(self, sample_config, sample_audio_paths):
        """Should detect improving trend when metric decreases."""
        prev_metrics = {"mrstft": {"mean": 0.5}}
        prev_report = {"metrics": prev_metrics}
        # Current values lower than previous
        eval_metrics = {"mrstft": [0.2, 0.3, 0.25]}
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=eval_metrics,
            audio_paths=sample_audio_paths,
            previous_report=prev_report,
        )
        assert report["trends"]["mrstft"] == "improving"

    def test_baseline_comparison(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """Should compare metrics to baseline distributions."""
        baseline = MetricDistribution(metric_name="mrstft", values=[0.2, 0.3, 0.4, 0.5, 0.6])
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
            baseline_dists={"mrstft": baseline},
        )
        assert "mrstft" in report["baseline_comparison"]
        comp = report["baseline_comparison"]["mrstft"]
        assert "in_band" in comp
        assert "baseline_median" in comp

    def test_empty_metrics_handled(self, sample_config, sample_audio_paths):
        """Should handle empty metric lists gracefully."""
        report = generate_iteration_report(
            iteration_id=0,
            config=sample_config,
            eval_metrics={},
            audio_paths=sample_audio_paths,
        )
        assert report["metrics"] == {}
        assert report["acceptance_rate"] is None

    def test_best_worst_samples(self, sample_config, sample_eval_metrics, sample_audio_paths):
        """Should identify best and worst samples."""
        report = generate_iteration_report(
            iteration_id=1,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        assert len(report["best_samples"]) <= 3
        assert len(report["worst_samples"]) <= 3
        for s in report["best_samples"]:
            assert "name" in s
            assert "score" in s


class TestSaveIterationReport:

    def test_saves_valid_json(self, tmp_path, sample_config, sample_eval_metrics, sample_audio_paths):
        """Saved report should be valid JSON."""
        report = generate_iteration_report(
            iteration_id=5,
            config=sample_config,
            eval_metrics=sample_eval_metrics,
            audio_paths=sample_audio_paths,
        )
        path = save_iteration_report(report, tmp_path)
        assert path.exists()
        assert path.name == "iteration-005.json"
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["iteration_id"] == 5


class TestSummariseMetric:

    def test_empty_list(self):
        """Empty list should return zeros."""
        result = _summarise_metric([])
        assert result["mean"] == 0.0
        assert result["n"] == 0

    def test_correct_values(self):
        """Should compute correct statistics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _summarise_metric(values)
        assert result["mean"] == pytest.approx(3.0)
        assert result["p50"] == pytest.approx(3.0)
        assert result["n"] == 5


class TestComputeTrend:

    def test_first_iteration(self):
        assert _compute_trend(0.5, None) == "first"

    def test_stable(self):
        assert _compute_trend(0.50, 0.51) == "stable"

    def test_improving(self):
        assert _compute_trend(0.3, 0.5) == "improving"

    def test_regressing(self):
        assert _compute_trend(0.7, 0.5) == "regressing"
