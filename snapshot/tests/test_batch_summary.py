"""Tests for batch summary report generator."""

import json

import pytest

from src.automation.batch_summary import (
    generate_batch_summary,
    save_batch_summary,
    format_markdown_summary,
    _compute_config_diff,
    _trend_arrow,
)


@pytest.fixture
def iteration_reports():
    """List of 3 mock iteration reports."""
    return [
        {
            "iteration_id": 0,
            "timestamp": "2026-03-01T10:00:00+00:00",
            "config": {
                "masking": {"p_tail": 0.08, "p_attack": 0.02},
                "sampling": {"temperature": 0.9},
            },
            "metrics": {
                "mrstft": {"mean": 0.45, "p50": 0.44, "std": 0.1, "n": 5},
                "mfcc": {"mean": 5.0, "p50": 4.8, "std": 1.0, "n": 5},
            },
            "best_samples": [{"name": "snare_01", "score": 0.2}],
            "worst_samples": [{"name": "kick_03", "score": 0.9}],
            "audio_paths": {
                "snare_01": {"source": "data/snare_01.wav", "variations": []},
                "kick_03": {"source": "data/kick_03.wav", "variations": []},
            },
            "acceptance_rate": 0.8,
            "trends": {"mrstft": "first", "mfcc": "first"},
        },
        {
            "iteration_id": 1,
            "timestamp": "2026-03-01T11:00:00+00:00",
            "config": {
                "masking": {"p_tail": 0.10, "p_attack": 0.02},
                "sampling": {"temperature": 0.9},
            },
            "metrics": {
                "mrstft": {"mean": 0.40, "p50": 0.39, "std": 0.09, "n": 5},
                "mfcc": {"mean": 4.8, "p50": 4.6, "std": 0.9, "n": 5},
            },
            "best_samples": [{"name": "snare_02", "score": 0.18}],
            "worst_samples": [{"name": "kick_01", "score": 0.85}],
            "audio_paths": {
                "snare_02": {"source": "data/snare_02.wav", "variations": []},
                "kick_01": {"source": "data/kick_01.wav", "variations": []},
            },
            "acceptance_rate": 0.85,
            "trends": {"mrstft": "improving", "mfcc": "improving"},
            "claude_reasoning": "Increased p_tail to increase variation magnitude.",
        },
        {
            "iteration_id": 2,
            "timestamp": "2026-03-01T12:00:00+00:00",
            "config": {
                "masking": {"p_tail": 0.12, "p_attack": 0.03},
                "sampling": {"temperature": 0.85},
            },
            "metrics": {
                "mrstft": {"mean": 0.38, "p50": 0.37, "std": 0.08, "n": 5},
                "mfcc": {"mean": 4.5, "p50": 4.3, "std": 0.8, "n": 5},
            },
            "best_samples": [{"name": "snare_01", "score": 0.15}],
            "worst_samples": [{"name": "kick_03", "score": 0.82}],
            "audio_paths": {},
            "acceptance_rate": 0.9,
            "trends": {"mrstft": "improving", "mfcc": "stable"},
        },
    ]


class TestGenerateBatchSummary:

    def test_has_required_sections(self, iteration_reports):
        """Summary should contain all required sections."""
        summary = generate_batch_summary("001", iteration_reports)
        required_keys = {
            "batch_id", "n_iterations", "stop_reason", "config_trajectory",
            "metric_trajectory", "best_samples", "worst_samples",
            "starting_config", "final_config",
        }
        assert required_keys.issubset(summary.keys())

    def test_iteration_count(self, iteration_reports):
        """Should report correct number of iterations."""
        summary = generate_batch_summary("001", iteration_reports)
        assert summary["n_iterations"] == 3

    def test_config_trajectory_computed(self, iteration_reports):
        """Should compute config diffs between iterations."""
        summary = generate_batch_summary("001", iteration_reports)
        traj = summary["config_trajectory"]
        assert len(traj) == 2  # 3 iterations = 2 diffs
        # First diff: p_tail changed 0.08 -> 0.10
        first_changes = traj[0]["changes"]
        changed_keys = [c["key"] for c in first_changes]
        assert "masking.p_tail" in changed_keys

    def test_metric_trajectory(self, iteration_reports):
        """Should track metric values across iterations."""
        summary = generate_batch_summary("001", iteration_reports)
        traj = summary["metric_trajectory"]
        assert "mrstft" in traj
        assert len(traj["mrstft"]["values"]) == 3
        assert traj["mrstft"]["trend"] in ("↓", "↑", "→", "—")

    def test_stop_reason_recorded(self, iteration_reports):
        """Should record stop reason."""
        summary = generate_batch_summary("001", iteration_reports, stop_reason="stagnation")
        assert summary["stop_reason"] == "stagnation"

    def test_empty_reports_handled(self):
        """Should handle empty reports list."""
        summary = generate_batch_summary("001", [])
        assert summary["n_iterations"] == 0
        assert "error" in summary

    def test_starting_and_final_config(self, iteration_reports):
        """Should capture starting and final configs."""
        summary = generate_batch_summary("001", iteration_reports)
        assert summary["starting_config"]["masking"]["p_tail"] == 0.08
        assert summary["final_config"]["masking"]["p_tail"] == 0.12

    def test_listening_notes_loaded(self, iteration_reports, tmp_path):
        """Should load listening notes from file."""
        notes_path = tmp_path / "notes.md"
        notes_path.write_text("# Notes\nSounds great!")
        summary = generate_batch_summary(
            "001", iteration_reports, listening_notes_path=notes_path,
        )
        assert summary["listening_notes"] == "# Notes\nSounds great!"


class TestSaveBatchSummary:

    def test_saves_json_and_markdown(self, iteration_reports, tmp_path):
        """Should save both JSON and markdown files."""
        summary = generate_batch_summary("001", iteration_reports)
        save_batch_summary(summary, tmp_path)
        assert (tmp_path / "batch-001-summary.json").exists()
        assert (tmp_path / "batch-001-summary.md").exists()

    def test_json_is_valid(self, iteration_reports, tmp_path):
        """Saved JSON should be loadable."""
        summary = generate_batch_summary("001", iteration_reports)
        save_batch_summary(summary, tmp_path)
        with open(tmp_path / "batch-001-summary.json") as f:
            loaded = json.load(f)
        assert loaded["batch_id"] == "001"


class TestFormatMarkdownSummary:

    def test_markdown_is_string(self, iteration_reports):
        """Should return a string."""
        summary = generate_batch_summary("001", iteration_reports)
        md = format_markdown_summary(summary)
        assert isinstance(md, str)
        assert "# Batch Summary: 001" in md

    def test_markdown_contains_sections(self, iteration_reports):
        """Should contain all major sections."""
        summary = generate_batch_summary("001", iteration_reports)
        md = format_markdown_summary(summary)
        assert "## Metadata" in md
        assert "## Config Trajectory" in md
        assert "## Metric Trajectory" in md
        assert "## Best Samples" in md
        assert "## Worst Samples" in md


class TestConfigDiff:

    def test_no_changes(self):
        """Identical configs should produce no diffs."""
        config = {"a": 1, "b": 2}
        assert _compute_config_diff(config, config) == []

    def test_simple_change(self):
        """Should detect simple value changes."""
        a = {"x": 1}
        b = {"x": 2}
        diffs = _compute_config_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "x"
        assert diffs[0]["old"] == 1
        assert diffs[0]["new"] == 2

    def test_nested_change(self):
        """Should detect nested value changes."""
        a = {"outer": {"inner": 1}}
        b = {"outer": {"inner": 2}}
        diffs = _compute_config_diff(a, b)
        assert len(diffs) == 1
        assert diffs[0]["key"] == "outer.inner"


class TestTrendArrow:

    def test_decreasing(self):
        assert _trend_arrow([0.5, 0.4, 0.3, 0.2]) == "↓"

    def test_increasing(self):
        assert _trend_arrow([0.2, 0.3, 0.4, 0.5]) == "↑"

    def test_stable(self):
        assert _trend_arrow([0.5, 0.5, 0.5, 0.5]) == "→"

    def test_single_value(self):
        assert _trend_arrow([0.5]) == "—"
