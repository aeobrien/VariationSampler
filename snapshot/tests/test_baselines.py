"""Tests for baseline metric distributions."""

import numpy as np
import pytest

from src.eval.baselines import (
    MetricDistribution,
    compute_pairwise_distances,
    is_in_band,
    save_baseline,
    load_baseline,
)


class TestMetricDistribution:

    def test_summary_stats(self):
        """Should compute correct summary statistics."""
        dist = MetricDistribution(metric_name="test", values=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert dist.mean == pytest.approx(3.0)
        assert dist.median == pytest.approx(3.0)
        assert dist.q25 == pytest.approx(2.0)
        assert dist.q75 == pytest.approx(4.0)

    def test_empty_distribution(self):
        """Empty distribution should return 0 for all stats."""
        dist = MetricDistribution(metric_name="test")
        assert dist.mean == 0.0
        assert dist.median == 0.0
        assert dist.std == 0.0

    def test_summary_dict(self):
        """summary() should return a well-formed dict."""
        dist = MetricDistribution(metric_name="mrstft", values=[0.1, 0.2, 0.3])
        s = dist.summary()
        assert s["metric_name"] == "mrstft"
        assert s["n"] == 3
        assert "mean" in s
        assert "median" in s


class TestIsInBand:

    def test_in_band(self):
        """Value within IQR should be in band."""
        dist = MetricDistribution(metric_name="test", values=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_in_band(3.0, dist) is True

    def test_below_band(self):
        """Value far below IQR should be out of band."""
        dist = MetricDistribution(metric_name="test", values=[10.0, 11.0, 12.0, 13.0, 14.0])
        assert is_in_band(0.0, dist) is False

    def test_above_band(self):
        """Value far above IQR should be out of band."""
        dist = MetricDistribution(metric_name="test", values=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_in_band(100.0, dist) is False

    def test_margin_widens_band(self):
        """Larger margin should accept more values."""
        dist = MetricDistribution(metric_name="test", values=[1.0, 2.0, 3.0, 4.0, 5.0])
        # With default margin, borderline value
        edge_value = 0.0
        # Narrow margin should reject, wide margin should accept
        assert is_in_band(edge_value, dist, margin=0.1) is False
        assert is_in_band(edge_value, dist, margin=2.0) is True


class TestSaveLoadRoundtrip:

    def test_roundtrip(self, tmp_path):
        """Save/load should preserve distribution."""
        original = MetricDistribution(
            metric_name="mrstft",
            values=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        path = tmp_path / "baseline.json"
        save_baseline(original, path)
        loaded = load_baseline(path)

        assert loaded.metric_name == original.metric_name
        assert loaded.values == pytest.approx(original.values)

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nonexistent.json")


class TestComputePairwiseDistances:

    def test_identical_pairs_have_zero_distance(self):
        """Pairwise distance of identical signals should be ~0."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        dist = compute_pairwise_distances([(audio, audio)], metric_fn="mrstft")
        assert len(dist.values) == 1
        assert dist.values[0] == pytest.approx(0.0, abs=1e-4)

    def test_different_pairs_have_positive_distance(self):
        """Different signals should have positive distance."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 0.3, 4410).astype(np.float32)
        b = rng.normal(0, 0.3, 4410).astype(np.float32)
        dist = compute_pairwise_distances([(a, b)], metric_fn="mrstft")
        assert dist.values[0] > 0.0
