"""Tests for machine-gun proxy evaluation."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE
from src.eval.machine_gun_proxy import (
    render_machine_gun,
    extract_hit_features,
    compute_self_similarity,
    machine_gun_score,
)


@pytest.fixture
def hit_audio(rng):
    """Short synthetic hit (100ms) with transient."""
    n_samples = int(0.1 * SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float32)
    # Sharp attack
    audio[:50] = 0.8
    t = np.arange(n_samples - 50, dtype=np.float32)
    audio[50:] = rng.normal(0, 0.2, n_samples - 50).astype(np.float32) * np.exp(-t / 1000)
    return audio


class TestRenderMachineGun:

    def test_correct_length(self, hit_audio):
        """Rendered audio should have correct length for BPM and hit count."""
        bpm = 120.0
        n_hits = 8
        result = render_machine_gun([hit_audio], bpm=bpm, n_hits=n_hits)

        sixteenth_note_s = 60.0 / bpm / 4.0
        interval_samples = int(sixteenth_note_s * SAMPLE_RATE)
        expected_len = (n_hits - 1) * interval_samples + len(hit_audio)
        assert len(result) == expected_len

    def test_single_hit(self, hit_audio):
        """Single hit should produce audio same length as hit."""
        result = render_machine_gun([hit_audio], n_hits=1)
        assert len(result) == len(hit_audio)

    def test_output_is_float32(self, hit_audio):
        """Output should be float32."""
        result = render_machine_gun([hit_audio])
        assert result.dtype == np.float32


class TestExtractHitFeatures:

    def test_feature_dimensionality(self, hit_audio):
        """Features should have n_mfcc + 3 dimensions."""
        n_mfcc = 13
        features = extract_hit_features(hit_audio, n_mfcc=n_mfcc)
        assert features.shape == (n_mfcc + 3,)

    def test_empty_audio_returns_zeros(self):
        """Empty audio should return zero feature vector."""
        features = extract_hit_features(np.array([], dtype=np.float32))
        assert np.all(features == 0.0)

    def test_features_are_float32(self, hit_audio):
        """Feature vector should be float32."""
        features = extract_hit_features(hit_audio)
        assert features.dtype == np.float32


class TestComputeSelfSimilarity:

    def test_identical_copies_score_one(self, hit_audio):
        """Identical feature vectors should give similarity ~1.0."""
        features = extract_hit_features(hit_audio)
        similarity = compute_self_similarity([features, features, features])
        assert similarity == pytest.approx(1.0, abs=1e-5)

    def test_different_signals_lower(self, rng):
        """Different signals should have lower self-similarity."""
        hits = []
        for i in range(4):
            freq = 200 + i * 500
            t = np.linspace(0, 0.1, int(0.1 * SAMPLE_RATE), dtype=np.float32)
            hit = np.sin(2 * np.pi * freq * t) * 0.5
            hits.append(hit)

        features = [extract_hit_features(h) for h in hits]
        similarity = compute_self_similarity(features)
        assert similarity < 0.99

    def test_single_hit_returns_one(self, hit_audio):
        """Single hit should return 1.0."""
        features = extract_hit_features(hit_audio)
        assert compute_self_similarity([features]) == 1.0


class TestMachineGunScore:

    def test_returns_expected_keys(self, hit_audio):
        """Score dict should have expected keys."""
        result = machine_gun_score([hit_audio, hit_audio])
        assert "self_similarity" in result
        assert "n_hits" in result
        assert "feature_dim" in result
        assert "features" in result

    def test_identical_copies_high_similarity(self, hit_audio):
        """Identical copies should have high self-similarity."""
        result = machine_gun_score([hit_audio] * 4)
        assert result["self_similarity"] > 0.99
        assert result["n_hits"] == 4
