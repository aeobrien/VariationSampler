"""Tests for preprocessing pipeline."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE, MAX_SAMPLES
from src.data.preprocessing import trim_silence, normalize_loudness, preprocess_sample


class TestTrimSilence:

    def test_trims_leading_silence(self, stereo_audio):
        """Should remove leading zeros."""
        # Add 0.5s of silence at the start
        silence = np.zeros((2, SAMPLE_RATE // 2), dtype=np.float32)
        padded = np.concatenate([silence, stereo_audio], axis=1)

        trimmed = trim_silence(padded)
        assert trimmed.shape[1] < padded.shape[1]

    def test_trims_trailing_silence(self, stereo_audio):
        """Should remove trailing zeros."""
        silence = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        padded = np.concatenate([stereo_audio, silence], axis=1)

        trimmed = trim_silence(padded)
        assert trimmed.shape[1] < padded.shape[1]

    def test_preserves_content(self, stereo_audio):
        """Trimmed content should still contain the original signal."""
        silence = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        padded = np.concatenate([silence, stereo_audio, silence], axis=1)

        trimmed = trim_silence(padded)
        # Peak amplitude should be preserved
        assert np.max(np.abs(trimmed)) == pytest.approx(np.max(np.abs(stereo_audio)), abs=1e-6)

    def test_handles_silent_audio(self):
        """Should return silent audio unchanged."""
        silent = np.zeros((2, 1000), dtype=np.float32)
        result = trim_silence(silent)
        assert result.shape == silent.shape


class TestNormalizeLoudness:

    def test_normalizes_to_target(self, stereo_audio):
        """Should bring RMS close to target level."""
        normalized = normalize_loudness(stereo_audio, target_db=-18.0, tolerance_db=0.5)
        rms = np.sqrt(np.mean(normalized ** 2))
        rms_db = 20.0 * np.log10(rms)
        assert abs(rms_db - (-18.0)) < 0.5

    def test_skips_when_within_tolerance(self, stereo_audio):
        """Should not modify audio already at target level."""
        # First normalize
        normalized = normalize_loudness(stereo_audio, target_db=-18.0, tolerance_db=1.0)
        # Normalize again — should be unchanged
        double_normalized = normalize_loudness(normalized, target_db=-18.0, tolerance_db=1.0)
        np.testing.assert_array_equal(normalized, double_normalized)

    def test_handles_zero_audio(self):
        """Should return zero audio unchanged."""
        silent = np.zeros((2, 1000), dtype=np.float32)
        result = normalize_loudness(silent)
        np.testing.assert_array_equal(result, silent)


class TestPreprocessSample:

    def test_output_shape(self, stereo_audio):
        """Output should be [channels, MAX_SAMPLES]."""
        result = preprocess_sample(stereo_audio)
        assert result.shape == (2, MAX_SAMPLES)

    def test_output_dtype(self, stereo_audio):
        """Output should be float32."""
        result = preprocess_sample(stereo_audio)
        assert result.dtype == np.float32

    def test_shorter_audio_gets_padded(self):
        """Audio shorter than target should be zero-padded."""
        short = np.random.default_rng(0).normal(0, 0.1, (2, 1000)).astype(np.float32)
        result = preprocess_sample(short, target_samples=5000)
        assert result.shape == (2, 5000)
        # Tail should be mostly zeros (after normalization + padding)
        assert np.allclose(result[:, 4000:], 0.0, atol=1e-6)
