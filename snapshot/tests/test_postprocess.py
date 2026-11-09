"""Tests for post-processing chain."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE
from src.postprocess.chain import (
    remove_dc,
    match_peak,
    fade_tail,
    dither_to_16bit,
    postprocess,
)


@pytest.fixture
def dc_audio(rng):
    """Audio with DC offset."""
    n_samples = SAMPLE_RATE // 2
    audio = rng.normal(0.5, 0.2, n_samples).astype(np.float32)
    return audio


@pytest.fixture
def short_audio(rng):
    """Short audio with transient for postprocess tests."""
    n_samples = SAMPLE_RATE // 4  # 0.25s
    audio = np.zeros(n_samples, dtype=np.float32)
    audio[:100] = 0.8
    t = np.arange(n_samples - 100, dtype=np.float32)
    audio[100:] = rng.normal(0, 0.3, n_samples - 100).astype(np.float32) * np.exp(-t / 2000)
    return audio


class TestRemoveDC:

    def test_centers_signal(self, dc_audio):
        """DC removal should center signal near zero mean."""
        result = remove_dc(dc_audio)
        assert abs(np.mean(result)) < 1e-6

    def test_already_centered_unchanged(self, rng):
        """Zero-mean audio should be essentially unchanged."""
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        audio -= np.mean(audio)  # Force exact zero mean
        result = remove_dc(audio)
        np.testing.assert_array_almost_equal(result, audio, decimal=6)


class TestMatchPeak:

    def test_preserves_reference_peak(self, short_audio):
        """Peak-matched audio should have same peak amplitude as reference."""
        reference = short_audio
        audio = short_audio * 0.5  # Quieter
        result = match_peak(audio, reference)

        peak_ref = np.max(np.abs(reference))
        peak_result = np.max(np.abs(result))
        assert peak_result == pytest.approx(peak_ref, rel=0.01)

    def test_preserves_waveform_shape(self, short_audio):
        """Peak matching should scale uniformly, preserving attack-to-body ratio."""
        reference = short_audio
        audio = short_audio * 0.5
        result = match_peak(audio, reference)

        # The ratio between any two samples should be preserved
        gain = np.max(np.abs(reference)) / np.max(np.abs(audio))
        np.testing.assert_array_almost_equal(result, audio * gain, decimal=5)

    def test_silent_audio_unchanged(self):
        """Silent audio should be returned unchanged."""
        silent = np.zeros(4410, dtype=np.float32)
        reference = np.ones(4410, dtype=np.float32) * 0.5
        result = match_peak(silent, reference)
        np.testing.assert_array_equal(result, silent)


class TestFadeTail:

    def test_end_is_zero(self, short_audio):
        """Last sample should be zero or near-zero after fade."""
        result = fade_tail(short_audio, fade_ms=10)
        assert abs(result[-1]) < 1e-6

    def test_start_preserved(self, short_audio):
        """Start of audio should be unchanged by tail fade."""
        result = fade_tail(short_audio, fade_ms=10)
        # First half should be identical
        mid = len(short_audio) // 2
        np.testing.assert_array_equal(result[:mid], short_audio[:mid])

    def test_does_not_modify_input(self, short_audio):
        """Should not modify the input array."""
        original = short_audio.copy()
        fade_tail(short_audio, fade_ms=10)
        np.testing.assert_array_equal(short_audio, original)


class TestDitherTo16Bit:

    def test_output_is_int16(self, short_audio):
        """Output should be int16."""
        result = dither_to_16bit(short_audio)
        assert result.dtype == np.int16

    def test_in_range(self, short_audio):
        """Output should be within int16 range."""
        result = dither_to_16bit(short_audio)
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)

    def test_same_length(self, short_audio):
        """Output should have same length as input."""
        result = dither_to_16bit(short_audio)
        assert len(result) == len(short_audio)


class TestPostprocess:

    def test_full_chain_valid_audio(self, short_audio):
        """Full chain should produce valid float32 audio."""
        config = {"fade_ms": 10}
        result = postprocess(short_audio, short_audio, config)
        assert result.dtype == np.float32
        assert len(result) == len(short_audio)

    def test_chain_removes_dc(self):
        """Full chain should remove DC offset."""
        rng = np.random.default_rng(42)
        audio = rng.normal(0.5, 0.2, 4410).astype(np.float32)
        reference = rng.normal(0, 0.3, 4410).astype(np.float32)
        config = {"fade_ms": 10}
        result = postprocess(audio, reference, config)
        # After DC removal and level match, mean should be near zero
        # (not exactly zero due to level matching)
        assert abs(np.mean(result)) < 0.1

    def test_chain_applies_fade(self, short_audio):
        """Full chain should apply fade at tail."""
        config = {"fade_ms": 10}
        result = postprocess(short_audio, short_audio, config)
        assert abs(result[-1]) < 1e-6
