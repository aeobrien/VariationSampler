"""Tests for onset detection and alignment."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE
from src.data.alignment import (
    detect_onset_spectral_flux,
    fine_align_cross_correlation,
    align_to_reference,
)


class TestDetectOnsetSpectralFlux:

    def test_detects_onset_near_expected_position(self, audio_with_onset):
        """Onset should be detected within ±10ms of the known position."""
        expected_sample = int(0.1 * SAMPLE_RATE)
        tolerance_samples = int(0.01 * SAMPLE_RATE)  # 10ms

        onset = detect_onset_spectral_flux(audio_with_onset)

        assert abs(onset - expected_sample) < tolerance_samples, (
            f"Onset at {onset} too far from expected {expected_sample}"
        )

    def test_raises_on_silence(self):
        """Should raise ValueError on silent audio."""
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with pytest.raises(ValueError, match="No onset detected"):
            detect_onset_spectral_flux(silent)

    def test_raises_on_2d_input(self, stereo_audio):
        """Should raise on non-mono input."""
        with pytest.raises(ValueError, match="1D mono"):
            detect_onset_spectral_flux(stereo_audio)


class TestFineAlignCrossCorrelation:

    @pytest.fixture
    def attack_signal(self, rng):
        """Signal with content from sample 0 (post-onset region)."""
        n = int(0.05 * SAMPLE_RATE)  # 50ms
        t = np.arange(n, dtype=np.float32)
        decay = np.exp(-t / (0.01 * SAMPLE_RATE))
        signal = rng.normal(0, 0.5, n).astype(np.float32) * decay
        signal[0:5] = 0.8  # strong transient at start
        return signal

    def test_recovers_known_shift(self, attack_signal):
        """Should recover a known sample offset."""
        shift = 20
        shifted = np.zeros_like(attack_signal)
        shifted[shift:] = attack_signal[:-shift]

        offset = fine_align_cross_correlation(attack_signal, shifted, window_ms=10.0)

        # The offset should recover roughly the applied shift
        assert abs(offset - shift) <= 5, f"Recovered offset {offset}, expected ~{shift}"

    def test_zero_shift_for_identical(self, attack_signal):
        """Identical signals should yield zero or near-zero offset."""
        offset = fine_align_cross_correlation(attack_signal, attack_signal)
        assert abs(offset) <= 1


class TestAlignToReference:

    def test_alignment_preserves_content(self, audio_with_onset):
        """Aligned audio should have similar energy to original."""
        # Shift the target
        shift = 50
        shifted = np.zeros_like(audio_with_onset)
        shifted[shift:] = audio_with_onset[:-shift]

        aligned = align_to_reference(audio_with_onset, shifted)

        # Energy should be preserved (within 10% — some samples may be lost at edges)
        orig_energy = np.sum(audio_with_onset ** 2)
        aligned_energy = np.sum(aligned ** 2)
        assert aligned_energy > 0.8 * orig_energy, "Alignment lost too much energy"

    def test_output_same_length(self, audio_with_onset):
        """Output should have same length as input target."""
        shift = 30
        shifted = np.zeros_like(audio_with_onset)
        shifted[shift:] = audio_with_onset[:-shift]

        aligned = align_to_reference(audio_with_onset, shifted)
        assert len(aligned) == len(audio_with_onset)
