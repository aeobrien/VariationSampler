"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE, NQ, T_MAX
from src.eval.metrics import (
    multi_resolution_stft_distance,
    mfcc_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
    high_frequency_energy_delta,
    spectral_peak_divergence,
    inter_variation_distances,
)


class TestMultiResolutionSTFTDistance:

    def test_zero_for_identical(self, mono_audio):
        """Distance between identical signals should be zero."""
        dist = multi_resolution_stft_distance(mono_audio, mono_audio)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self, mono_audio, rng):
        """Distance between different signals should be positive."""
        noise = rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
        dist = multi_resolution_stft_distance(mono_audio, mono_audio + noise)
        assert dist > 0.0

    def test_windowed_analysis(self, mono_audio, rng):
        """Windowed analysis should use fewer samples."""
        noise = rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
        modified = mono_audio.copy()
        # Only modify the tail (after first 2000 samples)
        modified[2000:] += noise[2000:]

        # Full analysis should see more difference than attack-only
        full_dist = multi_resolution_stft_distance(mono_audio, modified)
        attack_dist = multi_resolution_stft_distance(
            mono_audio, modified, window_samples=2000,
        )
        assert attack_dist < full_dist

    def test_symmetric(self, mono_audio, rng):
        """Distance should be approximately symmetric."""
        other = mono_audio + rng.normal(0, 0.05, len(mono_audio)).astype(np.float32)
        d1 = multi_resolution_stft_distance(mono_audio, other)
        d2 = multi_resolution_stft_distance(other, mono_audio)
        assert d1 == pytest.approx(d2, rel=0.1)


class TestMFCCDistance:

    def test_zero_for_identical(self, mono_audio):
        """Distance should be zero for identical signals."""
        dist = mfcc_distance(mono_audio, mono_audio)
        assert dist == pytest.approx(0.0, abs=1e-4)

    def test_positive_for_different(self, mono_audio, rng):
        """Distance should be positive for different signals."""
        noise = rng.normal(0, 0.3, len(mono_audio)).astype(np.float32)
        dist = mfcc_distance(mono_audio, noise)
        assert dist > 0.0


class TestTokenChangeRate:

    def test_zero_for_identical(self, codegram):
        """Change rate should be 0 for identical codegrams."""
        rate = token_change_rate(codegram, codegram)
        assert rate == 0.0

    def test_one_for_completely_different(self, rng):
        """Change rate should be 1.0 when all tokens differ."""
        a = np.zeros((NQ, T_MAX), dtype=np.int32)
        b = np.ones((NQ, T_MAX), dtype=np.int32)
        rate = token_change_rate(a, b)
        assert rate == 1.0

    def test_subset_codebooks(self, codegram_pair):
        """Should only count changes in specified codebooks."""
        a, b = codegram_pair
        # Changes were made in codebooks 5-8
        rate_all = token_change_rate(a, b)
        rate_late = token_change_rate(a, b, codebooks=[5, 6, 7, 8])
        rate_early = token_change_rate(a, b, codebooks=[0, 1, 2])

        assert rate_early == 0.0  # No changes in early codebooks
        assert rate_late > rate_all  # Higher rate when focused on changed codebooks

    def test_shape_mismatch_raises(self):
        """Should raise on shape mismatch."""
        a = np.zeros((9, 86), dtype=np.int32)
        b = np.zeros((9, 50), dtype=np.int32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            token_change_rate(a, b)


class TestAttackSmearScore:

    def test_identical_gives_one(self, rng):
        """Identical signals should have smear score of 1.0."""
        # Use audio with energy in attack region
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        score = attack_smear_score(audio, audio)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_zeroed_attack_gives_low(self, rng):
        """Zeroing the attack should give a low smear score."""
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        modified = audio.copy()
        modified[:1323] = 0.0  # Zero first 30ms
        score = attack_smear_score(audio, modified)
        assert score < 0.1

    def test_silent_source_returns_zero(self):
        """Silent source should return 0.0."""
        silent = np.zeros(4410, dtype=np.float32)
        noise = np.random.default_rng(42).normal(0, 0.1, 4410).astype(np.float32)
        score = attack_smear_score(silent, noise)
        assert score == 0.0


class TestTransientCrossCorrelation:

    def test_identical_gives_one(self, rng):
        """Identical signals should have xcorr of 1.0."""
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        xcorr = transient_cross_correlation(audio, audio)
        assert xcorr == pytest.approx(1.0, abs=1e-5)

    def test_different_signal_lower(self, rng):
        """Different signal should have lower xcorr."""
        audio = rng.normal(0, 0.3, 4410).astype(np.float32)
        noise = rng.normal(0, 0.3, 4410).astype(np.float32)
        xcorr = transient_cross_correlation(audio, noise)
        assert xcorr < 0.9

    def test_silent_returns_zero(self):
        """Silent signals should return 0.0."""
        silent = np.zeros(4410, dtype=np.float32)
        xcorr = transient_cross_correlation(silent, silent)
        assert xcorr == 0.0


class TestHighFrequencyEnergyDelta:

    def test_identical_gives_zero(self, mono_audio):
        """Identical signals should have ~0 dB HF delta."""
        delta = high_frequency_energy_delta(mono_audio, mono_audio)
        assert delta == pytest.approx(0.0, abs=0.01)

    def test_boosted_hf_positive(self, mono_audio):
        """Boosting HF in variation should give positive delta."""
        rng = np.random.default_rng(99)
        hf_noise = rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
        # High-pass the noise with a simple differencing filter
        hf_noise = np.diff(hf_noise, prepend=0.0).astype(np.float32)
        boosted = mono_audio + hf_noise * 5.0
        delta = high_frequency_energy_delta(mono_audio, boosted)
        assert delta > 0.0


class TestSpectralPeakDivergence:

    def test_identical_gives_zero(self, mono_audio):
        """Identical signals should have 0 divergent peaks."""
        count = spectral_peak_divergence(mono_audio, mono_audio)
        assert count == 0

    def test_added_tone_detected(self):
        """A loud added tone should be detected as divergent."""
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        source = np.random.default_rng(42).normal(0, 0.01, len(t)).astype(np.float32)
        # Add a loud sinusoid to variation
        tone = np.sin(2 * np.pi * 5000 * t).astype(np.float32) * 0.8
        variation = source + tone
        count = spectral_peak_divergence(source, variation, threshold_db=6.0)
        assert count >= 1


class TestInterVariationDistances:

    def test_single_audio_returns_zero(self, mono_audio):
        """Single audio should return zero distances with 0 pairs."""
        result = inter_variation_distances([mono_audio])
        assert result["n_pairs"] == 0
        assert result["mean"] == 0.0

    def test_identical_audios_zero_distance(self, mono_audio):
        """Identical copies should have ~zero pairwise distance."""
        result = inter_variation_distances([mono_audio, mono_audio, mono_audio])
        assert result["n_pairs"] == 3
        assert result["mean"] == pytest.approx(0.0, abs=1e-4)

    def test_different_audios_positive(self, mono_audio, rng):
        """Different audios should have positive distances."""
        variations = [
            mono_audio + rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
            for _ in range(4)
        ]
        result = inter_variation_distances(variations)
        assert result["n_pairs"] == 6  # C(4,2)
        assert result["mean"] > 0.0
        assert result["min"] <= result["mean"] <= result["max"]
        assert result["std"] >= 0.0

    def test_mfcc_metric(self, mono_audio, rng):
        """Should work with MFCC metric."""
        variations = [
            mono_audio + rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
            for _ in range(3)
        ]
        result = inter_variation_distances(variations, metric_fn="mfcc")
        assert result["n_pairs"] == 3
        assert result["mean"] > 0.0

    def test_unknown_metric_raises(self, mono_audio):
        """Unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric_fn"):
            inter_variation_distances([mono_audio, mono_audio], metric_fn="unknown")
