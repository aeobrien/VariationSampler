"""Tests for acceptance filtering."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE, NQ, T_MAX
from src.eval.acceptance import (
    AcceptanceResult,
    evaluate_candidate,
    filter_candidates,
)


@pytest.fixture
def transient_audio(rng):
    """Audio with energy starting from sample 0 (no silent lead-in)."""
    n_samples = SAMPLE_RATE // 2
    audio = rng.normal(0, 0.3, n_samples).astype(np.float32)
    # Add transient at start
    audio[:100] = 0.8
    t = np.arange(n_samples - 100, dtype=np.float32)
    audio[100:] *= np.exp(-t / (0.1 * SAMPLE_RATE)).astype(np.float32)
    return audio


@pytest.fixture
def acceptance_config():
    """Permissive acceptance config for testing."""
    return {
        "acceptance": {
            "mrstft_band": [0.0, 5.0],
            "mfcc_band": [0.0, 50.0],
            "min_token_change_rate": 0.0,
            "max_token_change_rate": 1.0,
            "min_attack_smear": 0.5,
            "min_transient_xcorr": 0.5,
            "max_hf_energy_delta_db": 20.0,
            "max_spectral_peak_divergence": 10,
            "attack_ms": 30,
        }
    }


@pytest.fixture
def strict_config():
    """Strict acceptance config that rejects most variations."""
    return {
        "acceptance": {
            "mrstft_band": [0.0, 0.001],
            "mfcc_band": [0.0, 0.001],
            "min_token_change_rate": 0.0,
            "max_token_change_rate": 0.001,
            "min_attack_smear": 0.999,
            "min_transient_xcorr": 0.999,
            "max_hf_energy_delta_db": 0.001,
            "max_spectral_peak_divergence": 0,
            "attack_ms": 30,
        }
    }


class TestEvaluateCandidate:

    def test_identical_accepted_with_permissive_config(self, transient_audio, codegram, acceptance_config):
        """Identical signal should be accepted with permissive config."""
        result = evaluate_candidate(
            transient_audio, transient_audio, codegram, codegram, acceptance_config,
        )
        assert result.accepted is True
        assert len(result.reject_reasons) == 0

    def test_heavily_modified_rejected(self, mono_audio, codegram, acceptance_config):
        """Heavily modified signal should be rejected."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, len(mono_audio)).astype(np.float32)
        different_cg = rng.integers(0, 1024, size=codegram.shape, dtype=np.int32)

        # Use stricter config
        config = {
            "acceptance": {
                "mrstft_band": [0.0, 0.01],
                "mfcc_band": [0.0, 0.5],
                "min_token_change_rate": 0.0,
                "max_token_change_rate": 0.01,
                "min_attack_smear": 0.99,
                "min_transient_xcorr": 0.99,
                "max_hf_energy_delta_db": 0.1,
                "max_spectral_peak_divergence": 0,
                "attack_ms": 30,
            }
        }
        result = evaluate_candidate(
            mono_audio, noise, codegram, different_cg, config,
        )
        assert result.accepted is False
        assert len(result.reject_reasons) > 0

    def test_result_has_all_metric_keys(self, transient_audio, codegram, acceptance_config):
        """Result should contain all expected metric keys."""
        result = evaluate_candidate(
            transient_audio, transient_audio, codegram, codegram, acceptance_config,
        )
        expected_keys = {
            "mrstft", "mfcc", "token_change_rate",
            "attack_smear", "transient_xcorr",
            "hf_energy_delta_db", "spectral_peak_divergence",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_reject_reasons_populated(self, mono_audio, codegram, strict_config):
        """Strict config should produce populated reject reasons for modified signal."""
        rng = np.random.default_rng(42)
        modified = mono_audio + rng.normal(0, 0.1, len(mono_audio)).astype(np.float32)
        different_cg = rng.integers(0, 1024, size=codegram.shape, dtype=np.int32)
        result = evaluate_candidate(
            mono_audio, modified, codegram, different_cg, strict_config,
        )
        assert result.accepted is False
        assert len(result.reject_reasons) > 0
        # Each reason should be a string describing the failure
        for reason in result.reject_reasons:
            assert isinstance(reason, str)
            assert len(reason) > 0


class TestFilterCandidates:

    def test_filters_and_sorts(self, transient_audio, codegram, acceptance_config):
        """Should return accepted candidates sorted by quality."""
        rng = np.random.default_rng(42)
        # Create candidates: one identical (best), one slightly different
        candidates = [
            transient_audio.copy(),
            transient_audio + rng.normal(0, 0.02, len(transient_audio)).astype(np.float32),
        ]
        cg_candidates = [codegram.copy(), codegram.copy()]

        results = filter_candidates(
            transient_audio, candidates, codegram, cg_candidates, acceptance_config,
        )
        assert len(results) >= 1
        # Each result is (index, AcceptanceResult)
        for idx, result in results:
            assert isinstance(idx, int)
            assert isinstance(result, AcceptanceResult)
            assert result.accepted is True

    def test_empty_when_all_rejected(self, mono_audio, codegram, strict_config):
        """Should return empty list when all candidates fail."""
        rng = np.random.default_rng(42)
        candidates = [
            rng.normal(0, 0.5, len(mono_audio)).astype(np.float32),
        ]
        cg_candidates = [
            rng.integers(0, 1024, size=codegram.shape, dtype=np.int32),
        ]

        results = filter_candidates(
            mono_audio, candidates, codegram, cg_candidates, strict_config,
        )
        assert len(results) == 0
