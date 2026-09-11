"""Shared test fixtures for VariationSampler."""

import numpy as np
import pytest

from src.utils.audio import SAMPLE_RATE, NQ, T_MAX


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def mono_audio(rng):
    """Synthetic mono audio: 0.5s of noise with a sharp onset at 0.05s."""
    n_samples = SAMPLE_RATE // 2  # 0.5s
    audio = np.zeros(n_samples, dtype=np.float32)
    onset = int(0.05 * SAMPLE_RATE)  # onset at 50ms
    # Sharp transient followed by exponential decay
    t = np.arange(n_samples - onset, dtype=np.float32)
    decay = np.exp(-t / (0.1 * SAMPLE_RATE))
    audio[onset:] = rng.normal(0, 0.3, n_samples - onset).astype(np.float32) * decay
    # Add a click at onset
    audio[onset:onset + 10] = 0.8
    return audio


@pytest.fixture
def stereo_audio(mono_audio):
    """Synthetic stereo audio [2, samples] from mono with slight channel difference."""
    left = mono_audio.copy()
    right = mono_audio * 0.95  # slightly different
    return np.stack([left, right], axis=0)


@pytest.fixture
def audio_with_onset(rng):
    """Mono audio with a clear onset at a known position for alignment tests."""
    n_samples = SAMPLE_RATE  # 1.0s
    audio = np.zeros(n_samples, dtype=np.float32)
    onset_sample = int(0.1 * SAMPLE_RATE)  # onset at 100ms
    # Broadband burst at onset
    burst_len = int(0.005 * SAMPLE_RATE)
    audio[onset_sample:onset_sample + burst_len] = rng.normal(0, 0.8, burst_len).astype(np.float32)
    # Decaying noise after burst
    tail_len = n_samples - onset_sample - burst_len
    if tail_len > 0:
        t = np.arange(tail_len, dtype=np.float32)
        decay = np.exp(-t / (0.15 * SAMPLE_RATE))
        audio[onset_sample + burst_len:] = rng.normal(0, 0.2, tail_len).astype(np.float32) * decay
    return audio


@pytest.fixture
def codegram(rng):
    """Synthetic codegram [NQ, T_MAX] with random token values."""
    return rng.integers(0, 1024, size=(NQ, T_MAX), dtype=np.int32)


@pytest.fixture
def codegram_pair(rng):
    """Pair of codegrams with small differences (simulating RR variation)."""
    base = rng.integers(0, 1024, size=(NQ, T_MAX), dtype=np.int32)
    varied = base.copy()
    # Change ~5% of tokens in codebooks 5-8
    for cb in range(5, 9):
        mask = rng.random(T_MAX) < 0.05
        varied[cb, mask] = rng.integers(0, 1024, size=mask.sum(), dtype=np.int32)
    return base, varied


@pytest.fixture
def sample_group_keys():
    """Sample group key strings for split testing."""
    keys = {}
    for lib in ["libA", "libB", "libC"]:
        for kit in ["kit1", "kit2"]:
            for inst in ["snare", "kick"]:
                key = f"{lib}/{kit}/{inst}/hit/default"
                keys[key] = [f"data/{lib}/{kit}/{inst}_rr{i}.npy" for i in range(5)]
    return keys
