"""Perceptual distance metrics for drum sample evaluation."""

import logging

import numpy as np

from src.utils.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Multi-resolution STFT config: (fft_size, hop_size, window_name)
MR_STFT_RESOLUTIONS = [
    (512, 88, "hann"),    # ~10ms window, ~2ms hop
    (1024, 220, "hann"),  # ~23ms window, ~5ms hop
    (2048, 441, "hann"),  # ~46ms window, ~10ms hop
]


def _stft_magnitude(
    audio: np.ndarray,
    fft_size: int,
    hop_size: int,
    window: str = "hann",
) -> np.ndarray:
    """Compute STFT magnitude spectrogram."""
    if window == "hann":
        win = np.hanning(fft_size)
    else:
        win = np.ones(fft_size)

    # Pad audio to ensure at least one frame
    if len(audio) < fft_size:
        audio = np.pad(audio, (0, fft_size - len(audio)))

    n_frames = 1 + (len(audio) - fft_size) // hop_size
    stft = np.zeros((fft_size // 2 + 1, max(1, n_frames)))

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + fft_size] * win
        spectrum = np.fft.rfft(frame)
        stft[:, i] = np.abs(spectrum)

    return stft


def multi_resolution_stft_distance(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    resolutions: list[tuple[int, int, str]] | None = None,
    window_samples: int | None = None,
) -> float:
    """Compute multi-resolution STFT distance between two signals.

    Combines spectral convergence and log-magnitude distance across
    multiple resolutions (3 by default: 10ms, 25ms, 50ms windows).

    Args:
        audio_a: 1D mono audio, float32.
        audio_b: 1D mono audio, float32.
        resolutions: List of (fft_size, hop_size, window) tuples.
            Defaults to MR_STFT_RESOLUTIONS.
        window_samples: If set, only analyze this many samples from start.

    Returns:
        Combined multi-resolution STFT distance (lower = more similar).
    """
    if resolutions is None:
        resolutions = MR_STFT_RESOLUTIONS

    if window_samples is not None:
        audio_a = audio_a[:window_samples]
        audio_b = audio_b[:window_samples]

    # Ensure same length
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]

    total_distance = 0.0

    for fft_size, hop_size, window in resolutions:
        mag_a = _stft_magnitude(audio_a, fft_size, hop_size, window)
        mag_b = _stft_magnitude(audio_b, fft_size, hop_size, window)

        # Match shapes
        min_frames = min(mag_a.shape[1], mag_b.shape[1])
        mag_a = mag_a[:, :min_frames]
        mag_b = mag_b[:, :min_frames]

        # Spectral convergence: Frobenius norm of difference / norm of target
        diff_norm = np.linalg.norm(mag_a - mag_b)
        ref_norm = np.linalg.norm(mag_a)
        if ref_norm > 0:
            spectral_convergence = diff_norm / ref_norm
        else:
            spectral_convergence = 0.0

        # Log-magnitude distance
        eps = 1e-7
        log_diff = np.mean(np.abs(np.log(mag_a + eps) - np.log(mag_b + eps)))

        total_distance += spectral_convergence + log_diff

    return total_distance / len(resolutions)


def mfcc_distance(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    window_samples: int | None = None,
) -> float:
    """Compute MFCC distance (Mel-Cepstral Distortion) between two signals.

    Args:
        audio_a: 1D mono audio, float32.
        audio_b: 1D mono audio, float32.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        window_samples: If set, only analyze this many samples from start.

    Returns:
        Mean Euclidean distance between MFCC frames.
    """
    import librosa

    if window_samples is not None:
        audio_a = audio_a[:window_samples]
        audio_b = audio_b[:window_samples]

    mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sr, n_mfcc=n_mfcc)
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr, n_mfcc=n_mfcc)

    # Match number of frames
    min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
    mfcc_a = mfcc_a[:, :min_frames]
    mfcc_b = mfcc_b[:, :min_frames]

    # Mean Euclidean distance across frames
    frame_distances = np.sqrt(np.sum((mfcc_a - mfcc_b) ** 2, axis=0))
    return float(np.mean(frame_distances))


def token_change_rate(
    codegram_a: np.ndarray,
    codegram_b: np.ndarray,
    codebooks: list[int] | None = None,
) -> float:
    """Compute fraction of tokens that differ between two codegrams.

    Args:
        codegram_a: Int array, shape [NQ, T].
        codegram_b: Int array, shape [NQ, T].
        codebooks: Which codebook indices to compare. If None, compares all.

    Returns:
        Fraction of tokens that differ (0.0 = identical, 1.0 = all different).
    """
    if codegram_a.shape != codegram_b.shape:
        raise ValueError(
            f"Shape mismatch: {codegram_a.shape} vs {codegram_b.shape}"
        )

    if codebooks is not None:
        codegram_a = codegram_a[codebooks]
        codegram_b = codegram_b[codebooks]

    total = codegram_a.size
    if total == 0:
        return 0.0

    changed = np.sum(codegram_a != codegram_b)
    return float(changed / total)


def attack_smear_score(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    attack_ms: float = 30,
    sr: int = SAMPLE_RATE,
) -> float:
    """Ratio of transient energy in variation vs input over attack window.

    Values near 1.0 mean transient is preserved. Values <<1 mean smearing.

    Args:
        audio_a: 1D mono audio (source), float32.
        audio_b: 1D mono audio (variation), float32.
        attack_ms: Attack window duration in milliseconds.
        sr: Sample rate.

    Returns:
        Energy ratio (variation / source). ~1.0 is ideal.
    """
    attack_samples = int(attack_ms / 1000.0 * sr)
    a_attack = audio_a[:attack_samples]
    b_attack = audio_b[:attack_samples]

    energy_a = float(np.sum(a_attack ** 2))
    energy_b = float(np.sum(b_attack ** 2))

    if energy_a < 1e-10:
        logger.warning("Source attack energy near zero; returning 0.0")
        return 0.0

    return energy_b / energy_a


def transient_cross_correlation(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    attack_ms: float = 30,
    sr: int = SAMPLE_RATE,
) -> float:
    """Normalized cross-correlation at zero lag on the attack region.

    Values >0.95 mean the transient shape is well preserved.

    Args:
        audio_a: 1D mono audio (source), float32.
        audio_b: 1D mono audio (variation), float32.
        attack_ms: Attack window duration in milliseconds.
        sr: Sample rate.

    Returns:
        Normalized cross-correlation in [-1, 1].
    """
    attack_samples = int(attack_ms / 1000.0 * sr)
    a_attack = audio_a[:attack_samples].astype(np.float64)
    b_attack = audio_b[:attack_samples].astype(np.float64)

    norm_a = np.linalg.norm(a_attack)
    norm_b = np.linalg.norm(b_attack)

    if norm_a < 1e-10 or norm_b < 1e-10:
        logger.warning("Attack region near-silent; returning 0.0")
        return 0.0

    return float(np.dot(a_attack, b_attack) / (norm_a * norm_b))


def high_frequency_energy_delta(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    cutoff_hz: int = 12000,
    sr: int = SAMPLE_RATE,
) -> float:
    """Difference in energy above cutoff between two signals, in dB.

    Positive values mean the variation has more HF energy than source.

    Args:
        audio_a: 1D mono audio (source), float32.
        audio_b: 1D mono audio (variation), float32.
        cutoff_hz: Frequency cutoff in Hz.
        sr: Sample rate.

    Returns:
        Energy delta in dB. 0 = same HF energy.
    """
    min_len = min(len(audio_a), len(audio_b))
    a = audio_a[:min_len]
    b = audio_b[:min_len]

    fft_a = np.fft.rfft(a)
    fft_b = np.fft.rfft(b)

    freqs = np.fft.rfftfreq(min_len, d=1.0 / sr)
    hf_mask = freqs >= cutoff_hz

    energy_a = float(np.sum(np.abs(fft_a[hf_mask]) ** 2))
    energy_b = float(np.sum(np.abs(fft_b[hf_mask]) ** 2))

    eps = 1e-10
    db_a = 10.0 * np.log10(energy_a + eps)
    db_b = 10.0 * np.log10(energy_b + eps)

    return float(db_b - db_a)


def spectral_peak_divergence(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    threshold_db: float = 10.0,
    sr: int = SAMPLE_RATE,
) -> int:
    """Count of prominent spectral peaks in variation absent from source.

    Detects tonal ringing artifacts introduced by the model.

    Args:
        audio_a: 1D mono audio (source), float32.
        audio_b: 1D mono audio (variation), float32.
        threshold_db: Minimum prominence in dB for a peak to be counted.
        sr: Sample rate.

    Returns:
        Number of divergent peaks (0 = no artifacts).
    """
    from scipy.signal import find_peaks

    min_len = min(len(audio_a), len(audio_b))
    a = audio_a[:min_len]
    b = audio_b[:min_len]

    eps = 1e-10
    mag_a = np.abs(np.fft.rfft(a))
    mag_b = np.abs(np.fft.rfft(b))

    # Smooth with a small moving average to reduce noise peaks
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    mag_a_smooth = np.convolve(mag_a, kernel, mode="same")
    mag_b_smooth = np.convolve(mag_b, kernel, mode="same")

    db_a = 20.0 * np.log10(mag_a_smooth + eps)
    db_b = 20.0 * np.log10(mag_b_smooth + eps)

    # Find prominent peaks in variation
    peaks_b, properties = find_peaks(db_b, prominence=threshold_db)

    if len(peaks_b) == 0:
        return 0

    # Check which peaks don't correspond to peaks in source
    # A peak is "divergent" if the source spectrum at that bin is much lower
    divergent = 0
    for peak_idx in peaks_b:
        if db_b[peak_idx] - db_a[peak_idx] > threshold_db:
            divergent += 1

    return divergent


def inter_variation_distances(
    audio_list: list[np.ndarray],
    metric_fn: str = "mrstft",
) -> dict:
    """Compute all pairwise distances between N generated variations.

    Args:
        audio_list: List of 1D mono audio arrays (the generated variations).
        metric_fn: Which metric to use: "mrstft" or "mfcc".

    Returns:
        Dict with "mean", "min", "max", "std", "n_pairs" keys.
    """
    n = len(audio_list)
    if n < 2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n_pairs": 0}

    if metric_fn == "mrstft":
        dist_fn = multi_resolution_stft_distance
    elif metric_fn == "mfcc":
        dist_fn = mfcc_distance
    else:
        raise ValueError(f"Unknown metric_fn: {metric_fn}. Use 'mrstft' or 'mfcc'.")

    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(audio_list[i], audio_list[j])
            distances.append(d)

    arr = np.array(distances)
    result = {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "n_pairs": len(distances),
    }

    logger.info(
        "Inter-variation distances (%s): mean=%.4f, std=%.4f, n_pairs=%d",
        metric_fn, result["mean"], result["std"], result["n_pairs"],
    )
    return result
