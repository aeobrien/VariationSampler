"""Machine-gun test proxy for evaluating perceptual repetition."""

import logging

import numpy as np

from src.utils.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)


def render_machine_gun(
    audio_hits: list[np.ndarray],
    bpm: float = 120.0,
    n_hits: int = 8,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render hits at regular BPM spacing (16th notes) into a single audio buffer.

    Args:
        audio_hits: List of 1D mono audio arrays for each hit.
        bpm: Tempo in beats per minute.
        n_hits: Number of hits to render (cycles through audio_hits).
        sr: Sample rate.

    Returns:
        1D mono audio with hits placed at 16th note intervals.
    """
    sixteenth_note_s = 60.0 / bpm / 4.0
    interval_samples = int(sixteenth_note_s * sr)

    # Total length: n_hits intervals + tail from last hit
    max_hit_len = max(len(h) for h in audio_hits)
    total_samples = (n_hits - 1) * interval_samples + max_hit_len
    output = np.zeros(total_samples, dtype=np.float32)

    for i in range(n_hits):
        hit = audio_hits[i % len(audio_hits)]
        start = i * interval_samples
        end = min(start + len(hit), total_samples)
        output[start:end] += hit[:end - start]

    return output


def extract_hit_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    attack_ms: float = 100,
) -> np.ndarray:
    """Extract perceptual features from a single hit.

    Features: MFCC means (13) + spectral centroid + spectral flux + RMS.

    Args:
        audio: 1D mono audio for one hit.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        attack_ms: Window in ms over which to compute features.

    Returns:
        1D feature vector of length n_mfcc + 3.
    """
    import librosa

    attack_samples = int(attack_ms / 1000.0 * sr)
    segment = audio[:attack_samples]

    if len(segment) == 0:
        return np.zeros(n_mfcc + 3, dtype=np.float32)

    # MFCCs — mean over frames
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Spectral centroid — mean over frames
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # Spectral flux — mean magnitude of frame-to-frame STFT difference
    stft = np.abs(librosa.stft(segment))
    if stft.shape[1] > 1:
        flux = np.mean(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0)))
    else:
        flux = 0.0

    # RMS energy
    rms = float(np.sqrt(np.mean(segment ** 2)))

    features = np.concatenate([
        mfcc_mean,
        [centroid_mean, flux, rms],
    ]).astype(np.float32)

    return features


def compute_pairwise_spectral_distance(
    audio_hits: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> float:
    """Compute mean pairwise log-spectral distance between hits.

    Uses smoothed magnitude spectra to measure how different the hits
    sound from each other. This directly captures perceptual differences
    rather than relying on feature extraction.

    Args:
        audio_hits: List of 1D mono audio arrays.
        sr: Sample rate.

    Returns:
        Mean pairwise distance. 0.0 = identical copies. Higher = more variation.
    """
    n = len(audio_hits)
    if n < 2:
        return 0.0

    # Compute smoothed magnitude spectra
    spectra = []
    for hit in audio_hits:
        fft = np.fft.rfft(hit)
        mag = np.abs(fft).astype(np.float64)
        # Smooth with a small window to reduce noise
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        mag_smooth = np.convolve(mag, kernel, mode="same")
        spectra.append(mag_smooth)

    # Ensure all spectra are the same length
    min_len = min(len(s) for s in spectra)
    spectra = [s[:min_len] for s in spectra]

    # Pairwise log-spectral distance
    eps = 1e-10
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            log_diff = np.log(spectra[i] + eps) - np.log(spectra[j] + eps)
            dist = float(np.sqrt(np.mean(log_diff ** 2)))
            distances.append(dist)

    return float(np.mean(distances))


def compute_self_similarity(features_list: list[np.ndarray]) -> float:
    """Compute mean pairwise cosine similarity across hit feature vectors.

    Note: This metric can be unreliable when features have very different scales.
    Prefer compute_pairwise_spectral_distance for machine-gun evaluation.

    Args:
        features_list: List of 1D feature vectors (one per hit).

    Returns:
        Mean cosine similarity for i!=j pairs. 1.0 = identical copies.
    """
    n = len(features_list)
    if n < 2:
        return 1.0

    features = np.array(features_list, dtype=np.float64)

    # Normalize each vector
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = features / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T

    # Mean of off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(sim_matrix[mask]))


def machine_gun_score(
    audio_hits: list[np.ndarray],
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
    attack_ms: float = 100,
) -> dict:
    """Compute machine-gun perceptual repetition score.

    Uses pairwise log-spectral distance as the primary metric (higher = more
    variation = less machine gun effect). Also computes feature-based cosine
    similarity for backward compatibility.

    Args:
        audio_hits: List of 1D mono audio arrays for each hit.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients for feature extraction.
        attack_ms: Attack window for feature extraction.

    Returns:
        Dict with 'self_similarity' (float, cosine similarity — legacy),
        'spectral_distance' (float, log-spectral distance — primary),
        'n_hits' (int), 'feature_dim' (int), 'features' (list of arrays).
    """
    features = [
        extract_hit_features(h, sr=sr, n_mfcc=n_mfcc, attack_ms=attack_ms)
        for h in audio_hits
    ]

    similarity = compute_self_similarity(features)
    spectral_dist = compute_pairwise_spectral_distance(audio_hits, sr=sr)

    logger.info(
        "Machine-gun score: spectral_dist=%.4f self_similarity=%.4f across %d hits",
        spectral_dist, similarity, len(audio_hits),
    )

    return {
        "self_similarity": similarity,
        "spectral_distance": spectral_dist,
        "n_hits": len(audio_hits),
        "feature_dim": len(features[0]) if features else 0,
        "features": features,
    }
