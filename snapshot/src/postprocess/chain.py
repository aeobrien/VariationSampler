"""Post-processing chain for generated drum sample variations."""

import logging

import numpy as np

from src.utils.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting per-channel mean.

    Args:
        audio: 1D mono audio, float32.

    Returns:
        DC-removed audio, same shape.
    """
    return audio - np.mean(audio)


def match_peak(
    audio: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match peak amplitude of audio to reference.

    For percussive material, the transient peak defines perceived loudness.
    RMS matching over-compensates when the attack is softer in the variation,
    boosting the body/sustain disproportionately. Peak matching preserves the
    attack-to-body ratio by aligning the loudest sample.

    Args:
        audio: 1D mono audio to adjust, float32.
        reference: 1D mono reference audio, float32.

    Returns:
        Peak-matched audio, same shape.
    """
    peak_ref = float(np.max(np.abs(reference)))
    peak_audio = float(np.max(np.abs(audio)))

    if peak_audio < 1e-10:
        logger.warning("Audio peak near zero; skipping peak match")
        return audio

    gain = peak_ref / peak_audio
    logger.debug("Peak match gain: %.4f", gain)
    return audio * gain


def fade_tail(
    audio: np.ndarray,
    fade_ms: float = 10,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply cosine fade-out at end of audio to prevent click.

    Args:
        audio: 1D mono audio, float32.
        fade_ms: Fade duration in milliseconds.
        sr: Sample rate.

    Returns:
        Audio with fade applied at tail.
    """
    fade_samples = int(fade_ms / 1000.0 * sr)
    fade_samples = min(fade_samples, len(audio))

    result = audio.copy()
    if fade_samples > 0:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        result[-fade_samples:] *= fade.astype(result.dtype)

    return result


def dither_to_16bit(audio: np.ndarray) -> np.ndarray:
    """Apply TPDF dither and convert to int16.

    Args:
        audio: 1D mono audio, float32, expected range [-1, 1].

    Returns:
        int16 array with TPDF dither applied.
    """
    max_val = 32767.0

    # TPDF dither: sum of two uniform random values (triangular distribution)
    rng = np.random.default_rng()
    dither = rng.uniform(-1.0, 1.0, len(audio)) + rng.uniform(-1.0, 1.0, len(audio))

    scaled = audio * max_val + dither
    clipped = np.clip(scaled, -32768.0, 32767.0)
    return clipped.astype(np.int16)


def postprocess(
    audio: np.ndarray,
    reference: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Run full post-processing chain: DC removal, level match, tail fade.

    Does NOT apply dither — that is a final export step.

    Args:
        audio: 1D mono audio, float32.
        reference: 1D mono reference audio, float32.
        config: Dict with optional keys 'window_ms' and 'fade_ms'.

    Returns:
        Post-processed float32 audio.
    """
    fade_ms = config.get("fade_ms", 10)

    result = remove_dc(audio)
    result = match_peak(result, reference)
    result = fade_tail(result, fade_ms=fade_ms)

    logger.debug("Post-processing complete: %d samples", len(result))
    return result
