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


def match_level(
    audio: np.ndarray,
    reference: np.ndarray,
    window_ms: float = 50,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Match RMS level of audio to reference over attack window.

    Args:
        audio: 1D mono audio to adjust, float32.
        reference: 1D mono reference audio, float32.
        window_ms: Window in ms over which to measure RMS.
        sr: Sample rate.

    Returns:
        Level-matched audio, same shape.
    """
    window_samples = int(window_ms / 1000.0 * sr)
    ref_segment = reference[:window_samples]
    audio_segment = audio[:window_samples]

    rms_ref = float(np.sqrt(np.mean(ref_segment ** 2)))
    rms_audio = float(np.sqrt(np.mean(audio_segment ** 2)))

    if rms_audio < 1e-10:
        logger.warning("Audio RMS near zero; skipping level match")
        return audio

    gain = rms_ref / rms_audio
    logger.debug("Level match gain: %.4f", gain)
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
    window_ms = config.get("window_ms", 50)
    fade_ms = config.get("fade_ms", 10)

    result = remove_dc(audio)
    result = match_level(result, reference, window_ms=window_ms)
    result = fade_tail(result, fade_ms=fade_ms)

    logger.info("Post-processing complete: %d samples", len(result))
    return result
