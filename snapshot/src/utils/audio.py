"""Shared audio constants and utilities for VariationSampler."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# DAC 44.1 kHz codec constants
SAMPLE_RATE = 44100
NQ = 9                    # Number of RVQ codebooks
CODEBOOK_SIZE = 1024      # Entries per codebook
STRIDE = 512              # DAC encoder stride in samples
FRAME_RATE = 86           # Latent frames per second (44100 / 512 ≈ 86)
MAX_AUDIO_LENGTH_S = 2.5  # V1 max audio length in seconds
T_MAX = 215               # Max codegram frames for 2.5s (ceil(2.5 * 86))
MAX_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_LENGTH_S)  # 66150


def load_wav(path: Path) -> np.ndarray:
    """Load a WAV file as float32 [channels, samples] at 44.1 kHz.

    Args:
        path: Path to WAV file.

    Returns:
        Audio array, shape [channels, samples], float32.

    Raises:
        ValueError: If sample rate is not 44100 Hz.
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # sf.read returns [samples, channels] — transpose to [channels, samples]
    audio = audio.T

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. "
            f"Please convert '{path.name}' to 44.1 kHz before processing."
        )

    logger.info(
        "Loaded: %s (%d ch, %d samples, %.3fs)",
        path.name, audio.shape[0], audio.shape[1], audio.shape[1] / sr,
    )
    return audio


def save_wav(audio: np.ndarray, path: Path, subtype: str = "PCM_16") -> None:
    """Save audio as WAV file.

    Args:
        audio: Float32 audio, shape [channels, samples].
        path: Output file path.
        subtype: WAV subtype (default PCM_16).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        logger.warning("Audio peak %.4f exceeds 1.0, clipping before save", peak)
        audio = np.clip(audio, -1.0, 1.0)

    # soundfile expects [samples, channels]
    sf.write(str(path), audio.T, SAMPLE_RATE, subtype=subtype)
    logger.info("Saved: %s (%.3fs)", path.name, audio.shape[1] / SAMPLE_RATE)


def pad_or_truncate(audio: np.ndarray, target_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Pad with zeros or truncate audio to exact target length.

    Args:
        audio: Float32 audio, shape [channels, samples].
        target_samples: Target number of samples.

    Returns:
        Audio array, shape [channels, target_samples], float32.
    """
    n_channels, n_samples = audio.shape

    if n_samples == target_samples:
        return audio
    elif n_samples > target_samples:
        logger.debug("Truncating %d -> %d samples", n_samples, target_samples)
        return audio[:, :target_samples]
    else:
        logger.debug("Padding %d -> %d samples", n_samples, target_samples)
        padded = np.zeros((n_channels, target_samples), dtype=audio.dtype)
        padded[:, :n_samples] = audio
        return padded


def validate_audio_shape(audio: np.ndarray) -> None:
    """Validate audio array shape and dtype conventions.

    Args:
        audio: Array to validate.

    Raises:
        ValueError: If shape or dtype is invalid.
    """
    if audio.ndim != 2:
        raise ValueError(
            f"Audio must be 2D [channels, samples], got {audio.ndim}D shape {audio.shape}"
        )
    if audio.shape[0] > audio.shape[1]:
        raise ValueError(
            f"Audio shape {audio.shape} looks transposed — "
            f"expected [channels, samples] where channels < samples"
        )
    if audio.dtype not in (np.float32, np.float64):
        raise ValueError(f"Audio must be float32 or float64, got {audio.dtype}")
