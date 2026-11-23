#!/usr/bin/env python3
"""
Procedural Baseline Variation Generator (Gate 0)

Generates perceptually convincing round-robin variations from a single
one-shot drum sample using simple DSP techniques. Produces individual
variation WAVs and a machine-gun test WAV for listening evaluation.

Usage:
    # Single file
    python scripts/procedural_baseline.py input.wav --n-variations 8 --output-dir outputs/procedural/

    # Batch (directory of WAVs)
    python scripts/procedural_baseline.py input_dir/ --n-variations 8 --output-dir outputs/procedural/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Variation techniques
# ---------------------------------------------------------------------------

def apply_gain_jitter(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply ±0.3 dB random gain change."""
    db = rng.uniform(-0.3, 0.3)
    gain = 10.0 ** (db / 20.0)
    logger.debug("Gain jitter: %.4f dB (factor %.6f)", db, gain)
    return audio * gain


def apply_pitch_shift(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply ±3 cents pitch shift via librosa with short STFT window."""
    import librosa

    cents = rng.uniform(-3.0, 3.0)
    semitones = cents / 100.0
    logger.debug("Pitch shift: %.2f cents", cents)

    # Process each channel with the same shift
    channels = []
    for ch in range(audio.shape[0]):
        shifted = librosa.effects.pitch_shift(
            audio[ch],
            sr=SAMPLE_RATE,
            n_steps=semitones,
            n_fft=512,
        )
        channels.append(shifted)

    result = np.stack(channels, axis=0)
    # pitch_shift can change length slightly — match original
    if result.shape[1] > audio.shape[1]:
        result = result[:, : audio.shape[1]]
    elif result.shape[1] < audio.shape[1]:
        pad = np.zeros((result.shape[0], audio.shape[1] - result.shape[1]), dtype=result.dtype)
        result = np.concatenate([result, pad], axis=1)
    return result


def apply_timing_offset(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Circular shift by random integer in [-5, +5] samples."""
    shift = int(rng.integers(-5, 6))
    logger.debug("Timing offset: %d samples", shift)
    return np.roll(audio, shift, axis=1)


def apply_transient_shaping(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light time-stretch of the attack window (~5ms) by factor 0.97–1.03.

    Uses scipy resampling on the attack portion, then crossfades back
    to the original at the boundary.
    """
    attack_samples = int(0.005 * SAMPLE_RATE)  # ~5ms = 220 samples
    stretch_factor = rng.uniform(0.97, 1.03)
    logger.debug("Transient shaping: factor %.4f on first %d samples", stretch_factor, attack_samples)

    if audio.shape[1] < attack_samples * 2:
        logger.debug("Audio too short for transient shaping, skipping")
        return audio

    result = audio.copy()
    attack = audio[:, :attack_samples]

    # Resample attack region to apply time-stretch
    new_len = max(1, int(round(attack_samples * stretch_factor)))
    stretched_attack = np.zeros((audio.shape[0], new_len), dtype=audio.dtype)
    for ch in range(audio.shape[0]):
        stretched_attack[ch] = resample(attack[ch], new_len)

    # Fit stretched attack back into original length with crossfade
    crossfade_len = min(32, attack_samples // 4, new_len)

    if new_len <= attack_samples:
        # Stretched attack is shorter or equal — pad then crossfade
        padded = np.zeros_like(attack)
        padded[:, :new_len] = stretched_attack
        # Crossfade at boundary between stretched and original
        fade_start = max(0, new_len - crossfade_len)
        fade = np.linspace(1.0, 0.0, crossfade_len)
        for ch in range(audio.shape[0]):
            padded[ch, fade_start:fade_start + crossfade_len] *= fade
            orig_blend = audio[ch, fade_start:fade_start + crossfade_len] * (1.0 - fade)
            padded[ch, fade_start:fade_start + crossfade_len] += orig_blend
            padded[ch, fade_start + crossfade_len:attack_samples] = audio[ch, fade_start + crossfade_len:attack_samples]
        result[:, :attack_samples] = padded
    else:
        # Stretched attack is longer — truncate and crossfade
        truncated = stretched_attack[:, :attack_samples]
        fade_start = attack_samples - crossfade_len
        fade = np.linspace(1.0, 0.0, crossfade_len)
        for ch in range(audio.shape[0]):
            truncated[ch, fade_start:] *= fade
            truncated[ch, fade_start:] += audio[ch, fade_start:attack_samples] * (1.0 - fade)
        result[:, :attack_samples] = truncated

    return result


def apply_micro_ir(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a short random FIR filter for subtle tonal colouring.

    Starts from a unit impulse and adds small Hann-windowed Gaussian
    perturbations, keeping the filter close to identity so peak levels
    stay within a few percent of the original.
    """
    n_taps = 32
    # Start with identity (unit impulse at tap 0)
    fir = np.zeros(n_taps)
    fir[0] = 1.0
    # Add small random perturbations — scale controls subtlety
    perturbation_scale = 0.015
    perturbation = rng.normal(0.0, perturbation_scale, n_taps)
    window = np.hanning(n_taps)
    fir += perturbation * window
    # Normalise to unit sum to preserve DC level
    fir /= np.sum(fir)

    logger.debug("Micro-IR: 32-tap FIR, energy=%.6f", np.sum(fir ** 2))

    result = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        convolved = fftconvolve(audio[ch], fir, mode="full")
        result[ch] = convolved[: audio.shape[1]]
    return result


# ---------------------------------------------------------------------------
# Variation pipeline
# ---------------------------------------------------------------------------

def generate_variation(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply all variation techniques to produce one variation.

    Args:
        audio: Input audio, shape [channels, samples], float64.
        rng: Random number generator for this variation.

    Returns:
        Variation audio, same shape as input, float64.
    """
    v = audio.copy()
    v = apply_gain_jitter(v, rng)
    v = apply_pitch_shift(v, rng)
    v = apply_timing_offset(v, rng)
    v = apply_transient_shaping(v, rng)
    v = apply_micro_ir(v, rng)
    return v


# ---------------------------------------------------------------------------
# Machine-gun renderer
# ---------------------------------------------------------------------------

def render_machine_gun(
    variations: list[np.ndarray],
    sr: int = SAMPLE_RATE,
    bpm: float = 120.0,
) -> np.ndarray:
    """Render variations as a rapid 16th-note sequence.

    Args:
        variations: List of audio arrays, each [channels, samples].
        sr: Sample rate.
        bpm: Tempo in BPM. 16th notes at 120 BPM = 125ms inter-onset.

    Returns:
        Rendered sequence as [channels, samples] float64.
    """
    sixteenth_interval = 60.0 / bpm / 4.0  # seconds per 16th note
    onset_samples = int(round(sixteenth_interval * sr))
    n_channels = variations[0].shape[0]

    # Total length: last onset + longest tail
    max_tail = max(v.shape[1] for v in variations)
    total_samples = onset_samples * (len(variations) - 1) + max_tail
    output = np.zeros((n_channels, total_samples), dtype=np.float64)

    for i, v in enumerate(variations):
        start = onset_samples * i
        end = start + v.shape[1]
        output[:, start:end] += v

    logger.info(
        "Rendered machine-gun: %d hits, %.1f BPM, %.2fs total",
        len(variations), bpm, total_samples / sr,
    )
    return output


# ---------------------------------------------------------------------------
# TPDF dither + 16-bit export
# ---------------------------------------------------------------------------

def float_to_int16_dither(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert float64 audio to int16 with TPDF dither.

    Args:
        audio: Float audio, expected range [-1, 1].
        rng: Random number generator.

    Returns:
        int16 numpy array.
    """
    # TPDF dither: sum of two uniform random variables in [-0.5, 0.5] LSB
    dither = rng.uniform(-0.5, 0.5, audio.shape) + rng.uniform(-0.5, 0.5, audio.shape)
    # Scale to 16-bit range
    scaled = audio * 32767.0 + dither
    clipped = np.clip(scaled, -32768, 32767)
    return clipped.astype(np.int16)


def export_wav(
    audio: np.ndarray,
    path: Path,
    sr: int = SAMPLE_RATE,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Export float64 audio as 16-bit stereo WAV with TPDF dither.

    Args:
        audio: Float64 audio, shape [channels, samples].
        path: Output file path.
        sr: Sample rate.
        rng: Random number generator for dither. Uses default if None.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clip to prevent overflow
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        logger.warning("Audio peak %.4f exceeds 1.0, clipping", peak)
        audio = np.clip(audio, -1.0, 1.0)

    int16_audio = float_to_int16_dither(audio, rng)
    # soundfile expects [samples, channels]
    sf.write(str(path), int16_audio.T, sr, subtype="PCM_16")
    logger.info("Exported: %s (%.2fs, %d channels)", path.name, audio.shape[1] / sr, audio.shape[0])


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
    """Load a WAV file as float64 [channels, samples] at 44.1 kHz.

    Args:
        path: Path to WAV file.

    Returns:
        Audio array, shape [channels, samples], float64.

    Raises:
        ValueError: If sample rate is not 44100 Hz.
    """
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    # sf.read returns [samples, channels] — transpose to [channels, samples]
    audio = audio.T

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. "
            f"Please convert '{path.name}' to 44.1 kHz before processing."
        )

    logger.info(
        "Loaded: %s (%d channels, %d samples, %.3fs)",
        path.name, audio.shape[0], audio.shape[1], audio.shape[1] / sr,
    )
    return audio


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_sample(
    input_path: Path,
    output_dir: Path,
    n_variations: int,
    seed: int,
) -> None:
    """Process a single drum sample: generate variations + machine-gun tests.

    Args:
        input_path: Path to input WAV.
        output_dir: Directory for output files.
        n_variations: Number of variations to generate.
        seed: Random seed for reproducibility.
    """
    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = load_wav(input_path)
    rng = np.random.default_rng(seed)

    # Generate variations
    variations = []
    for i in range(n_variations):
        # Each variation gets its own child RNG for independence
        var_seed = rng.integers(0, 2**63)
        var_rng = np.random.default_rng(var_seed)
        logger.info("Generating variation %d/%d (seed=%d)", i + 1, n_variations, var_seed)
        v = generate_variation(audio, var_rng)
        variations.append(v)

        var_path = output_dir / f"{stem}_var_{i + 1:02d}.wav"
        export_wav(v, var_path, rng=np.random.default_rng(var_seed + 1))

    # Machine-gun test: original + all variations, cycling
    logger.info("Rendering machine-gun test (variations)")
    all_hits = [audio] + variations
    mg_variations = render_machine_gun(all_hits)
    mg_var_path = output_dir / f"{stem}_machine_gun_variations.wav"
    export_wav(mg_variations, mg_var_path, rng=rng)

    # Machine-gun test: copies only (identical sample repeated)
    logger.info("Rendering machine-gun test (copies)")
    copies = [audio] * (n_variations + 1)
    mg_copies = render_machine_gun(copies)
    mg_copies_path = output_dir / f"{stem}_machine_gun_copies.wav"
    export_wav(mg_copies, mg_copies_path, rng=rng)

    logger.info("Done processing '%s': %d variations + 2 machine-gun tests", stem, n_variations)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate procedural round-robin variations from a drum sample.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input WAV file or directory of WAV files.",
    )
    parser.add_argument(
        "--n-variations",
        type=int,
        default=8,
        help="Number of variations to generate (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/procedural"),
        help="Output directory (default: outputs/procedural/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path: Path = args.input

    if input_path.is_dir():
        # Batch mode
        wav_files = sorted(input_path.glob("*.wav"))
        if not wav_files:
            logger.error("No .wav files found in '%s'", input_path)
            sys.exit(1)
        logger.info("Batch mode: found %d WAV files in '%s'", len(wav_files), input_path)
        for i, wav_path in enumerate(wav_files):
            logger.info("Processing file %d/%d: %s", i + 1, len(wav_files), wav_path.name)
            process_sample(wav_path, args.output_dir, args.n_variations, args.seed + i)
    elif input_path.is_file():
        process_sample(input_path, args.output_dir, args.n_variations, args.seed)
    else:
        logger.error("Input '%s' is not a file or directory", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
