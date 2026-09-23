#!/usr/bin/env python3
"""DAC coupling diagnostic: measures bidirectional frame coupling in the
DAC decoder.

Measures two coupling directions:
  1. Backward bleed: edits in the tail region degrade the attack
     (perturb at attack_frames + d, measure attack-region metrics)
  2. Forward bleed: edits in/near the attack region degrade the tail
     (perturb at attack_frames - d, measure tail-region metrics)

For each test codegram, we decode unmodified as reference, then for each
distance d, randomise all editable-codebook tokens at the target frame,
decode again, and measure region metrics vs reference.  Repeated over
multiple trials with different random tokens and averaged.

Usage:
    python3 scripts/dac_coupling_diagnostic.py \
        --codegrams-dir data/codegrams/pass-02 \
        --families CrossStick,SnareRim,HiHat \
        --output reports/dac_coupling.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram
from src.eval.metrics import (
    multi_resolution_stft_distance,
    attack_smear_score,
    transient_cross_correlation,
)
from src.utils.audio import SAMPLE_RATE, NQ, CODEBOOK_SIZE
from src.utils.instrument_families import infer_family

logger = logging.getLogger(__name__)

# Distances (in frames past attack_frames) at which to perturb for backward bleed
DEFAULT_BACKWARD_DISTANCES = [0, 1, 2, 3, 5, 7, 10, 15, 20]
# Distances (in frames before attack_frames) at which to perturb for forward bleed
DEFAULT_FORWARD_DISTANCES = [0, 1, 2, 3, 5, 7, 10, 15, 20]
DEFAULT_N_TRIALS = 10
DEFAULT_ATTACK_FRAMES = 3
DEFAULT_EDIT_CODEBOOKS = [3, 4, 5, 6, 7, 8]
DEFAULT_ATTACK_MS = 30.0


def find_sample_codegrams(
    codegrams_dir: Path,
    families: list[str],
    max_per_family: int = 3,
) -> list[dict]:
    """Find codegram files grouped by instrument family.

    Returns list of dicts with keys: path, family, group_key.
    """
    results = []
    family_counts: dict[str, int] = {}

    for group_dir in sorted(codegrams_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        family = infer_family(group_dir.name)
        if family is None or family not in families:
            continue
        if family_counts.get(family, 0) >= max_per_family:
            continue

        # Take the first hit from this group
        npy_files = sorted(group_dir.glob("*.npy"))
        if not npy_files:
            continue

        results.append({
            "path": npy_files[0],
            "family": family,
            "group_key": group_dir.name,
        })
        family_counts[family] = family_counts.get(family, 0) + 1

    return results


def measure_attack_metrics(
    ref_audio: np.ndarray,
    mod_audio: np.ndarray,
    attack_ms: float = DEFAULT_ATTACK_MS,
) -> dict:
    """Compute attack-region metrics between reference and modified audio."""
    attack_samples = int(attack_ms / 1000.0 * SAMPLE_RATE)
    min_len = min(len(ref_audio), len(mod_audio))
    ref = ref_audio[:min_len]
    mod = mod_audio[:min_len]

    # Attack-window L2
    attack_end = min(attack_samples, min_len)
    l2 = float(np.sqrt(np.mean((ref[:attack_end] - mod[:attack_end]) ** 2)))

    # Attack-window MR-STFT
    mrstft = multi_resolution_stft_distance(
        ref, mod, window_samples=attack_end,
    )

    xcorr = transient_cross_correlation(ref, mod, attack_ms=attack_ms)
    smear = attack_smear_score(ref, mod, attack_ms=attack_ms)

    return {
        "transient_xcorr": round(float(xcorr), 6),
        "attack_smear": round(float(smear), 6),
        "attack_mrstft": round(float(mrstft), 6),
        "attack_l2": round(float(l2), 8),
    }


def measure_tail_metrics(
    ref_audio: np.ndarray,
    mod_audio: np.ndarray,
    attack_ms: float = DEFAULT_ATTACK_MS,
    tail_ms: float = 100.0,
) -> dict:
    """Compute tail-region metrics between reference and modified audio.

    Measures degradation in the region immediately after the attack window,
    used to detect forward bleed from attack-region edits into the tail.
    """
    attack_samples = int(attack_ms / 1000.0 * SAMPLE_RATE)
    tail_samples = int(tail_ms / 1000.0 * SAMPLE_RATE)
    min_len = min(len(ref_audio), len(mod_audio))
    ref = ref_audio[:min_len]
    mod = mod_audio[:min_len]

    tail_start = min(attack_samples, min_len)
    tail_end = min(tail_start + tail_samples, min_len)

    if tail_end <= tail_start:
        return {"tail_l2": 0.0, "tail_mrstft": 0.0, "tail_xcorr": 1.0}

    # Tail-window L2
    l2 = float(np.sqrt(np.mean((ref[tail_start:tail_end] - mod[tail_start:tail_end]) ** 2)))

    # Tail-window MR-STFT (on full signal, but the damage shows in tail)
    mrstft = multi_resolution_stft_distance(ref, mod)

    # Tail-window cross-correlation
    ref_tail = ref[tail_start:tail_end]
    mod_tail = mod[tail_start:tail_end]
    ref_norm = np.sqrt(np.sum(ref_tail ** 2))
    mod_norm = np.sqrt(np.sum(mod_tail ** 2))
    if ref_norm > 1e-10 and mod_norm > 1e-10:
        tail_xcorr = float(np.sum(ref_tail * mod_tail) / (ref_norm * mod_norm))
    else:
        tail_xcorr = 1.0

    return {
        "tail_l2": round(float(l2), 8),
        "tail_mrstft": round(float(mrstft), 6),
        "tail_xcorr": round(float(tail_xcorr), 6),
    }


def _run_perturbation_sweep(
    dac_model,
    codegram: np.ndarray,
    ref_audio: np.ndarray,
    frame_indices: list[tuple[int, int]],
    edit_codebooks: list[int],
    metric_fn,
    n_trials: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Run perturbation trials at each frame index, measure with metric_fn.

    Args:
        frame_indices: list of (distance_label, frame_index) pairs.
        metric_fn: callable(ref_audio, mod_audio) -> dict of metrics.
    """
    t_len = codegram.shape[1]
    results = []

    for dist_label, frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= t_len:
            logger.warning("Frame %d out of range [0, %d), skipping", frame_idx, t_len)
            continue

        trial_metrics: list[dict] = []
        for _trial in range(n_trials):
            modified = codegram.copy()
            for cb in edit_codebooks:
                modified[cb, frame_idx] = rng.integers(0, CODEBOOK_SIZE)

            mod_audio = decode_codegram(dac_model, modified)[0]
            m = metric_fn(ref_audio, mod_audio)
            trial_metrics.append(m)

        # Average across trials
        avg = {}
        for key in trial_metrics[0]:
            values = [t[key] for t in trial_metrics]
            avg[key] = round(float(np.mean(values)), 6)
            avg[f"{key}_std"] = round(float(np.std(values)), 6)

        results.append({
            "distance_frames": dist_label,
            "frame_index": frame_idx,
            **avg,
        })

    return results


def run_coupling_diagnostic(
    dac_model,
    codegram: np.ndarray,
    backward_distances: list[int] | None = None,
    forward_distances: list[int] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    attack_frames: int = DEFAULT_ATTACK_FRAMES,
    edit_codebooks: list[int] | None = None,
    attack_ms: float = DEFAULT_ATTACK_MS,
    seed: int = 42,
) -> dict:
    """Measure bidirectional coupling decay for a single codegram.

    Returns dict with two keys:
      - "backward": edits at attack_frames + d, measuring attack degradation
        (tail edits bleed backward into attack)
      - "forward": edits at attack_frames - 1 - d, measuring tail degradation
        (attack-region edits bleed forward into tail)
    """
    if backward_distances is None:
        backward_distances = DEFAULT_BACKWARD_DISTANCES
    if forward_distances is None:
        forward_distances = DEFAULT_FORWARD_DISTANCES
    if edit_codebooks is None:
        edit_codebooks = DEFAULT_EDIT_CODEBOOKS

    rng = np.random.default_rng(seed)

    # Decode unmodified reference
    ref_audio = decode_codegram(dac_model, codegram)[0]  # 1D mono

    # Backward bleed: perturb tail frames, measure attack degradation
    backward_frames = [(d, attack_frames + d) for d in backward_distances]
    backward_results = _run_perturbation_sweep(
        dac_model, codegram, ref_audio, backward_frames,
        edit_codebooks,
        lambda ref, mod: measure_attack_metrics(ref, mod, attack_ms=attack_ms),
        n_trials, rng,
    )

    # Forward bleed: perturb attack/pre-attack frames, measure tail degradation
    # d=0 means last attack frame, d=1 means second-to-last, etc.
    forward_frames = [
        (d, attack_frames - 1 - d)
        for d in forward_distances
        if attack_frames - 1 - d >= 0
    ]
    forward_results = _run_perturbation_sweep(
        dac_model, codegram, ref_audio, forward_frames,
        edit_codebooks,
        lambda ref, mod: measure_tail_metrics(ref, mod, attack_ms=attack_ms),
        n_trials, rng,
    )

    return {
        "backward": backward_results,
        "forward": forward_results,
    }


def print_coupling_table(coupling_results: dict) -> None:
    """Print formatted tables for both coupling directions."""
    # Backward bleed
    backward = coupling_results["backward"]
    if backward:
        logger.info("--- Backward bleed (tail edits -> attack degradation) ---")
        header = (
            f"{'dist':>4s}  {'frame':>5s}  "
            f"{'xcorr':>8s}  {'smear':>8s}  "
            f"{'mrstft':>8s}  {'L2':>10s}"
        )
        logger.info(header)
        logger.info("-" * len(header))
        for row in backward:
            logger.info(
                "%4d  %5d  %8.4f  %8.4f  %8.4f  %10.6f",
                row["distance_frames"],
                row["frame_index"],
                row["transient_xcorr"],
                row["attack_smear"],
                row["attack_mrstft"],
                row["attack_l2"],
            )

    # Forward bleed
    forward = coupling_results["forward"]
    if forward:
        logger.info("--- Forward bleed (attack edits -> tail degradation) ---")
        header = (
            f"{'dist':>4s}  {'frame':>5s}  "
            f"{'tail_xcorr':>10s}  {'tail_mrstft':>11s}  "
            f"{'tail_L2':>10s}"
        )
        logger.info(header)
        logger.info("-" * len(header))
        for row in forward:
            logger.info(
                "%4d  %5d  %10.4f  %11.4f  %10.6f",
                row["distance_frames"],
                row["frame_index"],
                row["tail_xcorr"],
                row["tail_mrstft"],
                row["tail_l2"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAC coupling diagnostic: measure attack degradation vs edit distance."
    )
    parser.add_argument(
        "--codegrams-dir", type=str, required=True,
        help="Directory containing codegram subdirectories (e.g. data/codegrams/pass-02).",
    )
    parser.add_argument(
        "--families", type=str, default="CrossStick,SnareRim,HiHat",
        help="Comma-separated instrument families to test.",
    )
    parser.add_argument(
        "--output", type=str, default="reports/dac_coupling.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--backward-distances", type=str, default=None,
        help="Comma-separated backward distances to test (default: 0,1,2,3,5,7,10,15,20).",
    )
    parser.add_argument(
        "--forward-distances", type=str, default=None,
        help="Comma-separated forward distances to test (default: 0,1,2,3,5,7,10,15,20).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        help="Number of random-token trials per distance.",
    )
    parser.add_argument(
        "--max-per-family", type=int, default=3,
        help="Maximum codegrams to test per family.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    families = [f.strip() for f in args.families.split(",")]
    backward_distances = (
        [int(x) for x in args.backward_distances.split(",")]
        if args.backward_distances
        else DEFAULT_BACKWARD_DISTANCES
    )
    forward_distances = (
        [int(x) for x in args.forward_distances.split(",")]
        if args.forward_distances
        else DEFAULT_FORWARD_DISTANCES
    )

    codegrams_dir = Path(args.codegrams_dir)
    if not codegrams_dir.is_dir():
        logger.error("Codegrams directory not found: %s", codegrams_dir)
        sys.exit(1)

    # Find samples
    samples = find_sample_codegrams(codegrams_dir, families, args.max_per_family)
    if not samples:
        logger.error("No codegrams found for families: %s", families)
        sys.exit(1)

    logger.info("Found %d samples across families: %s", len(samples), families)

    # Load DAC
    logger.info("Loading DAC model...")
    dac_model = load_dac_model()

    # Run diagnostic
    all_results = []
    for sample_info in samples:
        logger.info(
            "Processing %s (%s)...",
            sample_info["group_key"], sample_info["family"],
        )
        codegram = load_codegram(sample_info["path"])
        coupling = run_coupling_diagnostic(
            dac_model, codegram,
            backward_distances=backward_distances,
            forward_distances=forward_distances,
            n_trials=args.n_trials,
        )

        print_coupling_table(coupling)

        all_results.append({
            "sample": sample_info["group_key"],
            "family": sample_info["family"],
            "codegram_path": str(sample_info["path"]),
            "codegram_shape": list(codegram.shape),
            "backward_coupling": coupling["backward"],
            "forward_coupling": coupling["forward"],
        })

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "backward_distances": backward_distances,
                "forward_distances": forward_distances,
                "n_trials": args.n_trials,
                "attack_frames": DEFAULT_ATTACK_FRAMES,
                "edit_codebooks": DEFAULT_EDIT_CODEBOOKS,
                "attack_ms": DEFAULT_ATTACK_MS,
            },
            "samples": all_results,
        }, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
