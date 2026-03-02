"""Compute and save ground-truth baseline metric distributions from real RR data.

Usage:
    python3 scripts/compute_baselines.py --data-dir data/processed/pass-02 --output-dir data/baselines
    python3 scripts/compute_baselines.py --max-sets-per-family 20  # faster, for testing
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.baselines import compute_baseline_for_family, save_baseline
from src.utils.audio import load_wav

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Pattern: Track{N}_{Instrument}_v{velocity}
DIR_PATTERN = re.compile(r"^Track(\d+)_(.+)_v(\d+)$")

# Map instrument names to families
INSTRUMENT_FAMILIES = {
    "Snare": "Snare",
    "SnareRim": "Snare",
    "Kick": "Kick",
    "HiHatClosed": "HiHat",
    "HiHatOpen": "HiHat",
    "PedalHat": "HiHat",
    "Rimshot": "Rimshot",
    "CrossStick": "CrossStick",
    "Tom": "Tom",
    "TomHi": "Tom",
    "TomLo": "Tom",
    "TomFloor": "Tom",
    "Crash": "Cymbal",
    "Ride": "Cymbal",
}


def discover_families(data_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Discover instrument families and round-robin sets from directory structure.

    Expects flat directories: data_dir/Track{N}_{Instrument}_v{velocity}/hit_*.wav
    Each directory is one round-robin set (10 hits of the same drum at same velocity).
    Groups by instrument family across all tracks and velocities.

    Returns:
        Mapping from family_name -> {set_id -> [wav_paths]}.
    """
    families: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        match = DIR_PATTERN.match(subdir.name)
        if not match:
            logger.debug("Skipping non-matching directory: %s", subdir.name)
            continue

        track_num = match.group(1)
        instrument = match.group(2)
        velocity = match.group(3)

        family = INSTRUMENT_FAMILIES.get(instrument, instrument)
        set_id = subdir.name  # e.g. "Track1_Snare_v127"

        wav_files = sorted(subdir.glob("*.wav"))
        if wav_files:
            families[family][set_id] = wav_files

    return dict(families)


def load_family_audio(
    family_sets: dict[str, list[Path]],
    min_set_size: int = 2,
    max_sets: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load audio for all sets in a family, filtering out small sets.

    Args:
        family_sets: Mapping from set_id to list of WAV paths.
        min_set_size: Minimum number of samples in a set to include.
        max_sets: If set, limit to this many sets (for faster testing).

    Returns:
        Mapping from set_id to list of mono audio arrays.
    """
    result: dict[str, list[np.ndarray]] = {}
    count = 0

    for set_id, paths in sorted(family_sets.items()):
        if max_sets is not None and count >= max_sets:
            break

        if len(paths) < min_set_size:
            continue

        audios = []
        for path in paths:
            try:
                audio = load_wav(path)
                # Convert to mono if stereo
                if audio.shape[0] > 1:
                    audio = np.mean(audio, axis=0)
                else:
                    audio = audio[0]
                audios.append(audio)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                continue

        if len(audios) >= min_set_size:
            result[set_id] = audios
            count += 1

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline metric distributions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/pass-02"),
        help="Directory containing processed audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/baselines"),
        help="Directory to save baseline distributions",
    )
    parser.add_argument(
        "--min-set-size",
        type=int,
        default=2,
        help="Minimum round-robin set size to include",
    )
    parser.add_argument(
        "--max-sets-per-family",
        type=int,
        default=None,
        help="Limit sets per family (for faster runs)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering instrument families in %s", args.data_dir)
    families = discover_families(args.data_dir)

    if not families:
        logger.error("No instrument families found")
        sys.exit(1)

    logger.info(
        "Found %d families: %s",
        len(families),
        {k: len(v) for k, v in sorted(families.items())},
    )

    # Summary table header
    print(f"\n{'Family':<15} {'Sets':>6} {'Samples':>8} {'Pairs':>7} {'MR-STFT med':>12} {'MFCC med':>10}")
    print("-" * 62)

    for family_name, family_sets in sorted(families.items()):
        logger.info(
            "Processing family: %s (%d sets)",
            family_name, len(family_sets),
        )

        family_audio = load_family_audio(
            family_sets,
            min_set_size=args.min_set_size,
            max_sets=args.max_sets_per_family,
        )
        if not family_audio:
            logger.warning(
                "No valid sets for family %s (min_set_size=%d)",
                family_name, args.min_set_size,
            )
            continue

        total_samples = sum(len(v) for v in family_audio.values())
        mrstft_median = 0.0
        mfcc_median = 0.0
        n_pairs = 0

        for metric_fn in ["mrstft", "mfcc"]:
            dist = compute_baseline_for_family(family_audio, metric_fn=metric_fn)
            if dist.values:
                output_path = args.output_dir / f"{family_name}_{metric_fn}.json"
                save_baseline(dist, output_path)

                if metric_fn == "mrstft":
                    mrstft_median = dist.median
                    n_pairs = len(dist.values)
                elif metric_fn == "mfcc":
                    mfcc_median = dist.median

        print(
            f"{family_name:<15} {len(family_audio):>6} {total_samples:>8} "
            f"{n_pairs:>7} {mrstft_median:>12.4f} {mfcc_median:>10.4f}"
        )

    print()
    logger.info("Baselines saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
