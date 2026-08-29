#!/usr/bin/env python3
"""Import training data from AD2 capture passes.

Scans raw capture directories, parses filenames, verifies duplicates,
analyzes durations, runs preprocessing, generates splits, and optionally
encodes through DAC.

Usage:
    # Step 1: Scan and analyze (no processing)
    python scripts/import_training_data.py scan training-data/regions/pass-01

    # Step 2: Preprocess (trim, align, normalize, pad)
    python scripts/import_training_data.py preprocess training-data/regions/pass-01

    # Step 3: Generate splits
    python scripts/import_training_data.py splits --test-tracks Track19 Track20

    # Step 4: Encode through DAC (requires GPU)
    python scripts/import_training_data.py encode
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.import_samples import (
    scan_pass_directory,
    verify_duplicates,
    analyze_durations,
    write_manifest_csv,
)
from src.data.preprocessing import preprocess_sample
from src.data.splits import generate_splits, verify_no_leakage, SplitManifest, generate_training_pairs
from src.utils.audio import load_wav, save_wav, SAMPLE_RATE, MAX_SAMPLES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a pass directory: parse, check duplicates, analyze durations."""
    pass_dir = Path(args.pass_dir)
    pass_id = pass_dir.name

    logger.info("=== Scanning %s ===", pass_dir)
    sets = scan_pass_directory(pass_dir, pass_id=pass_id)

    # Summary by instrument
    by_instrument: dict[str, list] = {}
    for rr_set in sets.values():
        by_instrument.setdefault(rr_set.instrument, []).append(rr_set)

    logger.info("--- Summary ---")
    for inst in sorted(by_instrument):
        rr_sets = by_instrument[inst]
        sizes = [len(s.samples) for s in rr_sets]
        logger.info(
            "  %s: %d sets, %d total samples, sizes %d-%d (median %d)",
            inst, len(rr_sets), sum(sizes),
            min(sizes), max(sizes), int(np.median(sizes)),
        )

    # Duplicate verification
    if args.check_duplicates:
        logger.info("--- Duplicate Check ---")
        dup_count = 0
        for group_key, rr_set in sorted(sets.items()):
            if len(rr_set.samples) < 2:
                continue
            dups = verify_duplicates(rr_set, threshold=args.dup_threshold)
            if dups:
                dup_count += len(dups)
                for i, j, corr in dups:
                    logger.warning(
                        "  Duplicate in %s: rr%02d <-> rr%02d (corr=%.6f)",
                        group_key,
                        rr_set.samples[i].rr_index,
                        rr_set.samples[j].rr_index,
                        corr,
                    )
        logger.info("  Total duplicate pairs found: %d", dup_count)

    # Duration analysis
    logger.info("--- Duration Analysis ---")
    durations = analyze_durations(sets)
    dur_path = DATA_DIR / "analysis" / "durations.json"
    dur_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dur_path, "w") as f:
        json.dump(durations, f, indent=2)
    logger.info("Duration analysis saved to %s", dur_path)

    # Write manifest
    manifest_path = DATA_DIR / "manifests" / f"manifest_{pass_id}.csv"
    write_manifest_csv(sets, manifest_path)


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run full preprocessing pipeline on a pass directory."""
    pass_dir = Path(args.pass_dir)
    pass_id = pass_dir.name
    output_dir = DATA_DIR / "processed" / pass_id

    logger.info("=== Preprocessing %s -> %s ===", pass_dir, output_dir)
    sets = scan_pass_directory(pass_dir, pass_id=pass_id)

    target_samples = int(args.pad_length * SAMPLE_RATE)
    processed_count = 0
    skipped_count = 0

    for group_key in sorted(sets.keys()):
        rr_set = sets[group_key]

        if len(rr_set.samples) < args.min_set_size:
            logger.debug("Skipping %s: only %d samples", group_key, len(rr_set.samples))
            skipped_count += 1
            continue

        # Use first sample as reference for alignment
        samples = sorted(rr_set.samples, key=lambda s: s.rr_index)
        reference_audio = load_wav(samples[0].filepath)

        group_dir = output_dir / group_key
        group_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            audio = load_wav(sample.filepath)

            ref = reference_audio if i > 0 else None
            processed = preprocess_sample(
                audio,
                reference=ref,
                target_samples=target_samples,
                target_db=args.target_db,
            )

            out_path = group_dir / f"hit_{i + 1:02d}.wav"
            save_wav(processed, out_path)
            processed_count += 1

    logger.info(
        "Preprocessing complete: %d samples processed, %d sets skipped (<%d samples)",
        processed_count, skipped_count, args.min_set_size,
    )


def cmd_splits(args: argparse.Namespace) -> None:
    """Generate train/dev/test splits from processed data."""
    processed_dir = DATA_DIR / "processed"

    if not processed_dir.exists():
        logger.error("No processed data found at %s. Run 'preprocess' first.", processed_dir)
        sys.exit(1)

    # Build groups from processed directory structure
    # Store paths relative to PROJECT_ROOT for portability across machines
    groups: dict[str, list[str]] = {}
    for pass_dir in sorted(processed_dir.iterdir()):
        if not pass_dir.is_dir():
            continue
        for group_dir in sorted(pass_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_key = f"{pass_dir.name}/{group_dir.name}"
            wav_files = sorted(
                str(p.relative_to(PROJECT_ROOT))
                for p in group_dir.glob("hit_*.wav")
            )
            if wav_files:
                groups[group_key] = wav_files

    logger.info("Found %d groups with processed audio", len(groups))

    # Determine test tracks — match track names within group keys
    # Group keys are "pass-01/Track19_Snare_v127", so we need to check
    # if any test track name appears in the key after the pass prefix.
    test_tracks = args.test_tracks or []
    logger.info("Holding out tracks for test: %s", test_tracks)

    # Partition manually since generate_splits matches on first /-component
    test_groups: dict[str, list[str]] = {}
    non_test_groups: dict[str, list[str]] = {}
    for key, files in groups.items():
        # Extract track name: "pass-01/Track19_Snare_v127" -> "Track19"
        group_part = key.split("/")[-1]  # "Track19_Snare_v127"
        track_name = group_part.split("_")[0]  # "Track19"
        if track_name in test_tracks:
            test_groups[key] = files
        else:
            non_test_groups[key] = files

    # Split non-test into train/dev using generate_splits with no test holdout
    manifest = generate_splits(
        non_test_groups,
        test_libraries=[],  # already partitioned
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    manifest.test = test_groups
    verify_no_leakage(manifest)

    splits_dir = DATA_DIR / "splits"
    manifest.save(splits_dir / "manifest.json")

    # Generate and save training pairs
    all_train_pairs = []
    for group_key, files in manifest.train.items():
        pairs = generate_training_pairs(files)
        all_train_pairs.extend(pairs)

    all_dev_pairs = []
    for group_key, files in manifest.dev.items():
        pairs = generate_training_pairs(files)
        all_dev_pairs.extend(pairs)

    pairs_info = {
        "train_pairs": len(all_train_pairs),
        "dev_pairs": len(all_dev_pairs),
        "train_groups": len(manifest.train),
        "dev_groups": len(manifest.dev),
        "test_groups": len(manifest.test),
    }
    logger.info(
        "Training pairs: %d train, %d dev (%d/%d/%d groups)",
        pairs_info["train_pairs"], pairs_info["dev_pairs"],
        pairs_info["train_groups"], pairs_info["dev_groups"], pairs_info["test_groups"],
    )

    # Save pairs lists
    with open(splits_dir / "train_pairs.json", "w") as f:
        json.dump(all_train_pairs, f)
    with open(splits_dir / "dev_pairs.json", "w") as f:
        json.dump(all_dev_pairs, f)
    with open(splits_dir / "split_info.json", "w") as f:
        json.dump(pairs_info, f, indent=2)


def cmd_encode(args: argparse.Namespace) -> None:
    """Encode processed audio through DAC and cache codegrams."""
    from src.data.codegram_cache import load_dac_model, encode_and_cache

    processed_dir = DATA_DIR / "processed"
    codegram_dir = DATA_DIR / "codegrams"

    if not processed_dir.exists():
        logger.error("No processed data found. Run 'preprocess' first.")
        sys.exit(1)

    logger.info("Loading DAC model...")
    model = load_dac_model()

    encoded_count = 0
    for pass_dir in sorted(processed_dir.iterdir()):
        if not pass_dir.is_dir():
            continue
        for group_dir in sorted(pass_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            for wav_path in sorted(group_dir.glob("hit_*.wav")):
                cache_path = (
                    codegram_dir / pass_dir.name / group_dir.name
                    / wav_path.with_suffix(".npy").name
                )
                audio = load_wav(wav_path)
                encode_and_cache(model, audio, cache_path)
                encoded_count += 1

    logger.info("Encoded %d files to codegrams", encoded_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import and process training data for VariationSampler.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan and analyze a capture pass.")
    p_scan.add_argument("pass_dir", type=Path, help="Path to capture pass directory.")
    p_scan.add_argument(
        "--check-duplicates", action="store_true",
        help="Run duplicate detection (slow — loads all audio).",
    )
    p_scan.add_argument(
        "--dup-threshold", type=float, default=0.999,
        help="Correlation threshold for duplicate detection (default: 0.999).",
    )

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run preprocessing pipeline.")
    p_pre.add_argument("pass_dir", type=Path, help="Path to capture pass directory.")
    p_pre.add_argument(
        "--pad-length", type=float, default=1.0,
        help="Target audio length in seconds (default: 1.0).",
    )
    p_pre.add_argument(
        "--target-db", type=float, default=-18.0,
        help="Target RMS loudness in dBFS (default: -18.0).",
    )
    p_pre.add_argument(
        "--min-set-size", type=int, default=2,
        help="Skip sets with fewer than this many samples (default: 2).",
    )

    # splits
    p_split = subparsers.add_parser("splits", help="Generate train/dev/test splits.")
    p_split.add_argument(
        "--test-tracks", nargs="*", default=[],
        help="Track names to hold out for test (e.g., Track19 Track20).",
    )
    p_split.add_argument(
        "--dev-fraction", type=float, default=0.1,
        help="Fraction of non-test groups for dev (default: 0.1).",
    )
    p_split.add_argument("--seed", type=int, default=42, help="Random seed.")

    # encode
    subparsers.add_parser("encode", help="Encode processed audio through DAC.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "scan": cmd_scan,
        "preprocess": cmd_preprocess,
        "splits": cmd_splits,
        "encode": cmd_encode,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
