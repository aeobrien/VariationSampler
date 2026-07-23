#!/usr/bin/env python3
"""Generate variations from a trained VariationTransformer checkpoint.

This is the Gate A listening script. It loads a checkpoint, takes source
codegrams, generates N variations, decodes them back to WAV through DAC,
and optionally runs the acceptance filter and postprocessing chain.

Usage:
    # Generate 8 variations from a single source codegram
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy \
        --n 8

    # Generate from all hits in a round-robin set
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source-dir data/codegrams/pass-02/Track1_Snare_v127/ \
        --n 4

    # Generate from a WAV file (encodes through DAC first)
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source-wav data/processed/pass-02/Track1_Snare_v127/hit_01.wav \
        --n 8

    # Use a specific config (default: configs/default.yaml)
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy \
        --n 8 --config configs/default.yaml

    # Skip acceptance filter (listen to everything)
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy \
        --n 8 --no-filter

    # Override sampling temperature
    python scripts/generate.py \
        --checkpoint checkpoints/best.pt \
        --source data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy \
        --n 8 --temperature 0.7
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.audio import SAMPLE_RATE, save_wav
from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram, encode_audio
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.model.inference import generate_k_candidates
from src.eval.metrics import (
    multi_resolution_stft_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
)
from src.postprocess.chain import postprocess

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> VariationTransformer:
    """Load a trained model from checkpoint."""
    model = VariationTransformer.from_config(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


def generate_for_source(
    model: VariationTransformer,
    dac_model: "dac.DAC",
    source_codegram: np.ndarray,
    source_audio: np.ndarray | None,
    n: int,
    config: dict,
    output_dir: Path,
    source_name: str,
    no_filter: bool,
) -> dict:
    """Generate N variations for a single source, decode, and save.

    Returns dict with generation stats.
    """
    device = next(model.parameters()).device
    z_in = torch.from_numpy(source_codegram).long().to(device)

    # Pad/truncate to model's expected t_max
    t_max = config["model"]["t_max"]
    if z_in.shape[1] < t_max:
        padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=device)
        padded[:, :z_in.shape[1]] = z_in
        z_in = padded
    elif z_in.shape[1] > t_max:
        z_in = z_in[:, :t_max]

    actual_t = source_codegram.shape[1]

    logger.info("Generating %d candidates for %s (T=%d)...", n, source_name, actual_t)
    t0 = time.time()
    candidates = generate_k_candidates(model, z_in, n, config)
    gen_time = time.time() - t0
    logger.info("Generated %d candidates in %.1fs", n, gen_time)

    # Decode source through DAC (for reference WAV and metrics)
    if source_audio is None:
        logger.info("Decoding source through DAC...")
        source_audio = decode_codegram(dac_model, source_codegram)

    # Save source
    source_path = output_dir / f"{source_name}_source.wav"
    save_wav(source_audio, source_path)

    # Decode and save each candidate
    results = []
    for i, z_out in enumerate(candidates):
        z_out_np = z_out.cpu().numpy()
        # Trim back to original temporal length
        z_out_np = z_out_np[:, :actual_t]

        # Token change rate
        tcr = token_change_rate(source_codegram, z_out_np)

        # Decode through DAC
        audio_out = decode_codegram(dac_model, z_out_np)

        # Postprocess
        pp_config = {"window_ms": 50, "fade_ms": 10}
        # Ensure matching length for postprocessing
        min_samples = min(source_audio.shape[1], audio_out.shape[1])
        audio_pp = postprocess(
            audio_out[0, :min_samples],
            source_audio[0, :min_samples],
            pp_config,
        )
        # Back to [1, samples] shape
        audio_pp = audio_pp.reshape(1, -1)

        # Compute metrics (mono)
        src_mono = source_audio[0, :min_samples]
        out_mono = audio_pp[0]

        mrstft = multi_resolution_stft_distance(src_mono, out_mono)
        smear = attack_smear_score(src_mono, out_mono)
        xcorr = transient_cross_correlation(src_mono, out_mono)

        status = "kept"
        if not no_filter:
            # Simple Gate A filter: reject obvious failures
            if smear < 0.5:
                status = "REJECTED(smear)"
            elif xcorr < 0.5:
                status = "REJECTED(xcorr)"

        label = f"var_{i + 1:02d}"
        result = {
            "index": i,
            "label": label,
            "status": status,
            "token_change_rate": round(tcr, 4),
            "mrstft": round(mrstft, 4),
            "attack_smear": round(smear, 4),
            "transient_xcorr": round(xcorr, 4),
        }
        results.append(result)

        # Save WAV
        wav_name = f"{source_name}_{label}"
        if status.startswith("REJECTED"):
            wav_name += f"_{status}"
        wav_path = output_dir / f"{wav_name}.wav"
        save_wav(audio_pp.astype(np.float32), wav_path)

        logger.info(
            "  %s: tcr=%.3f mrstft=%.3f smear=%.3f xcorr=%.3f [%s]",
            label, tcr, mrstft, smear, xcorr, status,
        )

    return {
        "source": source_name,
        "n_generated": len(candidates),
        "n_kept": sum(1 for r in results if r["status"] == "kept"),
        "gen_time_s": round(gen_time, 2),
        "candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate variations for Gate A listening.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")
    parser.add_argument("--n", type=int, default=8, help="Number of variations to generate.")
    parser.add_argument("--output-dir", type=str, default="outputs/gate_a", help="Output directory.")
    parser.add_argument("--no-filter", action="store_true", help="Skip acceptance filtering.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")

    # Source input (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=str, help="Path to a single .npy codegram.")
    source_group.add_argument("--source-dir", type=str, help="Directory of .npy codegrams (process all).")
    source_group.add_argument("--source-wav", type=str, help="Path to a .wav file (encodes through DAC).")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(Path(args.config))

    # Apply overrides
    if args.temperature is not None:
        config["sampling"]["temperature"] = args.temperature
        logger.info("Temperature override: %.2f", args.temperature)
    if args.top_p is not None:
        config["sampling"]["top_p"] = args.top_p
        logger.info("Top-p override: %.2f", args.top_p)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load model
    model = load_model(Path(args.checkpoint), config, device)

    # Load DAC
    logger.info("Loading DAC model for decoding...")
    dac_model = load_dac_model()

    # Collect sources
    sources: list[tuple[str, np.ndarray, np.ndarray | None]] = []  # (name, codegram, audio_or_none)

    if args.source:
        p = Path(args.source)
        cg = load_codegram(p)
        sources.append((p.stem, cg, None))

    elif args.source_dir:
        d = Path(args.source_dir)
        for npy in sorted(d.glob("*.npy")):
            cg = load_codegram(npy)
            sources.append((npy.stem, cg, None))
        if not sources:
            logger.error("No .npy files found in %s", d)
            sys.exit(1)

    elif args.source_wav:
        from src.utils.audio import load_wav
        p = Path(args.source_wav)
        audio = load_wav(p)
        cg = encode_audio(dac_model, audio)
        sources.append((p.stem, cg, audio))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    all_results = []
    for name, codegram, audio in sources:
        result = generate_for_source(
            model, dac_model, codegram, audio,
            args.n, config, output_dir, name, args.no_filter,
        )
        all_results.append(result)

    # Save results summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    total_gen = sum(r["n_generated"] for r in all_results)
    total_kept = sum(r["n_kept"] for r in all_results)
    print(f"\n{'=' * 50}")
    print(f"  Generated {total_gen} variations from {len(sources)} source(s)")
    print(f"  Kept: {total_kept} / {total_gen}")
    print(f"  Output: {output_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
