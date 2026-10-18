#!/usr/bin/env python3
"""Sweep mask rates and temperatures to find the variation sweet spot.

Tests multiple mask rate multipliers against a single source, measuring
spectral distance and token change rate. No retraining needed — mask rate
is an inference-time parameter.

Usage:
    python3 scripts/mask_rate_sweep.py --checkpoint checkpoints/best.pt
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.audio import SAMPLE_RATE, save_wav
from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.model.inference import generate_k_candidates
from src.eval.metrics import (
    multi_resolution_stft_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
)
from src.eval.machine_gun_proxy import compute_pairwise_spectral_distance, render_machine_gun
from src.postprocess.chain import postprocess

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sweep configurations: (label, mask_multiplier, temperature)
# mask_multiplier scales ALL codebook_multipliers by this factor
SWEEP = [
    ("baseline_t09", 1.0, 0.9),
    ("2x_mask_t09", 2.0, 0.9),
    ("4x_mask_t09", 4.0, 0.9),
    ("8x_mask_t09", 8.0, 0.9),
    ("4x_mask_t07", 4.0, 0.7),
    ("4x_mask_t12", 4.0, 1.2),
    ("8x_mask_t07", 8.0, 0.7),
    ("8x_mask_t12", 8.0, 1.2),
    # Aggressive: push mask rate to see where quality breaks
    ("12x_mask_t09", 12.0, 0.9),
    ("16x_mask_t09", 16.0, 0.9),
]


def make_config(base_config: dict, mask_mult: float, temperature: float) -> dict:
    """Create a config with scaled mask rates and temperature."""
    cfg = copy.deepcopy(base_config)
    for cb in cfg["masking"]["codebook_multipliers"]:
        cfg["masking"]["codebook_multipliers"][cb] = min(
            cfg["masking"]["codebook_multipliers"][cb] * mask_mult, 1.0,
        )
    cfg["masking"]["p_tail"] = min(cfg["masking"]["p_tail"] * mask_mult, 0.5)
    cfg["masking"]["p_attack"] = min(cfg["masking"]["p_attack"] * mask_mult, 0.25)
    cfg["sampling"]["temperature"] = temperature
    return cfg


def run_sweep_point(
    label: str,
    model: "VariationTransformer",
    dac_model,
    source_codegram: np.ndarray,
    source_audio: np.ndarray,
    config: dict,
    output_dir: Path,
    n: int = 8,
) -> dict:
    """Run one sweep configuration."""
    device = next(model.parameters()).device
    z_in = torch.from_numpy(source_codegram).long().to(device)

    t_max = config["model"]["t_max"]
    if z_in.shape[1] < t_max:
        padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=device)
        padded[:, :z_in.shape[1]] = z_in
        z_in = padded
    elif z_in.shape[1] > t_max:
        z_in = z_in[:, :t_max]

    actual_t = source_codegram.shape[1]

    candidates = generate_k_candidates(model, z_in, n, config)

    tcrs = []
    mrstfts = []
    smears = []
    xcorrs = []
    var_monos = []
    src_mono = source_audio[0]

    for z_out in candidates:
        z_out_np = z_out.cpu().numpy()[:, :actual_t]

        tcr = token_change_rate(source_codegram, z_out_np)
        tcrs.append(tcr)

        audio_out = decode_codegram(dac_model, z_out_np)
        min_samples = min(source_audio.shape[1], audio_out.shape[1])

        audio_pp = postprocess(
            audio_out[0, :min_samples],
            source_audio[0, :min_samples],
            {"window_ms": 50, "fade_ms": 10},
        )
        var_monos.append(audio_pp)

        mrstft = multi_resolution_stft_distance(src_mono[:min_samples], audio_pp)
        smear = attack_smear_score(src_mono[:min_samples], audio_pp)
        xcorr = transient_cross_correlation(src_mono[:min_samples], audio_pp)
        mrstfts.append(mrstft)
        smears.append(smear)
        xcorrs.append(xcorr)

    # Spectral distance between variations
    spectral_dist = compute_pairwise_spectral_distance(var_monos)

    # Save machine gun audio for listening
    point_dir = output_dir / label
    point_dir.mkdir(parents=True, exist_ok=True)

    mg_audio = render_machine_gun(var_monos, bpm=120.0, n_hits=8, sr=SAMPLE_RATE)
    save_wav(mg_audio.reshape(1, -1).astype(np.float32), point_dir / "machinegun_variations.wav")

    # Save individual variations
    for i, mono in enumerate(var_monos):
        save_wav(mono.reshape(1, -1).astype(np.float32), point_dir / f"var_{i+1:02d}.wav")

    result = {
        "label": label,
        "mean_tcr": round(float(np.mean(tcrs)), 4),
        "mean_mrstft": round(float(np.mean(mrstfts)), 4),
        "mean_smear": round(float(np.mean(smears)), 4),
        "mean_xcorr": round(float(np.mean(xcorrs)), 4),
        "spectral_distance": round(float(spectral_dist), 4),
    }

    logger.info(
        "  %s: tcr=%.3f mrstft=%.3f smear=%.3f xcorr=%.3f spec_dist=%.4f",
        label, result["mean_tcr"], result["mean_mrstft"],
        result["mean_smear"], result["mean_xcorr"], result["spectral_distance"],
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mask rates and temperatures.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to .npy codegram (default: first snare in pass-02)")
    parser.add_argument("--output-dir", type=str, default="outputs/mask_sweep")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n", type=int, default=8, help="Variations per sweep point")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    base_config = load_config(Path(args.config))

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = VariationTransformer.from_config(base_config).to(device)
    load_checkpoint(Path(args.checkpoint), model)
    model.eval()

    logger.info("Loading DAC model...")
    dac_model = load_dac_model()

    # Find source
    if args.source:
        src_path = Path(args.source)
    else:
        src_path = PROJECT_ROOT / "data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy"

    if not src_path.exists():
        logger.error("Source not found: %s", src_path)
        sys.exit(1)

    source_codegram = load_codegram(src_path)
    source_audio = decode_codegram(dac_model, source_codegram)

    # Save source machine gun for reference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_wav(source_audio, output_dir / "source.wav")
    src_mg = render_machine_gun([source_audio[0]], bpm=120.0, n_hits=8, sr=SAMPLE_RATE)
    save_wav(src_mg.reshape(1, -1).astype(np.float32), output_dir / "machinegun_source_repeated.wav")

    # Also compute spectral distance for real RR samples if available
    rr_dir = PROJECT_ROOT / "data/processed/pass-02/Track1_Snare_v127"
    if rr_dir.exists():
        import soundfile
        rr_wavs = sorted(rr_dir.glob("hit_*.wav"))[:8]
        rr_audios = []
        for p in rr_wavs:
            a, _ = soundfile.read(p)
            if a.ndim > 1:
                a = a[:, 0]
            rr_audios.append(a.astype(np.float32))
        rr_dist = compute_pairwise_spectral_distance(rr_audios)
        logger.info("Real RR spectral distance (target): %.4f", rr_dist)
    else:
        rr_dist = None

    # Run sweep
    results = []
    for label, mask_mult, temp in SWEEP:
        cfg = make_config(base_config, mask_mult, temp)
        result = run_sweep_point(
            label, model, dac_model, source_codegram, source_audio,
            cfg, output_dir, n=args.n,
        )
        result["mask_multiplier"] = mask_mult
        result["temperature"] = temp
        results.append(result)

    # Summary
    summary = {
        "checkpoint": str(args.checkpoint),
        "source": str(src_path),
        "real_rr_spectral_distance": rr_dist,
        "results": results,
    }
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results table
    print(f"\n{'=' * 80}")
    print(f"  Mask Rate Sweep Results")
    print(f"  Real RR spectral distance (target): {rr_dist:.4f}" if rr_dist else "  Real RR: N/A")
    print(f"{'=' * 80}")
    print(f"  {'Label':<20} {'Mask×':>6} {'Temp':>5} {'TCR':>6} {'MRSTFT':>7} {'Smear':>6} {'Xcorr':>6} {'SpecDist':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*9}")
    for r in results:
        print(
            f"  {r['label']:<20} {r['mask_multiplier']:>5.0f}× {r['temperature']:>5.1f}"
            f" {r['mean_tcr']:>6.3f} {r['mean_mrstft']:>7.3f} {r['mean_smear']:>6.3f}"
            f" {r['mean_xcorr']:>6.3f} {r['spectral_distance']:>9.4f}"
        )
    print(f"{'=' * 80}")
    print(f"  Target spectral distance: ~{rr_dist:.2f}" if rr_dist else "")
    print(f"  Listen: {output_dir}/*/machinegun_variations.wav")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
