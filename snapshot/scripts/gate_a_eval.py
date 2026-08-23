#!/usr/bin/env python3
"""Gate A evaluation: generate variations, render machine-gun comparisons, save everything.

Runs all generation tests in one pass and produces:
  - Source + variation WAVs for each test
  - Machine-gun renders: source repeated vs variations at 120 BPM 16th notes
  - A single generation_summary.json with all metrics

Usage:
    python3 scripts/gate_a_eval.py --checkpoint checkpoints/best.pt
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
from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.model.inference import generate_k_candidates
from src.eval.metrics import (
    multi_resolution_stft_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
    high_frequency_energy_delta,
)
from src.eval.machine_gun_proxy import render_machine_gun, machine_gun_score
from src.postprocess.chain import postprocess

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Test definitions ----------

TESTS = [
    {
        "name": "snare_v127_t09",
        "source": "data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "Snare hard hit, default temperature",
    },
    {
        "name": "snare_v127_t07",
        "source": "data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.7,
        "top_p": 0.95,
        "description": "Snare hard hit, conservative temperature",
    },
    {
        "name": "snare_v127_t05",
        "source": "data/codegrams/pass-02/Track1_Snare_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.5,
        "top_p": 0.95,
        "description": "Snare hard hit, low temperature",
    },
    {
        "name": "snare_v076",
        "source": "data/codegrams/pass-02/Track1_Snare_v076/hit_01.npy",
        "n": 8,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "Snare medium velocity",
    },
    {
        "name": "kick_v127",
        "source": "data/codegrams/pass-02/Track1_Kick_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "Kick hard hit",
    },
    {
        "name": "hihat_v127",
        "source": "data/codegrams/pass-02/Track1_HiHatClosed_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "Closed hi-hat hard hit",
    },
    {
        "name": "rimshot_v127",
        "source": "data/codegrams/pass-02/Track1_Rimshot_v127/hit_01.npy",
        "n": 8,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "Rimshot hard hit",
    },
    {
        "name": "snare_rr_set",
        "source_dir": "data/codegrams/pass-02/Track1_Snare_v127/",
        "n": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "description": "All hits in snare RR set, 4 variations each",
    },
]


def generate_and_decode(
    model: VariationTransformer,
    dac_model,
    source_codegram: np.ndarray,
    source_audio: np.ndarray,
    n: int,
    config: dict,
) -> list[dict]:
    """Generate N variations, decode, postprocess, compute metrics."""
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

    results = []
    src_mono = source_audio[0]

    for i, z_out in enumerate(candidates):
        z_out_np = z_out.cpu().numpy()[:, :actual_t]

        tcr = token_change_rate(source_codegram, z_out_np)

        audio_out = decode_codegram(dac_model, z_out_np)

        # Postprocess
        min_samples = min(source_audio.shape[1], audio_out.shape[1])
        audio_pp = postprocess(
            audio_out[0, :min_samples],
            source_audio[0, :min_samples],
            {"window_ms": 50, "fade_ms": 10},
        )
        audio_pp_2d = audio_pp.reshape(1, -1).astype(np.float32)

        out_mono = audio_pp

        mrstft = multi_resolution_stft_distance(src_mono[:min_samples], out_mono)
        smear = attack_smear_score(src_mono[:min_samples], out_mono)
        xcorr = transient_cross_correlation(src_mono[:min_samples], out_mono)
        hf_delta = high_frequency_energy_delta(src_mono[:min_samples], out_mono)

        results.append({
            "index": i,
            "audio": audio_pp_2d,
            "mono": out_mono,
            "codegram": z_out_np,
            "metrics": {
                "token_change_rate": round(float(tcr), 4),
                "mrstft": round(float(mrstft), 4),
                "attack_smear": round(float(smear), 4),
                "transient_xcorr": round(float(xcorr), 4),
                "hf_energy_delta_db": round(float(hf_delta), 2),
            },
        })

    return results


def render_machine_gun_comparison(
    source_mono: np.ndarray,
    variation_monos: list[np.ndarray],
    output_dir: Path,
    name: str,
    bpm: float = 120.0,
    n_hits: int = 8,
) -> dict:
    """Render machine-gun test: source repeated vs variations cycling."""
    # Source repeated (the "machine gun" reference)
    source_mg = render_machine_gun(
        [source_mono], bpm=bpm, n_hits=n_hits, sr=SAMPLE_RATE,
    )
    save_wav(
        source_mg.reshape(1, -1).astype(np.float32),
        output_dir / f"{name}_machinegun_source_repeated.wav",
    )

    # Variations cycling
    if variation_monos:
        var_mg = render_machine_gun(
            variation_monos, bpm=bpm, n_hits=n_hits, sr=SAMPLE_RATE,
        )
        save_wav(
            var_mg.reshape(1, -1).astype(np.float32),
            output_dir / f"{name}_machinegun_variations.wav",
        )

        # Machine gun score for source repeated
        source_score = machine_gun_score([source_mono] * n_hits)

        # Machine gun score for variations
        var_hits = [variation_monos[i % len(variation_monos)] for i in range(n_hits)]
        var_score = machine_gun_score(var_hits)

        return {
            "source_repeated_spectral_dist": round(source_score["spectral_distance"], 4),
            "variations_spectral_dist": round(var_score["spectral_distance"], 4),
            "source_repeated_similarity": round(source_score["self_similarity"], 4),
            "variations_similarity": round(var_score["self_similarity"], 4),
        }

    return {}


def run_test(
    test: dict,
    model: VariationTransformer,
    dac_model,
    config: dict,
    output_dir: Path,
) -> dict:
    """Run a single test definition."""
    name = test["name"]
    test_dir = output_dir / name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Apply sampling overrides (use deepcopy, not JSON, to preserve int keys)
    import copy
    config_copy = copy.deepcopy(config)
    config_copy["sampling"]["temperature"] = test["temperature"]
    config_copy["sampling"]["top_p"] = test["top_p"]

    # Collect sources
    sources = []
    if "source_dir" in test:
        src_dir = PROJECT_ROOT / test["source_dir"]
        for npy in sorted(src_dir.glob("*.npy")):
            sources.append((npy.stem, load_codegram(npy)))
    else:
        src_path = PROJECT_ROOT / test["source"]
        sources.append((src_path.stem, load_codegram(src_path)))

    test_result = {
        "name": name,
        "description": test["description"],
        "temperature": test["temperature"],
        "top_p": test["top_p"],
        "n_per_source": test["n"],
        "sources": [],
    }

    for src_name, codegram in sources:
        logger.info("--- %s / %s ---", name, src_name)

        # Decode source
        source_audio = decode_codegram(dac_model, codegram)
        save_wav(source_audio, test_dir / f"{src_name}_source.wav")

        # Generate variations
        t0 = time.time()
        results = generate_and_decode(
            model, dac_model, codegram, source_audio, test["n"], config_copy,
        )
        gen_time = time.time() - t0

        # Save variation WAVs
        for r in results:
            save_wav(r["audio"], test_dir / f"{src_name}_var_{r['index'] + 1:02d}.wav")

        # Log metrics
        for r in results:
            m = r["metrics"]
            logger.info(
                "  var_%02d: tcr=%.3f mrstft=%.3f smear=%.3f xcorr=%.3f hf=%.1fdB",
                r["index"] + 1, m["token_change_rate"], m["mrstft"],
                m["attack_smear"], m["transient_xcorr"], m["hf_energy_delta_db"],
            )

        # Machine-gun comparison
        src_mono = source_audio[0]
        var_monos = [r["mono"] for r in results]
        mg_scores = render_machine_gun_comparison(
            src_mono, var_monos, test_dir, src_name,
        )

        if mg_scores:
            logger.info(
                "  Machine-gun spectral dist: source_repeated=%.4f, variations=%.4f",
                mg_scores["source_repeated_spectral_dist"],
                mg_scores["variations_spectral_dist"],
            )

        source_result = {
            "source_name": src_name,
            "gen_time_s": round(gen_time, 2),
            "machine_gun": mg_scores,
            "candidates": [r["metrics"] for r in results],
        }
        test_result["sources"].append(source_result)

    return test_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate A full evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")
    parser.add_argument("--output-dir", type=str, default="outputs/gate_a", help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(Path(args.config))

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

    # Load models
    model = VariationTransformer.from_config(config).to(device)
    load_checkpoint(Path(args.checkpoint), model)
    model.eval()

    logger.info("Loading DAC model...")
    dac_model = load_dac_model()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all tests
    all_results = []
    total_t0 = time.time()

    for test in TESTS:
        # Check source exists
        if "source_dir" in test:
            src_path = PROJECT_ROOT / test["source_dir"]
            if not src_path.exists():
                logger.warning("Skipping %s: %s not found", test["name"], src_path)
                continue
        else:
            src_path = PROJECT_ROOT / test["source"]
            if not src_path.exists():
                logger.warning("Skipping %s: %s not found", test["name"], src_path)
                continue

        result = run_test(test, model, dac_model, config, output_dir)
        all_results.append(result)

    total_time = time.time() - total_t0

    # Save summary
    summary = {
        "checkpoint": str(args.checkpoint),
        "total_time_s": round(total_time, 1),
        "tests": all_results,
    }
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    total_variations = sum(
        len(s["candidates"]) for t in all_results for s in t["sources"]
    )
    print(f"\n{'=' * 60}")
    print(f"  Gate A Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"  Tests run:        {len(all_results)}")
    print(f"  Total variations: {total_variations}")
    print(f"  Total time:       {total_time:.0f}s")
    print(f"  Output:           {output_dir}/")
    print(f"  Summary:          {summary_path}")
    print(f"{'=' * 60}")
    print(f"\n  Listen to the machine-gun files first:")
    print(f"    *_machinegun_source_repeated.wav  (same hit looped — the 'bad' reference)")
    print(f"    *_machinegun_variations.wav       (variations cycling — should sound natural)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
