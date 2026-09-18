#!/usr/bin/env python3
"""Gate B Evaluation — three-way comparison of ML vs procedural vs real RR.

Generates a comprehensive evaluation pack with metrics, machine-gun listening
tests, and a structured report for project owner sign-off.

Usage:
    python scripts/gate_b_eval.py \
        --checkpoint checkpoints/best.pt \
        --config configs/default.yaml \
        --output-dir outputs/gate_b \
        --samples-per-family 10 \
        --k-candidates 8 \
        --n-variations 6 \
        --seed 42
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# -- Project imports --
from src.data.codegram_cache import (
    decode_codegram,
    load_codegram,
    load_dac_model,
)
from src.eval.acceptance import evaluate_candidate, filter_candidates
from src.eval.baselines import MetricDistribution, is_in_band, load_baseline
from src.eval.machine_gun_proxy import (
    compute_pairwise_spectral_distance,
    render_machine_gun,
)
from src.eval.metrics import (
    attack_smear_score,
    high_frequency_energy_delta,
    inter_variation_distances,
    mfcc_distance,
    multi_resolution_stft_distance,
    spectral_peak_divergence,
    token_change_rate,
    transient_cross_correlation,
)
from src.model.inference import generate_k_candidates
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.postprocess.chain import postprocess
from src.utils.audio import SAMPLE_RATE, T_MAX, load_wav, save_wav
from src.utils.config import load_config
from src.utils.instrument_families import infer_family

# Procedural baseline functions (operates on [channels, samples] float64)
from scripts.procedural_baseline import generate_variation as proc_generate_variation

logger = logging.getLogger(__name__)

CANONICAL_FAMILIES = ["Snare", "Kick", "HiHat", "Rimshot", "CrossStick"]


# ---------------------------------------------------------------------------
# Phase A — Sample selection
# ---------------------------------------------------------------------------

def select_samples(
    splits_dir: Path,
    codegrams_dir: Path,
    processed_dir: Path,
    samples_per_family: int,
    seed: int,
) -> list[dict]:
    """Select dev samples with sibling paths for real RR computation.

    Returns list of dicts with keys:
        name, codegram, audio, family, group_key, sibling_audio_paths
    """
    manifest_path = splits_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    dev_groups = manifest.get("dev", {})

    all_samples: list[dict] = []
    for group_key, file_paths in sorted(dev_groups.items()):
        if not file_paths:
            continue

        # Parse paths — first hit is the representative
        wav_path = Path(file_paths[0])
        rel_tail = Path(wav_path.parent.name) / wav_path.name

        codegram_path = codegrams_dir / str(rel_tail).replace(".wav", ".npy")
        audio_path = processed_dir / rel_tail

        if not codegram_path.exists() or not audio_path.exists():
            continue

        group_name = Path(group_key).name
        family = infer_family(group_name) or "Unknown"

        # Build sibling audio paths (all hits in this group)
        sibling_paths = []
        for fp in file_paths:
            sp = processed_dir / Path(fp).parent.name / Path(fp).name
            if sp.exists():
                sibling_paths.append(str(sp))

        name = group_key.replace("/", "_").replace(" ", "_")
        all_samples.append({
            "name": name,
            "codegram": str(codegram_path),
            "audio": str(audio_path),
            "family": family,
            "group_key": group_key,
            "sibling_audio_paths": sibling_paths,
        })

    # Per-family selection
    by_family: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        by_family[s["family"]].append(s)

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for family_name in sorted(by_family):
        family_samples = by_family[family_name]
        # Shuffle for randomness, then take up to N
        rng.shuffle(family_samples)
        selected.extend(family_samples[:samples_per_family])

    logger.info(
        "Selected %d samples across %d families: %s",
        len(selected),
        len(by_family),
        {f: min(len(v), samples_per_family) for f, v in sorted(by_family.items())},
    )
    return selected


# ---------------------------------------------------------------------------
# Phase B — ML variation generation
# ---------------------------------------------------------------------------

def generate_ml_variations(
    samples: list[dict],
    model: VariationTransformer,
    dac_model: object,
    config: dict,
    k_candidates: int,
    n_variations: int,
    output_dir: Path,
    device: str,
) -> dict[str, dict]:
    """Generate ML variations for all samples.

    Returns: {sample_name: {
        "variations_mono": [1D arrays],
        "variation_codegrams": [nq,t arrays],
        "source_mono": 1D array,
        "source_codegram": [nq, actual_t],
        "metrics": [list of per-variation metric dicts],
        "acceptance_rate": float,
        "inter_var": dict,
    }}
    """
    results = {}
    t_max = config["model"]["t_max"]

    for i, sample in enumerate(samples):
        name = sample["name"]
        logger.info("ML generation [%d/%d]: %s", i + 1, len(samples), name)

        # Load source
        source_codegram = load_codegram(Path(sample["codegram"]))  # [NQ, T]
        source_audio = load_wav(Path(sample["audio"]))  # [channels, samples]
        source_mono = _to_mono(source_audio)

        # Prepare input tensor
        z_in = torch.from_numpy(source_codegram).long().to(device)
        actual_t = z_in.shape[1]

        # Pad to t_max
        if z_in.shape[1] < t_max:
            padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=device)
            padded[:, :z_in.shape[1]] = z_in
            z_in = padded

        # Generate K candidates
        candidates = generate_k_candidates(model, z_in, k_candidates, config)

        # Decode, postprocess, evaluate each candidate
        candidate_audios = []
        candidate_codegrams = []
        candidate_results = []

        for z_out in candidates:
            # Truncate to actual length before decoding
            z_out_np = z_out.cpu().numpy()[:, :actual_t]
            candidate_codegrams.append(z_out_np)

            audio_out = decode_codegram(dac_model, z_out_np)  # [channels, samples]
            out_mono = _to_mono(audio_out)

            # Match lengths
            min_len = min(len(source_mono), len(out_mono))
            src_m = source_mono[:min_len]
            var_m = out_mono[:min_len]

            # Postprocess
            var_m = postprocess(var_m, src_m, {"fade_ms": 10})
            candidate_audios.append(var_m)

            # Evaluate
            result = evaluate_candidate(
                src_m, var_m,
                source_codegram[:, :actual_t], z_out_np,
                config,
            )
            candidate_results.append(result)

        # Sort by composite score, keep best N accepted (or best N overall if not enough accepted)
        scored = []
        for idx, res in enumerate(candidate_results):
            scored.append((idx, res))

        accepted = [(idx, res) for idx, res in scored if res.accepted]
        if len(accepted) >= n_variations:
            # Sort accepted by composite score (lower better — use mrstft as proxy)
            accepted.sort(key=lambda x: x[1].metrics.get("mrstft", float("inf")))
            chosen = accepted[:n_variations]
        else:
            # Not enough accepted — take all accepted + fill with best rejected
            rejected = [(idx, res) for idx, res in scored if not res.accepted]
            rejected.sort(key=lambda x: x[1].metrics.get("mrstft", float("inf")))
            chosen = accepted + rejected[: n_variations - len(accepted)]

        chosen_audios = [candidate_audios[idx] for idx, _ in chosen]
        chosen_codegrams = [candidate_codegrams[idx] for idx, _ in chosen]
        chosen_metrics = [r.metrics for _, r in chosen]

        # Save variation WAVs
        sample_dir = output_dir / "by_family" / sample["family"] / _safe_dirname(name)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save source
        save_wav(source_audio, sample_dir / "source.wav")

        for j, var_mono in enumerate(chosen_audios):
            var_stereo = np.stack([var_mono, var_mono], axis=0).astype(np.float32)
            save_wav(var_stereo, sample_dir / f"ml_var_{j + 1:02d}.wav")

        # Inter-variation distances
        inter_var = {}
        if len(chosen_audios) >= 2:
            inter_var = inter_variation_distances(chosen_audios, metric_fn="mrstft")

        accepted_count = sum(1 for _, r in scored if r.accepted)

        results[name] = {
            "variations_mono": chosen_audios,
            "variation_codegrams": chosen_codegrams,
            "source_mono": source_mono,
            "source_audio": source_audio,
            "source_codegram": source_codegram[:, :actual_t],
            "metrics": chosen_metrics,
            "acceptance_rate": accepted_count / max(len(candidates), 1),
            "inter_var": inter_var,
            "family": sample["family"],
            "sample_dir": sample_dir,
        }

    return results


# ---------------------------------------------------------------------------
# Phase C — Procedural baseline generation
# ---------------------------------------------------------------------------

def generate_procedural_variations(
    samples: list[dict],
    ml_results: dict[str, dict],
    n_variations: int,
    output_dir: Path,
    seed: int,
) -> dict[str, dict]:
    """Generate procedural variations for all samples.

    Returns: {sample_name: {
        "variations_mono": [1D arrays],
        "metrics": [per-variation metric dicts],
        "inter_var": dict,
    }}
    """
    results = {}
    rng = np.random.default_rng(seed)

    for i, sample in enumerate(samples):
        name = sample["name"]
        logger.info("Procedural generation [%d/%d]: %s", i + 1, len(samples), name)

        ml_data = ml_results[name]
        source_mono = ml_data["source_mono"]

        # Load source as float64 [channels, samples] for procedural baseline
        source_audio_f64 = load_wav(Path(sample["audio"])).astype(np.float64)

        proc_audios_mono = []
        proc_metrics = []

        for j in range(n_variations):
            var_seed = int(rng.integers(0, 2**63))
            var_rng = np.random.default_rng(var_seed)

            # Generate procedural variation ([channels, samples] float64)
            var_f64 = proc_generate_variation(source_audio_f64, var_rng)

            # Convert to mono float32 for metrics
            var_mono = _to_mono(var_f64.astype(np.float32))

            # Match lengths
            min_len = min(len(source_mono), len(var_mono))
            src_m = source_mono[:min_len]
            var_m = var_mono[:min_len]

            proc_audios_mono.append(var_m)

            # Compute metrics
            metrics = _compute_metric_dict(src_m, var_m)
            proc_metrics.append(metrics)

            # Save WAV
            sample_dir = ml_data["sample_dir"]
            var_stereo = np.stack([var_m, var_m], axis=0).astype(np.float32)
            save_wav(var_stereo, sample_dir / f"proc_var_{j + 1:02d}.wav")

        # Inter-variation distances
        inter_var = {}
        if len(proc_audios_mono) >= 2:
            inter_var = inter_variation_distances(proc_audios_mono, metric_fn="mrstft")

        results[name] = {
            "variations_mono": proc_audios_mono,
            "metrics": proc_metrics,
            "inter_var": inter_var,
        }

    return results


# ---------------------------------------------------------------------------
# Phase D — Real RR distance computation
# ---------------------------------------------------------------------------

def compute_real_rr_distances(
    samples: list[dict],
    ml_results: dict[str, dict],
    baselines_dir: Path,
) -> dict[str, dict]:
    """Compute real round-robin pairwise distances for all samples.

    Returns: {sample_name: {
        "sibling_monos": [1D arrays],
        "pairwise_mrstft": MetricDistribution,
        "pairwise_mfcc": MetricDistribution,
    }}

    Also returns family-level baselines loaded from disk.
    """
    results = {}

    for i, sample in enumerate(samples):
        name = sample["name"]
        logger.info("Real RR distances [%d/%d]: %s", i + 1, len(samples), name)

        sibling_paths = sample["sibling_audio_paths"]
        if len(sibling_paths) < 2:
            logger.warning("Sample %s has < 2 siblings, skipping RR", name)
            results[name] = {
                "sibling_monos": [],
                "pairwise_mrstft": MetricDistribution("mrstft"),
                "pairwise_mfcc": MetricDistribution("mfcc"),
            }
            continue

        # Load all siblings as mono
        sibling_monos = []
        for sp in sibling_paths:
            audio = load_wav(Path(sp))
            mono = _to_mono(audio)
            sibling_monos.append(mono)

        # Save sibling hits
        sample_dir = ml_results[name]["sample_dir"]
        for j, mono in enumerate(sibling_monos):
            stereo = np.stack([mono, mono], axis=0).astype(np.float32)
            save_wav(stereo, sample_dir / f"rr_hit_{j + 1:02d}.wav")

        # Compute pairwise distances
        pairs = [
            (sibling_monos[a], sibling_monos[b])
            for a in range(len(sibling_monos))
            for b in range(a + 1, len(sibling_monos))
        ]

        mrstft_values = [multi_resolution_stft_distance(a, b) for a, b in pairs]
        mfcc_values = [mfcc_distance(a, b) for a, b in pairs]

        results[name] = {
            "sibling_monos": sibling_monos,
            "pairwise_mrstft": MetricDistribution("mrstft", mrstft_values),
            "pairwise_mfcc": MetricDistribution("mfcc", mfcc_values),
        }

    # Load family-level baselines from disk
    family_baselines: dict[str, dict[str, MetricDistribution]] = {}
    for family in CANONICAL_FAMILIES:
        family_baselines[family] = {}
        for metric in ["mrstft", "mfcc"]:
            bl_path = baselines_dir / f"{family}_{metric}.json"
            if bl_path.exists():
                family_baselines[family][metric] = load_baseline(bl_path)
            else:
                logger.warning("Baseline not found: %s", bl_path)

    return results, family_baselines


# ---------------------------------------------------------------------------
# Phase E — Machine-gun rendering & diversity scoring
# ---------------------------------------------------------------------------

def render_machine_guns(
    samples: list[dict],
    ml_results: dict[str, dict],
    proc_results: dict[str, dict],
    rr_results: dict[str, dict],
    output_dir: Path,
    bpm: float = 120.0,
) -> dict[str, dict]:
    """Render machine-gun WAVs and compute spectral distance scores.

    Returns: {sample_name: {
        "ml_spectral_dist": float,
        "proc_spectral_dist": float,
        "rr_spectral_dist": float,
    }}
    """
    results = {}
    ab_dir = output_dir / "machine_gun_ab"
    ab_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        name = sample["name"]
        family = sample["family"]
        logger.info("Machine-gun render [%d/%d]: %s", i + 1, len(samples), name)

        sample_dir = ml_results[name]["sample_dir"]
        ml_monos = ml_results[name]["variations_mono"]
        proc_monos = proc_results[name]["variations_mono"]
        rr_monos = rr_results[name]["sibling_monos"]

        scores = {}
        _safe_idx = _family_sample_index(family)

        # ML machine gun
        if ml_monos:
            mg_ml = render_machine_gun(ml_monos, bpm=bpm, sr=SAMPLE_RATE)
            mg_stereo = np.stack([mg_ml, mg_ml], axis=0).astype(np.float32)
            save_wav(mg_stereo, sample_dir / "ml_machine_gun.wav")
            scores["ml_spectral_dist"] = compute_pairwise_spectral_distance(
                ml_monos, sr=SAMPLE_RATE,
            )
            save_wav(mg_stereo, ab_dir / f"{family}_{_safe_idx}_ml.wav")
        else:
            scores["ml_spectral_dist"] = 0.0

        # Procedural machine gun
        if proc_monos:
            mg_proc = render_machine_gun(proc_monos, bpm=bpm, sr=SAMPLE_RATE)
            mg_stereo = np.stack([mg_proc, mg_proc], axis=0).astype(np.float32)
            save_wav(mg_stereo, sample_dir / "proc_machine_gun.wav")
            scores["proc_spectral_dist"] = compute_pairwise_spectral_distance(
                proc_monos, sr=SAMPLE_RATE,
            )
            save_wav(mg_stereo, ab_dir / f"{family}_{_safe_idx}_proc.wav")
        else:
            scores["proc_spectral_dist"] = 0.0

        # Real RR machine gun
        if len(rr_monos) >= 2:
            mg_rr = render_machine_gun(rr_monos, bpm=bpm, sr=SAMPLE_RATE)
            mg_stereo = np.stack([mg_rr, mg_rr], axis=0).astype(np.float32)
            save_wav(mg_stereo, sample_dir / "rr_machine_gun.wav")
            scores["rr_spectral_dist"] = compute_pairwise_spectral_distance(
                rr_monos, sr=SAMPLE_RATE,
            )
            save_wav(mg_stereo, ab_dir / f"{family}_{_safe_idx}_rr.wav")
        else:
            scores["rr_spectral_dist"] = 0.0

        results[name] = scores

    return results


# ---------------------------------------------------------------------------
# Phase F — Reporting
# ---------------------------------------------------------------------------

def build_report(
    samples: list[dict],
    ml_results: dict[str, dict],
    proc_results: dict[str, dict],
    rr_results: dict[str, dict],
    family_baselines: dict[str, dict[str, MetricDistribution]],
    mg_scores: dict[str, dict],
    config: dict,
    checkpoint_path: str,
    output_dir: Path,
    elapsed_s: float,
) -> dict:
    """Build JSON report and markdown evaluation document."""

    # Aggregate per-family and overall
    families_data: dict[str, dict] = defaultdict(lambda: {
        "ml_metrics": defaultdict(list),
        "proc_metrics": defaultdict(list),
        "rr_mrstft": [],
        "rr_mfcc": [],
        "ml_spectral_dist": [],
        "proc_spectral_dist": [],
        "rr_spectral_dist": [],
        "ml_acceptance_rate": [],
        "ml_inter_var_mrstft": [],
        "proc_inter_var_mrstft": [],
        "sample_count": 0,
    })

    metric_keys = [
        "mrstft", "mfcc", "attack_smear", "transient_xcorr",
        "hf_energy_delta_db", "spectral_peak_divergence",
    ]

    for sample in samples:
        name = sample["name"]
        family = sample["family"]
        fd = families_data[family]
        fd["sample_count"] += 1

        # ML metrics (average across chosen variations per sample)
        ml_data = ml_results[name]
        for mk in metric_keys:
            vals = [m.get(mk, 0.0) for m in ml_data["metrics"]]
            if vals:
                fd["ml_metrics"][mk].append(float(np.mean(vals)))

        fd["ml_metrics"]["token_change_rate"].append(
            float(np.mean([m.get("token_change_rate", 0.0) for m in ml_data["metrics"]]))
        )
        fd["ml_acceptance_rate"].append(ml_data["acceptance_rate"])

        if ml_data["inter_var"]:
            fd["ml_inter_var_mrstft"].append(ml_data["inter_var"].get("mean", 0.0))

        # Procedural metrics
        proc_data = proc_results[name]
        for mk in metric_keys:
            vals = [m.get(mk, 0.0) for m in proc_data["metrics"]]
            if vals:
                fd["proc_metrics"][mk].append(float(np.mean(vals)))

        if proc_data["inter_var"]:
            fd["proc_inter_var_mrstft"].append(proc_data["inter_var"].get("mean", 0.0))

        # Real RR distances
        rr_data = rr_results[name]
        fd["rr_mrstft"].extend(rr_data["pairwise_mrstft"].values)
        fd["rr_mfcc"].extend(rr_data["pairwise_mfcc"].values)

        # Machine-gun scores
        mg = mg_scores[name]
        fd["ml_spectral_dist"].append(mg["ml_spectral_dist"])
        fd["proc_spectral_dist"].append(mg["proc_spectral_dist"])
        fd["rr_spectral_dist"].append(mg["rr_spectral_dist"])

    # Build JSON report
    report = {
        "meta": {
            "date": datetime.now().isoformat(),
            "checkpoint": checkpoint_path,
            "config": str(config),
            "total_samples": len(samples),
            "elapsed_seconds": round(elapsed_s, 1),
        },
        "overall": _aggregate_overall(families_data, metric_keys, family_baselines),
        "per_family": {},
    }

    for family in sorted(families_data):
        fd = families_data[family]
        report["per_family"][family] = _aggregate_family(
            fd, metric_keys, family_baselines.get(family, {}),
        )

    # Write JSON
    json_path = output_dir / "gate_b_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    logger.info("Wrote JSON report: %s", json_path)

    # Write markdown
    md_path = Path("reports") / "gate-B-evaluation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = _render_markdown(report, checkpoint_path, config)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Wrote markdown report: %s", md_path)

    return report


def _aggregate_family(
    fd: dict,
    metric_keys: list[str],
    baselines: dict[str, MetricDistribution],
) -> dict:
    """Aggregate metrics for a single family."""

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0, "median": 0, "q25": 0, "q75": 0, "std": 0, "n": 0}
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
            "std": round(float(np.std(arr)), 4),
            "n": len(values),
        }

    result = {"sample_count": fd["sample_count"]}

    # ML metrics
    result["ml"] = {}
    for mk in metric_keys + ["token_change_rate"]:
        result["ml"][mk] = _stats(fd["ml_metrics"].get(mk, []))

    result["ml"]["acceptance_rate"] = _stats(fd["ml_acceptance_rate"])
    result["ml"]["inter_var_mrstft"] = _stats(fd["ml_inter_var_mrstft"])

    # Procedural metrics
    result["proc"] = {}
    for mk in metric_keys:
        result["proc"][mk] = _stats(fd["proc_metrics"].get(mk, []))

    result["proc"]["inter_var_mrstft"] = _stats(fd["proc_inter_var_mrstft"])

    # Real RR
    result["rr"] = {
        "mrstft": _stats(fd["rr_mrstft"]),
        "mfcc": _stats(fd["rr_mfcc"]),
    }

    # Machine-gun spectral distances
    result["machine_gun"] = {
        "ml": _stats(fd["ml_spectral_dist"]),
        "proc": _stats(fd["proc_spectral_dist"]),
        "rr": _stats(fd["rr_spectral_dist"]),
    }

    # In-band checks against family baselines
    result["in_band"] = {}
    ml_mrstft_median = result["ml"]["mrstft"]["median"]
    if "mrstft" in baselines:
        result["in_band"]["mrstft"] = is_in_band(ml_mrstft_median, baselines["mrstft"])
    ml_mfcc_median = result["ml"].get("mfcc", {}).get("median", 0)
    if "mfcc" in baselines:
        result["in_band"]["mfcc"] = is_in_band(ml_mfcc_median, baselines["mfcc"])

    return result


def _aggregate_overall(
    families_data: dict[str, dict],
    metric_keys: list[str],
    family_baselines: dict[str, dict[str, MetricDistribution]],
) -> dict:
    """Aggregate across all families into overall stats."""
    merged = {
        "ml_metrics": defaultdict(list),
        "proc_metrics": defaultdict(list),
        "rr_mrstft": [],
        "rr_mfcc": [],
        "ml_spectral_dist": [],
        "proc_spectral_dist": [],
        "rr_spectral_dist": [],
        "ml_acceptance_rate": [],
        "ml_inter_var_mrstft": [],
        "proc_inter_var_mrstft": [],
        "sample_count": 0,
    }

    for fd in families_data.values():
        merged["sample_count"] += fd["sample_count"]
        for mk in metric_keys + ["token_change_rate"]:
            merged["ml_metrics"][mk].extend(fd["ml_metrics"].get(mk, []))
            if mk != "token_change_rate":
                merged["proc_metrics"][mk].extend(fd["proc_metrics"].get(mk, []))
        merged["rr_mrstft"].extend(fd["rr_mrstft"])
        merged["rr_mfcc"].extend(fd["rr_mfcc"])
        merged["ml_spectral_dist"].extend(fd["ml_spectral_dist"])
        merged["proc_spectral_dist"].extend(fd["proc_spectral_dist"])
        merged["rr_spectral_dist"].extend(fd["rr_spectral_dist"])
        merged["ml_acceptance_rate"].extend(fd["ml_acceptance_rate"])
        merged["ml_inter_var_mrstft"].extend(fd["ml_inter_var_mrstft"])
        merged["proc_inter_var_mrstft"].extend(fd["proc_inter_var_mrstft"])

    return _aggregate_family(merged, metric_keys, {})


def _render_markdown(report: dict, checkpoint_path: str, config: dict) -> str:
    """Render the Gate B evaluation markdown document."""
    meta = report["meta"]
    overall = report["overall"]

    lines = [
        "# Gate B Evaluation Report",
        "",
        "## Summary",
        "",
        f"- **Date**: {meta['date']}",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Total samples**: {meta['total_samples']}",
        f"- **Elapsed time**: {meta['elapsed_seconds']}s",
        f"- **Mask p_tail**: {config.get('masking', {}).get('p_tail', 'N/A')}",
        f"- **Temperature**: {config.get('sampling', {}).get('temperature', 'N/A')}",
        "",
        "---",
        "",
        "## Overall Metric Comparison",
        "",
        _metric_comparison_table(overall),
        "",
        "## Machine-Gun Proxy Scores (spectral distance, higher = more variation)",
        "",
        _machine_gun_table(report),
        "",
        "## Acceptance Rate",
        "",
        _acceptance_table(report),
        "",
        "## Per-Family Breakdown",
        "",
    ]

    for family in sorted(report["per_family"]):
        fd = report["per_family"][family]
        lines.extend([
            f"### {family} (n={fd['sample_count']})",
            "",
            _metric_comparison_table(fd),
            "",
            f"Machine-gun spectral distance: "
            f"ML={fd['machine_gun']['ml']['median']:.4f}, "
            f"Proc={fd['machine_gun']['proc']['median']:.4f}, "
            f"RR={fd['machine_gun']['rr']['median']:.4f}",
            "",
            f"In-band: {fd.get('in_band', {})}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Listening Assessment",
        "",
        "*(To be filled in by project owner after listening to `machine_gun_ab/` directory)*",
        "",
        "### Machine-gun test (does the ML output break repetition?)",
        "",
        "| Family | ML breaks repetition? | ML vs Proc | ML vs Real RR | Notes |",
        "|--------|-----------------------|------------|---------------|-------|",
    ])
    for family in sorted(report["per_family"]):
        lines.append(f"| {family} | [ ] Yes / [ ] No | [ ] Better / [ ] Same / [ ] Worse | "
                      f"[ ] Close / [ ] Gap | |")
    lines.extend([
        "",
        "### Overall quality",
        "",
        "- [ ] Variations sound like the same instrument struck again",
        "- [ ] Variations sound different from the source (not identical copies)",
        "- [ ] No audible artifacts (clicks, tonal smearing, HF loss)",
        "- [ ] Attack transients are preserved",
        "- [ ] ML output consistently beats procedural baseline",
        "",
        "### Any samples that stood out (good or bad)?",
        "",
        "*(notes here)*",
        "",
        "---",
        "",
        "## Decision",
        "",
        "- [ ] **PASS** — ML variations match real RR magnitude and beat procedural baseline",
        "- [ ] **ADJUST** — Promising but needs parameter tuning (specify what)",
        "- [ ] **FAIL** — Fundamental issues prevent Gate B passage",
        "",
        "**Signed**: _________________________ **Date**: _____________",
        "",
    ])

    return "\n".join(lines)


def _metric_comparison_table(data: dict) -> str:
    """Build a markdown metric comparison table for ML vs Proc vs Real RR."""
    metrics = [
        ("MR-STFT", "mrstft"),
        ("MFCC", "mfcc"),
        ("Attack Smear", "attack_smear"),
        ("Transient Xcorr", "transient_xcorr"),
        ("HF Energy Delta (dB)", "hf_energy_delta_db"),
        ("Spectral Peak Div.", "spectral_peak_divergence"),
        ("Token Change Rate", "token_change_rate"),
        ("Inter-var MR-STFT", "inter_var_mrstft"),
    ]

    header = "| Metric | ML (median) | Proc (median) | Real RR (median) |"
    sep = "|--------|-------------|---------------|------------------|"
    rows = [header, sep]

    for label, key in metrics:
        ml_val = data.get("ml", {}).get(key, {}).get("median", "—")
        proc_val = data.get("proc", {}).get(key, {}).get("median", "—")

        if key in ("mrstft", "mfcc"):
            rr_val = data.get("rr", {}).get(key, {}).get("median", "—")
        else:
            rr_val = "—"

        ml_str = f"{ml_val:.4f}" if isinstance(ml_val, (int, float)) else str(ml_val)
        proc_str = f"{proc_val:.4f}" if isinstance(proc_val, (int, float)) else str(proc_val)
        rr_str = f"{rr_val:.4f}" if isinstance(rr_val, (int, float)) else str(rr_val)

        rows.append(f"| {label} | {ml_str} | {proc_str} | {rr_str} |")

    return "\n".join(rows)


def _machine_gun_table(report: dict) -> str:
    """Build machine-gun spectral distance table."""
    header = "| Family | ML | Procedural | Real RR |"
    sep = "|--------|----|------------|---------|"
    rows = [header, sep]

    for family in sorted(report["per_family"]):
        fd = report["per_family"][family]
        mg = fd["machine_gun"]
        rows.append(
            f"| {family} | {mg['ml']['median']:.4f} | "
            f"{mg['proc']['median']:.4f} | {mg['rr']['median']:.4f} |"
        )

    # Overall
    omg = report["overall"]["machine_gun"]
    rows.append(
        f"| **Overall** | **{omg['ml']['median']:.4f}** | "
        f"**{omg['proc']['median']:.4f}** | **{omg['rr']['median']:.4f}** |"
    )

    return "\n".join(rows)


def _acceptance_table(report: dict) -> str:
    """Build acceptance rate table."""
    header = "| Family | Acceptance Rate (median) | n |"
    sep = "|--------|-------------------------|---|"
    rows = [header, sep]

    for family in sorted(report["per_family"]):
        fd = report["per_family"][family]
        ar = fd["ml"]["acceptance_rate"]
        rows.append(f"| {family} | {ar['median']:.2%} | {ar['n']} |")

    oar = report["overall"]["ml"]["acceptance_rate"]
    rows.append(f"| **Overall** | **{oar['median']:.2%}** | **{oar['n']}** |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert [channels, samples] to 1D mono float32."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.shape[0] > 1:
        return np.mean(audio, axis=0).astype(np.float32)
    return audio[0].astype(np.float32)


def _compute_metric_dict(source_mono: np.ndarray, var_mono: np.ndarray) -> dict:
    """Compute the full metric suite between source and variation."""
    return {
        "mrstft": float(multi_resolution_stft_distance(source_mono, var_mono)),
        "mfcc": float(mfcc_distance(source_mono, var_mono)),
        "attack_smear": float(attack_smear_score(source_mono, var_mono)),
        "transient_xcorr": float(transient_cross_correlation(source_mono, var_mono)),
        "hf_energy_delta_db": float(high_frequency_energy_delta(source_mono, var_mono)),
        "spectral_peak_divergence": float(spectral_peak_divergence(source_mono, var_mono)),
    }


def _safe_dirname(name: str) -> str:
    """Create a filesystem-safe directory name."""
    return name.replace("/", "_").replace(" ", "_")


_family_sample_counters: dict[str, int] = defaultdict(int)


def _family_sample_index(family: str) -> str:
    """Return a zero-padded sample index for flat AB directory naming."""
    _family_sample_counters[family] += 1
    return f"{_family_sample_counters[family]:02d}"


def _json_default(obj: object) -> object:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate B evaluation: ML vs procedural vs real RR.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="Config YAML path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_b"),
                        help="Output directory for evaluation pack.")
    parser.add_argument("--samples-per-family", type=int, default=10,
                        help="Max samples per instrument family.")
    parser.add_argument("--k-candidates", type=int, default=8,
                        help="Number of ML candidates to generate per sample.")
    parser.add_argument("--n-variations", type=int, default=6,
                        help="Number of variations to keep per sample.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detected if omitted).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # Paths
    splits_dir = Path("data/splits")
    codegrams_dir = Path("data/codegrams/pass-02")
    processed_dir = Path("data/processed/pass-02")
    baselines_dir = Path("data/baselines")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Phase A — Sample selection
    logger.info("=" * 60)
    logger.info("Phase A: Sample selection")
    logger.info("=" * 60)
    samples = select_samples(
        splits_dir, codegrams_dir, processed_dir,
        args.samples_per_family, args.seed,
    )
    if not samples:
        logger.error("No samples selected. Check data paths.")
        sys.exit(1)

    # Phase B — ML variation generation
    logger.info("=" * 60)
    logger.info("Phase B: ML variation generation")
    logger.info("=" * 60)
    logger.info("Loading model and DAC...")
    model = VariationTransformer.from_config(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    dac_model = load_dac_model()

    ml_results = generate_ml_variations(
        samples, model, dac_model, config,
        args.k_candidates, args.n_variations,
        args.output_dir, device,
    )

    # Phase C — Procedural baseline generation
    logger.info("=" * 60)
    logger.info("Phase C: Procedural baseline generation")
    logger.info("=" * 60)
    proc_results = generate_procedural_variations(
        samples, ml_results, args.n_variations,
        args.output_dir, args.seed + 1000,
    )

    # Phase D — Real RR distance computation
    logger.info("=" * 60)
    logger.info("Phase D: Real RR distance computation")
    logger.info("=" * 60)
    rr_results, family_baselines = compute_real_rr_distances(
        samples, ml_results, baselines_dir,
    )

    # Phase E — Machine-gun rendering
    logger.info("=" * 60)
    logger.info("Phase E: Machine-gun rendering & diversity scoring")
    logger.info("=" * 60)
    mg_scores = render_machine_guns(
        samples, ml_results, proc_results, rr_results, args.output_dir,
    )

    # Phase F — Reporting
    logger.info("=" * 60)
    logger.info("Phase F: Reporting")
    logger.info("=" * 60)
    elapsed = time.time() - start_time
    report = build_report(
        samples, ml_results, proc_results, rr_results,
        family_baselines, mg_scores, config,
        str(args.checkpoint), args.output_dir, elapsed,
    )

    logger.info("=" * 60)
    logger.info("Gate B evaluation complete in %.1fs", elapsed)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("JSON report: %s/gate_b_report.json", args.output_dir)
    logger.info("Markdown report: reports/gate-B-evaluation.md")
    logger.info("Listening pack: %s/machine_gun_ab/", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
