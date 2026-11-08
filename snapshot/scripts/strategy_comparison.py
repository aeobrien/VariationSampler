#!/usr/bin/env python3
"""Strategy comparison pipeline for attack-preservation mitigation.

Generates machine-gun WAVs for 4 strategies applied to problem instrument
families (CrossStick, SnareRim, HiHat) plus a Kick control sample.

Strategies:
  A — Baseline: standard masking + default composite scoring
  B — Buffer zone: ramp masking near attack + default composite scoring
  C — Attack-scored: standard masking + attack quality scoring
  D — Strict acceptance: standard masking + tight attack thresholds

Usage:
    python3 scripts/strategy_comparison.py \
        --checkpoint checkpoints/best.pt \
        --config configs/default.yaml \
        --output-dir outputs/strategy_comparison
"""

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
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
from src.model.masking import build_mask
from src.eval.metrics import (
    multi_resolution_stft_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
    high_frequency_energy_delta,
)
from src.eval.acceptance import evaluate_candidate, _composite_score, AcceptanceResult
from src.eval.machine_gun_proxy import render_machine_gun, compute_pairwise_spectral_distance
from src.postprocess.chain import postprocess
from src.utils.instrument_families import infer_family

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Target families and how many samples to pick per family
TARGET_FAMILIES = ["CrossStick", "SnareRim", "HiHat", "Kick"]


# ---------------------------------------------------------------------------
# Strategy B: buffer zone masking
# ---------------------------------------------------------------------------

def build_mask_with_buffer(
    batch_size: int,
    t_max: int,
    config: dict,
    ramp_width: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Like build_mask() but adds a probability ramp after attack_frames.

    The ramp zone linearly interpolates from p_attack to p_tail over
    `ramp_width` frames immediately after the attack region, reducing
    coupling-induced attack degradation.

    Ramp schedule (default ramp_width=3):
      frame attack_frames + 1: p_tail * 0.25 * mult
      frame attack_frames + 2: p_tail * 0.50 * mult
      frame attack_frames + 3: p_tail * 0.75 * mult
      frame attack_frames + 4+: p_tail * mult  (normal)

    Returns:
        Boolean mask [batch_size, n_edit, t_max].
    """
    masking_cfg = config["masking"]
    edit_codebooks = config["model"]["edit_codebooks"]
    n_edit = len(edit_codebooks)

    p_tail = masking_cfg["p_tail"]
    p_attack = masking_cfg["p_attack"]
    attack_frames = masking_cfg["attack_frames"]
    multipliers = masking_cfg["codebook_multipliers"]

    probs = torch.zeros(n_edit, t_max)
    for i, cb in enumerate(edit_codebooks):
        mult = multipliers[cb]
        # Attack region
        probs[i, :attack_frames] = p_attack * mult
        # Ramp zone
        for r in range(1, ramp_width + 1):
            frame = attack_frames + r - 1
            if frame < t_max:
                ramp_frac = r / (ramp_width + 1)
                probs[i, frame] = p_tail * ramp_frac * mult
        # Normal tail
        ramp_end = attack_frames + ramp_width
        if ramp_end < t_max:
            probs[i, ramp_end:] = p_tail * mult

    probs = probs.unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.bernoulli(probs, generator=generator).bool()
    return mask


@torch.no_grad()
def generate_k_candidates_with_buffer(
    model: "VariationTransformer",
    z_in: torch.Tensor,
    k: int,
    config: dict,
    ramp_width: int = 3,
) -> list[torch.Tensor]:
    """Generate k candidates using buffer-zone masking."""
    from src.model.sampling import sample_tokens, apply_mask

    device = z_in.device
    t_max = config["model"]["t_max"]
    edit_codebooks = config["model"]["edit_codebooks"]

    sampling_cfg = config["sampling"]
    temperature = sampling_cfg["temperature"]
    top_p = sampling_cfg["top_p"]

    results = []
    for _ in range(k):
        mask = build_mask_with_buffer(1, t_max, config, ramp_width=ramp_width)
        mask = mask.to(device)

        z_batch = z_in.unsqueeze(0)  # [1, nq, T]
        logits = model(z_batch, mask)
        tokens = sample_tokens(logits, temperature=temperature, top_p=top_p)
        z_out = apply_mask(z_batch, tokens, mask, edit_codebooks)
        results.append(z_out.squeeze(0))

    return results


# ---------------------------------------------------------------------------
# Strategy C: attack-quality scoring
# ---------------------------------------------------------------------------

def attack_quality_score(metrics: dict) -> float:
    """Composite attack quality score (lower = better attack preservation)."""
    return (
        (1.0 - metrics.get("transient_xcorr", 1.0))
        + (1.0 - metrics.get("attack_smear", 1.0))
        + abs(metrics.get("hf_energy_delta_db", 0.0)) / 6.0
    )


# ---------------------------------------------------------------------------
# Strategy D: strict acceptance thresholds
# ---------------------------------------------------------------------------

STRICT_THRESHOLDS = {
    "min_transient_xcorr": 0.95,
    "min_attack_smear": 0.95,
    "max_hf_energy_delta_db": 3.0,
}


def make_strict_config(config: dict) -> dict:
    """Return a config copy with tighter attack thresholds."""
    strict = copy.deepcopy(config)
    acc = strict.setdefault("acceptance", {})
    acc.update(STRICT_THRESHOLDS)
    return strict


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def find_one_sample_per_family(
    codegrams_dir: Path,
    families: list[str],
) -> list[dict]:
    """Find one codegram per target family.

    Returns list of dicts: {path, family, group_key, name}.
    """
    found: dict[str, dict] = {}

    for group_dir in sorted(codegrams_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        family = infer_family(group_dir.name)
        if family is None or family not in families:
            continue
        if family in found:
            continue

        npy_files = sorted(group_dir.glob("*.npy"))
        if not npy_files:
            continue

        found[family] = {
            "path": npy_files[0],
            "family": family,
            "group_key": group_dir.name,
            "name": f"{family}_{group_dir.name}",
        }

    # Return in the order requested
    return [found[f] for f in families if f in found]


# ---------------------------------------------------------------------------
# Candidate evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_all_candidates(
    source_audio: np.ndarray,
    source_codegram: np.ndarray,
    candidate_codegrams: list[np.ndarray],
    dac_model,
    config: dict,
) -> list[dict]:
    """Decode and evaluate all candidates, returning metrics per candidate."""
    results = []
    min_t = source_codegram.shape[1]

    for i, z_out in enumerate(candidate_codegrams):
        z_np = z_out.cpu().numpy() if hasattr(z_out, "cpu") else z_out
        z_np = z_np[:, :min_t]

        audio_out = decode_codegram(dac_model, z_np)[0]  # 1D
        min_samples = min(len(source_audio), len(audio_out))
        src = source_audio[:min_samples]
        out = audio_out[:min_samples]

        # Postprocess
        pp = postprocess(out, src, {"fade_ms": 10})

        # Metrics
        result = evaluate_candidate(
            src, pp, source_codegram, z_np, config,
        )

        results.append({
            "index": i,
            "codegram": z_np,
            "audio": pp,
            "acceptance": result,
            "metrics": result.metrics,
        })

    return results


def select_by_composite(candidates: list[dict], n: int) -> list[dict]:
    """Select top-n by default composite score (lower = better)."""
    scored = [
        (c, _composite_score(c["acceptance"])) for c in candidates
    ]
    scored.sort(key=lambda x: x[1])
    return [c for c, _ in scored[:n]]


def select_by_attack_score(candidates: list[dict], n: int) -> list[dict]:
    """Select top-n by attack quality score (lower = better)."""
    scored = [
        (c, attack_quality_score(c["metrics"])) for c in candidates
    ]
    scored.sort(key=lambda x: x[1])
    return [c for c, _ in scored[:n]]


def count_strict_passing(
    candidates: list[dict],
    strict_config: dict,
) -> int:
    """Count how many candidates pass strict thresholds."""
    acc = strict_config.get("acceptance", strict_config)
    count = 0
    for c in candidates:
        m = c["metrics"]
        if m.get("transient_xcorr", 0) < acc.get("min_transient_xcorr", 0.95):
            continue
        if m.get("attack_smear", 0) < acc.get("min_attack_smear", 0.95):
            continue
        if abs(m.get("hf_energy_delta_db", 99)) > acc.get("max_hf_energy_delta_db", 3.0):
            continue
        count += 1
    return count


def select_by_strict_acceptance(
    candidates: list[dict],
    n: int,
    strict_config: dict,
) -> list[dict]:
    """Select top-n that pass strict thresholds, sorted by composite."""
    acc = strict_config.get("acceptance", strict_config)

    passing = []
    for c in candidates:
        m = c["metrics"]
        if m.get("transient_xcorr", 0) < acc.get("min_transient_xcorr", 0.95):
            continue
        if m.get("attack_smear", 0) < acc.get("min_attack_smear", 0.95):
            continue
        if abs(m.get("hf_energy_delta_db", 99)) > acc.get("max_hf_energy_delta_db", 3.0):
            continue
        passing.append(c)

    # Sort passing by composite score
    passing.sort(key=lambda c: _composite_score(c["acceptance"]))

    if len(passing) < n:
        logger.warning(
            "Strict acceptance: only %d/%d candidates passed, "
            "filling with best remaining by composite score",
            len(passing), n,
        )
        used_indices = {c["index"] for c in passing}
        remaining = [c for c in candidates if c["index"] not in used_indices]
        remaining.sort(key=lambda c: _composite_score(c["acceptance"]))
        passing.extend(remaining[:n - len(passing)])

    return passing[:n]


# ---------------------------------------------------------------------------
# Machine-gun rendering
# ---------------------------------------------------------------------------

def render_and_save_machinegun(
    audio_hits: list[np.ndarray],
    output_path: Path,
    bpm: float = 120.0,
    n_hits: int = 8,
) -> None:
    """Render machine-gun WAV from a list of mono hit audios."""
    mg = render_machine_gun(audio_hits, bpm=bpm, n_hits=n_hits)
    # Convert to [1, samples] for save_wav
    save_wav(mg.reshape(1, -1).astype(np.float32), output_path)


def average_metrics(candidates: list[dict]) -> dict:
    """Compute average metrics across selected candidates."""
    if not candidates:
        return {}

    keys = list(candidates[0]["metrics"].keys())
    avg = {}
    for k in keys:
        vals = [c["metrics"][k] for c in candidates if k in c["metrics"]]
        if vals:
            avg[k] = round(float(np.mean(vals)), 4)
    return avg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_RAMP_WIDTHS: dict[str, int] = {
    "CrossStick": 5,
    "SnareRim": 5,
    "HiHat": 3,
    "Kick": 3,
}

LOW_ACCEPTANCE_THRESHOLD = 0.20


def compute_selection_stats(
    all_candidates: list[dict],
    selected: list[dict],
    k_total: int,
    strict_config: dict | None = None,
) -> dict:
    """Compute acceptance rate and diversity metrics for a selection."""
    # Acceptance rate: for strategies C/D, how many of k_total would qualify
    # For A/B, all candidates are eligible (acceptance_rate = 1.0 by definition)
    if strict_config is not None:
        n_passing = count_strict_passing(all_candidates, strict_config)
        acceptance_rate = n_passing / k_total if k_total > 0 else 0.0
    else:
        acceptance_rate = 1.0

    # Inter-variation spectral distance (diversity measure)
    hits = [c["audio"] for c in selected]
    if len(hits) >= 2:
        spectral_diversity = compute_pairwise_spectral_distance(hits)
    else:
        spectral_diversity = 0.0

    return {
        "acceptance_rate": round(acceptance_rate, 4),
        "n_passing": int(n_passing) if strict_config is not None else k_total,
        "n_selected": len(selected),
        "spectral_diversity": round(float(spectral_diversity), 4),
    }


def run_strategy_comparison(
    checkpoint_path: Path,
    config: dict,
    codegrams_dir: Path,
    output_dir: Path,
    k_candidates: int = 32,
    n_select: int = 8,
    family_ramp_widths: dict[str, int] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Run the full strategy comparison pipeline."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if family_ramp_widths is None:
        family_ramp_widths = DEFAULT_FAMILY_RAMP_WIDTHS

    # Load model
    logger.info("Loading model from %s...", checkpoint_path)
    model = VariationTransformer.from_config(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Load DAC
    logger.info("Loading DAC model...")
    dac_model = load_dac_model()

    # Find samples
    samples = find_one_sample_per_family(codegrams_dir, TARGET_FAMILIES)
    if not samples:
        logger.error("No samples found in %s", codegrams_dir)
        sys.exit(1)
    logger.info("Found %d samples: %s", len(samples), [s["family"] for s in samples])

    # Strict config for strategy D
    strict_config = make_strict_config(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    bpm = config.get("eval", {}).get("machine_gun_bpm", 120.0)

    manifest_samples = []

    for sample_info in samples:
        name = sample_info["name"]
        family = sample_info["family"]
        logger.info("=== Processing %s (%s) ===", name, family)

        codegram = load_codegram(sample_info["path"])
        t_max = config["model"]["t_max"]
        z_in = torch.from_numpy(codegram).long().to(device)

        # Pad/truncate
        actual_t = z_in.shape[1]
        if z_in.shape[1] < t_max:
            padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=device)
            padded[:, :z_in.shape[1]] = z_in
            z_in = padded
        elif z_in.shape[1] > t_max:
            z_in = z_in[:, :t_max]

        # Decode source
        source_audio = decode_codegram(dac_model, codegram)[0]  # 1D mono

        # Create sample output directory
        sample_dir = output_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save source single hit
        source_single_path = sample_dir / "source_single.wav"
        save_wav(source_audio.reshape(1, -1), source_single_path)

        # Render source machine-gun
        logger.info("Rendering source machine-gun...")
        source_mg_path = sample_dir / "source_machinegun.wav"
        render_and_save_machinegun(
            [source_audio], source_mg_path, bpm=bpm, n_hits=n_select,
        )

        # Generate candidates with standard masking (for A, C, D)
        logger.info("Generating %d candidates with standard masking...", k_candidates)
        t0 = time.time()
        standard_candidates_tensor = generate_k_candidates(
            model, z_in, k_candidates, config,
        )
        logger.info("Standard generation: %.1fs", time.time() - t0)

        # Generate candidates with buffer-zone masking (for B)
        # Use per-family ramp width (wider for sharp transients)
        ramp_width = family_ramp_widths.get(family, 3)
        logger.info(
            "Generating %d candidates with buffer-zone masking "
            "(ramp_width=%d for %s)...",
            k_candidates, ramp_width, family,
        )
        t0 = time.time()
        buffer_candidates_tensor = generate_k_candidates_with_buffer(
            model, z_in, k_candidates, config, ramp_width=ramp_width,
        )
        logger.info("Buffer-zone generation: %.1fs", time.time() - t0)

        # Evaluate all candidates
        logger.info("Evaluating standard candidates...")
        standard_candidates = evaluate_all_candidates(
            source_audio, codegram, standard_candidates_tensor, dac_model, config,
        )

        logger.info("Evaluating buffer-zone candidates...")
        buffer_candidates = evaluate_all_candidates(
            source_audio, codegram, buffer_candidates_tensor, dac_model, config,
        )

        # Apply each strategy's selection logic
        strategies = {}

        # Strategy A: baseline (standard masking + composite score)
        selected_a = select_by_composite(standard_candidates, n_select)
        strategies["A"] = selected_a

        # Strategy B: buffer zone masking + composite score
        selected_b = select_by_composite(buffer_candidates, n_select)
        strategies["B"] = selected_b

        # Strategy C: standard masking + attack quality score
        selected_c = select_by_attack_score(standard_candidates, n_select)
        strategies["C"] = selected_c

        # Strategy D: standard masking + strict acceptance thresholds
        selected_d = select_by_strict_acceptance(
            standard_candidates, n_select, strict_config,
        )
        strategies["D"] = selected_d

        # Render machine-gun WAVs and compute stats for each strategy
        strategy_paths = {}
        strategy_metrics = {}
        strategy_stats = {}

        for strat_id, selected in strategies.items():
            hits = [c["audio"] for c in selected]
            mg_path = sample_dir / f"strategy_{strat_id}_machinegun.wav"
            render_and_save_machinegun(hits, mg_path, bpm=bpm, n_hits=n_select)
            strategy_paths[strat_id] = str(mg_path.relative_to(output_dir))
            strategy_metrics[strat_id] = average_metrics(selected)

            # Compute acceptance rate and diversity
            # Strategy D uses strict thresholds; C uses attack scoring
            # (not threshold-based, so acceptance_rate=1.0 for C)
            strict_cfg = strict_config if strat_id == "D" else None
            pool = buffer_candidates if strat_id == "B" else standard_candidates
            stats = compute_selection_stats(
                pool, selected, k_candidates, strict_config=strict_cfg,
            )
            strategy_stats[strat_id] = stats

            logger.info(
                "Strategy %s: xcorr=%.3f smear=%.3f hf=%.1fdB mrstft=%.3f "
                "accept=%.0f%% diversity=%.3f",
                strat_id,
                strategy_metrics[strat_id].get("transient_xcorr", 0),
                strategy_metrics[strat_id].get("attack_smear", 0),
                strategy_metrics[strat_id].get("hf_energy_delta_db", 0),
                strategy_metrics[strat_id].get("mrstft", 0),
                stats["acceptance_rate"] * 100,
                stats["spectral_diversity"],
            )

            # Warn on low acceptance rate
            if stats["acceptance_rate"] < LOW_ACCEPTANCE_THRESHOLD:
                logger.warning(
                    "WARNING: Strategy %s acceptance rate for %s is %.0f%% "
                    "(%d/%d) — below %d%% viability threshold",
                    strat_id, family,
                    stats["acceptance_rate"] * 100,
                    stats["n_passing"], k_candidates,
                    int(LOW_ACCEPTANCE_THRESHOLD * 100),
                )

        manifest_samples.append({
            "name": name,
            "family": family,
            "ramp_width": ramp_width,
            "source_single": str(source_single_path.relative_to(output_dir)),
            "source_machinegun": str(source_mg_path.relative_to(output_dir)),
            "strategies": strategy_paths,
            "metrics": strategy_metrics,
            "stats": strategy_stats,
        })

    # Build manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "checkpoint": str(checkpoint_path),
            "k_candidates": k_candidates,
            "n_select": n_select,
            "family_ramp_widths": family_ramp_widths,
            "config_file": "configs/default.yaml",
        },
        "strategies": {
            "A": "Baseline (standard masking + default scoring)",
            "B": "Buffer zone (ramp masking + default scoring)",
            "C": "Attack-scored (standard masking + attack quality scoring)",
            "D": "Strict acceptance (standard masking + tight attack thresholds)",
        },
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)

    # Copy blind listening test HTML into output directory
    html_src = PROJECT_ROOT / "tools" / "blind_listening_test.html"
    if html_src.exists():
        html_dst = output_dir / "blind_test.html"
        shutil.copy2(html_src, html_dst)
        logger.info("Copied blind test HTML to %s", html_dst)
    else:
        logger.warning("Blind test HTML not found at %s", html_src)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy comparison pipeline for attack-preservation mitigation."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Config YAML file.",
    )
    parser.add_argument(
        "--codegrams-dir", type=str, default="data/codegrams/pass-02",
        help="Directory containing codegram subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/strategy_comparison",
        help="Output directory for WAVs and manifest.",
    )
    parser.add_argument(
        "--k-candidates", type=int, default=32,
        help="Number of candidates to generate per strategy.",
    )
    parser.add_argument(
        "--n-select", type=int, default=8,
        help="Number of variations to select per strategy.",
    )
    parser.add_argument(
        "--family-ramp-widths", type=str, default=None,
        help="Per-family ramp widths as key=value pairs, e.g. "
             "'CrossStick=5,SnareRim=5,HiHat=3,Kick=3'. "
             "Defaults: CrossStick=5, SnareRim=5, HiHat=3, Kick=3.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda, mps, cpu).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(Path(args.config))
    device = torch.device(args.device) if args.device else None

    # Parse per-family ramp widths
    family_ramp_widths = None
    if args.family_ramp_widths:
        family_ramp_widths = {}
        for pair in args.family_ramp_widths.split(","):
            k, v = pair.strip().split("=")
            family_ramp_widths[k.strip()] = int(v.strip())

    manifest = run_strategy_comparison(
        checkpoint_path=Path(args.checkpoint),
        config=config,
        codegrams_dir=Path(args.codegrams_dir),
        output_dir=Path(args.output_dir),
        k_candidates=args.k_candidates,
        n_select=args.n_select,
        family_ramp_widths=family_ramp_widths,
        device=device,
    )

    logger.info(
        "Strategy comparison complete. %d samples processed.",
        len(manifest["samples"]),
    )


if __name__ == "__main__":
    main()
