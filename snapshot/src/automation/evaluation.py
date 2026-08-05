"""Evaluation pipeline for the automation loop.

Connects model inference, DAC decoding, metrics computation, and acceptance
filtering into a single callable that the AutomationRunner uses.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram
from src.eval.acceptance import evaluate_candidate
from src.eval.metrics import (
    multi_resolution_stft_distance,
    mfcc_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
    high_frequency_energy_delta,
    spectral_peak_divergence,
    inter_variation_distances,
)
from src.model.inference import generate_k_candidates
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.postprocess.chain import postprocess
from src.utils.audio import load_wav, save_wav, T_MAX
from src.utils.config import load_config
from src.utils.instrument_families import infer_family

logger = logging.getLogger(__name__)


def reconstruct_stereo(
    var_mono: np.ndarray,
    source_stereo: np.ndarray,
) -> np.ndarray:
    """Reconstruct stereo output from mono variation using source's stereo image.

    Uses mid-side decomposition: the source's "side" channel (L-R difference,
    i.e. the stereo image from mic placement) is preserved, while the "mid"
    channel (shared content) is replaced with the mono variation.

    Args:
        var_mono: 1D mono variation audio.
        source_stereo: 2D stereo source audio, shape [2, samples].

    Returns:
        Stereo audio [2, samples].
    """
    if source_stereo.shape[0] < 2:
        # Source is mono, just duplicate
        return np.stack([var_mono, var_mono], axis=0)

    min_len = min(len(var_mono), source_stereo.shape[1])
    var_mono = var_mono[:min_len]
    src_l = source_stereo[0, :min_len]
    src_r = source_stereo[1, :min_len]

    # Extract side channel from source: side = (L - R) / 2
    side = (src_l - src_r) / 2.0

    # Reconstruct: L = mid + side, R = mid - side
    out_l = var_mono + side
    out_r = var_mono - side

    return np.stack([out_l, out_r], axis=0).astype(np.float32)


class EvaluationPipeline:
    """End-to-end pipeline: load samples -> generate -> decode -> evaluate.

    Loads model and DAC once, then evaluates multiple dev samples.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict[str, Any],
        device: str = "cpu",
        dac_model: Any = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            checkpoint_path: Path to model checkpoint.
            config: Full config dict.
            device: Torch device string.
            dac_model: Optional pre-loaded DAC model (to avoid reloading).
        """
        self.config = config
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        # Load variation model
        logger.info("Loading variation model from %s", self.checkpoint_path)
        self.model = VariationTransformer.from_config(config).to(device)
        load_checkpoint(self.checkpoint_path, self.model)
        self.model.eval()

        # Load DAC model for decoding
        if dac_model is not None:
            self.dac_model = dac_model
        else:
            logger.info("Loading DAC model...")
            self.dac_model = load_dac_model()

    def evaluate_samples(
        self,
        sample_paths: list[dict[str, str]],
        output_dir: str | Path | None = None,
        k_candidates: int | None = None,
    ) -> tuple[dict[str, list[float]], dict[str, dict]]:
        """Evaluate a list of dev samples.

        Args:
            sample_paths: List of dicts with keys:
                - "codegram": path to .npy codegram file
                - "audio": path to source .wav file
                - "name": sample identifier
            output_dir: Optional directory to save generated WAV files.
            k_candidates: Number of candidates per sample (overrides config).

        Returns:
            Tuple of (eval_metrics, audio_paths_dict).
            eval_metrics: metric_name -> list of values (one per sample).
            audio_paths_dict: sample_name -> {"source": path, "variations": [paths]}.
        """
        if k_candidates is None:
            k_candidates = self.config.get("sampling", {}).get("k_candidates", 8)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics to collect
        metric_names = [
            "mrstft", "mrstft_attack", "mfcc", "token_change_rate",
            "attack_smear", "transient_xcorr", "hf_energy_delta_db",
            "spectral_peak_divergence", "accepted",
            "inter_var_mrstft_mean",
        ]
        eval_metrics: dict[str, list[float]] = {m: [] for m in metric_names}
        audio_paths_dict: dict[str, dict] = {}

        for sample_info in sample_paths:
            name = sample_info["name"]
            logger.info("Evaluating sample: %s", name)

            try:
                sample_metrics, sample_audio_paths = self._evaluate_one_sample(
                    sample_info, k_candidates, output_dir,
                )
                for metric_name in metric_names:
                    if metric_name in sample_metrics:
                        eval_metrics[metric_name].append(sample_metrics[metric_name])
                audio_paths_dict[name] = sample_audio_paths

            except Exception as e:
                logger.error("Failed to evaluate sample %s: %s", name, e)
                continue

        logger.info(
            "Evaluation complete: %d/%d samples evaluated",
            len(audio_paths_dict), len(sample_paths),
        )
        return eval_metrics, audio_paths_dict

    def _evaluate_one_sample(
        self,
        sample_info: dict[str, str],
        k_candidates: int,
        output_dir: Path | None,
    ) -> tuple[dict[str, float], dict]:
        """Evaluate one sample: generate candidates, decode, compute metrics.

        Returns:
            Tuple of (per-sample metrics dict, audio paths dict).
        """
        name = sample_info["name"]
        codegram_path = Path(sample_info["codegram"])
        audio_path = Path(sample_info["audio"])

        # Load source
        source_codegram = load_codegram(codegram_path)  # [NQ, T]
        source_audio = load_wav(audio_path)  # [channels, samples]
        source_mono = (
            np.mean(source_audio, axis=0)
            if source_audio.shape[0] > 1
            else source_audio[0]
        )

        # Prepare input tensor
        z_in = torch.from_numpy(source_codegram).long().to(self.device)
        actual_t = z_in.shape[1]

        # Pad to model's t_max if needed
        t_max = self.config["model"]["t_max"]
        if z_in.shape[1] < t_max:
            padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=self.device)
            padded[:, :z_in.shape[1]] = z_in
            z_in = padded

        # Generate K candidates
        candidates = generate_k_candidates(self.model, z_in, k_candidates, self.config)

        # Decode and evaluate each candidate
        variation_audios = []
        variation_codegrams = []
        accepted_count = 0
        candidate_metrics_list = []
        variation_paths = []

        for i, z_out in enumerate(candidates):
            z_out_np = z_out.cpu().numpy()[:, :actual_t]
            variation_codegrams.append(z_out_np)

            # Decode through DAC
            audio_out = decode_codegram(self.dac_model, z_out_np)  # [channels, samples]

            # Convert to mono for metrics
            if audio_out.shape[0] > 1:
                out_mono = np.mean(audio_out, axis=0)
            else:
                out_mono = audio_out[0]

            # Match lengths
            min_len = min(len(source_mono), len(out_mono))
            src_mono = source_mono[:min_len]
            var_mono = out_mono[:min_len]

            # Postprocess
            var_mono = postprocess(
                var_mono, src_mono,
                {"fade_ms": 10},
            )

            variation_audios.append(var_mono)

            # Acceptance filter
            result = evaluate_candidate(
                src_mono, var_mono,
                source_codegram[:, :actual_t], z_out_np,
                self.config,
            )
            if result.accepted:
                accepted_count += 1
            candidate_metrics_list.append(result.metrics)

            # Save WAV if output_dir provided
            if output_dir is not None:
                sample_dir = output_dir / name
                sample_dir.mkdir(parents=True, exist_ok=True)
                var_path = sample_dir / f"var_{i + 1:02d}.wav"
                # Reconstruct stereo from source's stereo image
                var_stereo = reconstruct_stereo(var_mono, source_audio)
                save_wav(var_stereo, var_path)
                variation_paths.append(str(var_path))

        # Save source if output_dir provided
        source_out_path = ""
        if output_dir is not None:
            sample_dir = output_dir / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_out_path = str(sample_dir / "source.wav")
            save_wav(source_audio, Path(source_out_path))

        # Aggregate metrics across candidates: use the best accepted, or best overall
        if candidate_metrics_list:
            # Average across all candidates (accepted or not)
            avg_metrics = {}
            for key in candidate_metrics_list[0]:
                vals = [m[key] for m in candidate_metrics_list if key in m]
                avg_metrics[key] = float(np.mean(vals))
        else:
            avg_metrics = {}

        # Token change rate (average across candidates)
        tcr_values = []
        for z_out_np in variation_codegrams:
            tcr = token_change_rate(
                source_codegram[:, :actual_t], z_out_np,
                codebooks=self.config["model"]["edit_codebooks"],
            )
            tcr_values.append(tcr)
        avg_tcr = float(np.mean(tcr_values)) if tcr_values else 0.0

        # MR-STFT on attack window
        attack_samples = int(0.08 * 44100)
        mrstft_attack_values = []
        for var_mono in variation_audios:
            min_len = min(len(source_mono), len(var_mono))
            d = multi_resolution_stft_distance(
                source_mono[:min_len], var_mono[:min_len],
                window_samples=attack_samples,
            )
            mrstft_attack_values.append(d)
        avg_mrstft_attack = float(np.mean(mrstft_attack_values)) if mrstft_attack_values else 0.0

        # Inter-variation pairwise distance
        if len(variation_audios) >= 2:
            inter_var = inter_variation_distances(variation_audios, metric_fn="mrstft")
            inter_var_mean = inter_var["mean"]
        else:
            inter_var_mean = 0.0

        # Build per-sample metrics
        sample_metrics = {
            "mrstft": avg_metrics.get("mrstft", 0.0),
            "mrstft_attack": avg_mrstft_attack,
            "mfcc": avg_metrics.get("mfcc", 0.0),
            "token_change_rate": avg_tcr,
            "attack_smear": avg_metrics.get("attack_smear", 0.0),
            "transient_xcorr": avg_metrics.get("transient_xcorr", 0.0),
            "hf_energy_delta_db": avg_metrics.get("hf_energy_delta_db", 0.0),
            "spectral_peak_divergence": avg_metrics.get("spectral_peak_divergence", 0.0),
            "accepted": accepted_count / max(len(candidates), 1),
            "inter_var_mrstft_mean": inter_var_mean,
        }

        audio_paths = {
            "source": source_out_path or str(audio_path),
            "variations": variation_paths,
        }

        logger.info(
            "Sample %s: mrstft=%.4f, tcr=%.4f, accepted=%d/%d",
            name, sample_metrics["mrstft"], avg_tcr,
            accepted_count, len(candidates),
        )
        return sample_metrics, audio_paths


def build_dev_sample_list(
    splits_dir: str | Path,
    codegrams_dir: str | Path,
    processed_dir: str | Path,
    max_samples: int | None = None,
    samples_per_family: int | None = None,
) -> list[dict[str, str]]:
    """Build a list of dev sample paths from the split manifest.

    Selects one representative sample per dev group (first hit).

    Args:
        splits_dir: Directory containing manifest.json.
        codegrams_dir: Base directory for codegram .npy files.
        processed_dir: Base directory for processed .wav files.
        max_samples: Limit total number of samples (for testing).
        samples_per_family: If set, select this many samples per instrument
            family instead of applying a global max_samples limit.

    Returns:
        List of dicts with "name", "codegram", "audio", "family" keys.
    """
    splits_dir = Path(splits_dir)
    codegrams_dir = Path(codegrams_dir)
    processed_dir = Path(processed_dir)

    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    dev_groups = manifest.get("dev", {})

    # Build all valid samples first
    all_samples: list[dict[str, str]] = []

    for group_key, file_paths in sorted(dev_groups.items()):
        if not file_paths:
            continue

        # Use first hit as representative
        wav_rel = file_paths[0]  # e.g. "data/processed/pass-02/Track3_PedalHat_v025/hit_01.wav"

        # Extract the group_dir/filename part (last 2 components)
        # e.g. "Track3_PedalHat_v025/hit_01.wav"
        wav_path = Path(wav_rel)
        rel_tail = Path(wav_path.parent.name) / wav_path.name

        codegram_rel = str(rel_tail).replace(".wav", ".npy")
        codegram_path = codegrams_dir / codegram_rel
        audio_path = processed_dir / rel_tail

        if not codegram_path.exists():
            logger.debug("Codegram not found: %s, skipping", codegram_path)
            continue
        if not audio_path.exists():
            logger.debug("Audio not found: %s, skipping", audio_path)
            continue

        # Clean name from group key
        name = group_key.replace("/", "_").replace(" ", "_")
        # infer_family expects just the group dir name (e.g. "Track10_Kick_v025"),
        # but manifest keys may include path prefixes (e.g. "pass-02/Track10_Kick_v025")
        group_name = Path(group_key).name
        family = infer_family(group_name) or "Unknown"

        all_samples.append({
            "name": name,
            "codegram": str(codegram_path),
            "audio": str(audio_path),
            "family": family,
        })

    # Apply selection strategy
    if samples_per_family is not None:
        # Group by family, take N per family
        from collections import defaultdict
        by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
        for s in all_samples:
            by_family[s["family"]].append(s)

        samples: list[dict[str, str]] = []
        for family_name in sorted(by_family):
            family_samples = by_family[family_name]
            samples.extend(family_samples[:samples_per_family])

        logger.info(
            "Per-family sampling: %d families, %d per family, %d total",
            len(by_family), samples_per_family, len(samples),
        )
    elif max_samples is not None:
        samples = all_samples[:max_samples]
    else:
        samples = all_samples

    logger.info("Built dev sample list: %d samples", len(samples))
    return samples
