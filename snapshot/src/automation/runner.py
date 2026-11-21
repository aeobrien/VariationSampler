"""Automation runner for the hyperparameter tuning loop.

Orchestrates: generate -> evaluate -> report -> Claude API -> validate -> apply.
Does NOT retrain the model — tunes inference-time parameters on a fixed checkpoint.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.automation.report import generate_iteration_report, save_iteration_report
from src.automation.batch_summary import (
    generate_batch_summary,
    save_batch_summary,
    assemble_listening_pack,
)
from src.automation.claude_loop import (
    build_system_prompt,
    build_iteration_prompt,
    call_claude,
    parse_config_response,
    validate_config_update,
)
from src.eval.baselines import MetricDistribution, load_baseline
from src.utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)

# Lazy import to avoid requiring DAC/torch for unit tests
_EvaluationPipeline = None
_build_dev_sample_list = None


def _get_evaluation_imports():
    """Lazy-load evaluation pipeline to avoid heavy imports in tests."""
    global _EvaluationPipeline, _build_dev_sample_list
    if _EvaluationPipeline is None:
        from src.automation.evaluation import EvaluationPipeline, build_dev_sample_list
        _EvaluationPipeline = EvaluationPipeline
        _build_dev_sample_list = build_dev_sample_list
    return _EvaluationPipeline, _build_dev_sample_list


class AutomationRunner:
    """Runs the hyperparameter tuning automation loop.

    The loop generates variations with current config, evaluates metrics,
    sends results to Claude for config recommendations, validates and applies
    changes, then repeats until a stopping condition is met.
    """

    def __init__(self, batch_config_path: str | Path, device: str = "cpu") -> None:
        """Initialize the automation runner.

        Args:
            batch_config_path: Path to batch config JSON file.
            device: Torch device string.
        """
        self.batch_config_path = Path(batch_config_path)
        self.device = device

        with open(self.batch_config_path) as f:
            self.batch_config = json.load(f)

        self.batch_id = self.batch_config.get("batch_id", "unknown")
        self.max_iterations = self.batch_config.get("max_iterations", 10)
        self.stagnation_limit = self.batch_config.get("stagnation_limit", 3)
        self.dry_run = self.batch_config.get("dry_run", False)

        # Current hyperparameters (mutable during loop)
        self.current_hyperparams = dict(self.batch_config.get("hyperparameters", {}))

        # Guardrails
        self.rollback_thresholds = self.batch_config.get(
            "guardrails", {}
        ).get("rollback_thresholds", {})

        # Automation config (allowed params, step sizes)
        self.allowed_params = self.batch_config.get("automation", {}).get(
            "allowed_params",
            [
                "mask_p_attack", "mask_p_tail", "temperature", "top_p",
                "k_candidates", "learning_rate", "batch_size",
                "attack_frames", "editable_codebooks",
                "acceptance_band_low", "acceptance_band_high",
            ],
        )
        self.step_size_limits = self.batch_config.get("automation", {}).get(
            "step_size_limits",
            {
                "mask_p_tail": 0.05,
                "mask_p_attack": 0.02,
                "temperature": 0.2,
                "top_p": 0.1,
                "k_candidates": 4,
            },
        )

        # Output paths
        self.reports_dir = Path(self.batch_config.get("reports_dir", "reports"))
        self.outputs_dir = Path(self.batch_config.get("outputs_dir", "outputs"))
        self.claude_log_dir = self.reports_dir / "claude-log"

        # Per-family baselines (Phase 4 diagnostic)
        self._family_baselines: dict[str, dict[str, MetricDistribution]] = {}
        baselines_dir = Path(self.batch_config.get("baselines_dir", "data/baselines"))
        if baselines_dir.exists():
            self._load_family_baselines(baselines_dir)

        # State
        self.iteration_history: list[dict[str, Any]] = []
        self.stagnation_counter = 0
        self.best_metrics: dict[str, float] | None = None

        # Evaluation pipeline (lazy-loaded on first real evaluation)
        self._eval_pipeline = None
        self._dev_samples: list[dict[str, str]] | None = None

        logger.info(
            "AutomationRunner initialized: batch=%s, max_iter=%d, device=%s",
            self.batch_id, self.max_iterations, self.device,
        )

    def run(self) -> dict[str, Any]:
        """Execute the main automation loop.

        Returns:
            Batch summary dict.
        """
        stop_reason = "iteration_cap"

        for iteration in range(self.max_iterations):
            logger.info("=== Iteration %d / %d ===", iteration, self.max_iterations)

            try:
                # 1-3. Generate, decode, compute metrics
                eval_metrics, audio_paths = self._run_evaluation(
                    self.current_hyperparams, iteration,
                )

                # 4. Write iteration report
                previous_report = self.iteration_history[-1] if self.iteration_history else None

                # Build per-family baseline dists for this iteration's samples
                sample_families = {}
                if self._dev_samples:
                    for s in self._dev_samples:
                        sample_families[s["name"]] = s.get("family", "Unknown")

                report = generate_iteration_report(
                    iteration_id=iteration,
                    config=dict(self.current_hyperparams),
                    eval_metrics=eval_metrics,
                    audio_paths=audio_paths,
                    previous_report=previous_report,
                    family_baselines=self._family_baselines if self._family_baselines else None,
                    sample_families=sample_families if sample_families else None,
                )
                self.iteration_history.append(report)

                # 5-6. Send to Claude API, get config update
                if not self.dry_run:
                    new_config = self._get_claude_update(report)
                    if new_config is not None:
                        report["claude_config_update"] = new_config
                else:
                    new_config = None
                    logger.info("Dry run: skipping Claude API call")

                # 7. Regression check — only after a config change was applied
                # (iteration 0 is the baseline; no point rolling back to itself)
                config_changed = iteration > 0 and report.get("config") != self.iteration_history[-2].get("config")
                if config_changed:
                    regressed, regression_reasons = self._check_regression(
                        eval_metrics, self.rollback_thresholds,
                    )
                    if regressed:
                        logger.warning("Regression detected: %s", regression_reasons)
                        self.current_hyperparams = self._rollback_config(
                            self.iteration_history,
                        )
                        self.stagnation_counter += 1
                        report["rolled_back"] = True
                        report["regression_reasons"] = regression_reasons
                else:
                    regressed = False

                if not regressed and new_config:
                    # Apply validated config for next iteration
                    self.current_hyperparams.update(new_config)
                    logger.info("Applied config update: %s", new_config)

                # Save report AFTER Claude call and rollback status are set
                save_iteration_report(report, self.reports_dir)

                # 8. Check stopping conditions
                should_stop, reason = self._check_stopping(self.iteration_history)
                if should_stop:
                    stop_reason = reason
                    logger.info("Stopping: %s", reason)
                    break

                # Update stagnation tracking (skip if already incremented by rollback)
                if not regressed:
                    self._update_stagnation(eval_metrics)

            except Exception as e:
                logger.error("Iteration %d failed: %s", iteration, e)
                stop_reason = f"error: {e}"
                break

        # 9-10. Write batch summary, assemble listening pack
        summary = generate_batch_summary(
            batch_id=self.batch_id,
            iteration_reports=self.iteration_history,
            stop_reason=stop_reason,
        )
        save_batch_summary(summary, self.reports_dir)

        try:
            assemble_listening_pack(
                batch_id=self.batch_id,
                iteration_reports=self.iteration_history,
                output_dir=self.outputs_dir,
            )
        except Exception as e:
            logger.warning("Failed to assemble listening pack: %s", e)

        logger.info(
            "Batch '%s' complete: %d iterations, stopped: %s",
            self.batch_id, len(self.iteration_history), stop_reason,
        )
        return summary

    def _load_family_baselines(self, baselines_dir: Path) -> None:
        """Load per-family baseline distributions from JSON files.

        Expects files like: baselines_dir/Snare_mrstft.json, Kick_mfcc.json, etc.
        """
        for path in sorted(baselines_dir.glob("*.json")):
            # Parse filename: {Family}_{metric}.json
            stem = path.stem  # e.g. "Snare_mrstft"
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            family, metric = parts
            try:
                dist = load_baseline(path)
                self._family_baselines.setdefault(family, {})[metric] = dist
            except Exception as e:
                logger.warning("Failed to load baseline %s: %s", path, e)

        if self._family_baselines:
            logger.info(
                "Loaded per-family baselines: %s",
                {k: list(v.keys()) for k, v in self._family_baselines.items()},
            )

    def _init_evaluation_pipeline(self) -> None:
        """Lazy-initialize the evaluation pipeline and dev sample list."""
        if self._eval_pipeline is not None:
            return

        EvaluationPipeline, build_dev_sample_list = _get_evaluation_imports()

        checkpoint_path = self.batch_config.get(
            "model_checkpoint", "checkpoints/best.pt",
        )

        # Build full config from default + hyperparams overlay
        base_config_path = self.batch_config.get(
            "base_config", "configs/default.yaml",
        )
        base_config = load_config(base_config_path)
        config = self._apply_hyperparams_to_config(base_config, self.current_hyperparams)

        self._eval_pipeline = EvaluationPipeline(
            checkpoint_path=checkpoint_path,
            config=config,
            device=self.device,
        )

        # Build dev sample list
        splits_dir = self.batch_config.get("splits_dir", "data/splits")
        codegrams_dir = self.batch_config.get("codegrams_dir", "data/codegrams/pass-02")
        processed_dir = self.batch_config.get("processed_dir", "data/processed/pass-02")
        max_dev_samples = self.batch_config.get("eval", {}).get("dev_samples", 50)
        samples_per_family = self.batch_config.get("eval", {}).get("samples_per_family")

        self._dev_samples = build_dev_sample_list(
            splits_dir=splits_dir,
            codegrams_dir=codegrams_dir,
            processed_dir=processed_dir,
            max_samples=max_dev_samples if samples_per_family is None else None,
            samples_per_family=samples_per_family,
        )
        logger.info("Evaluation pipeline ready: %d dev samples", len(self._dev_samples))

    def _apply_hyperparams_to_config(
        self, base_config: dict, hyperparams: dict,
    ) -> dict:
        """Apply flat hyperparameters to the nested config structure."""
        import copy
        config = copy.deepcopy(base_config)

        # Map flat hyperparam keys to nested config paths
        mapping = {
            "mask_p_attack": ("masking", "p_attack"),
            "mask_p_tail": ("masking", "p_tail"),
            "temperature": ("sampling", "temperature"),
            "top_p": ("sampling", "top_p"),
            "k_candidates": ("sampling", "k_candidates"),
            "learning_rate": ("training", "learning_rate"),
            "batch_size": ("training", "batch_size"),
            "attack_frames": ("masking", "attack_frames"),
            "editable_codebooks": ("model", "edit_codebooks"),
            "acceptance_band_low": None,  # handled specially
            "acceptance_band_high": None,  # handled specially
        }

        for key, value in hyperparams.items():
            if key in mapping and mapping[key] is not None:
                section, param = mapping[key]
                if section not in config:
                    config[section] = {}
                config[section][param] = value
            elif key == "acceptance_band_low":
                config.setdefault("acceptance", {})
                band = config["acceptance"].get("mrstft_band", [0.1, 0.8])
                config["acceptance"]["mrstft_band"] = [value, band[1]]
            elif key == "acceptance_band_high":
                config.setdefault("acceptance", {})
                band = config["acceptance"].get("mrstft_band", [0.1, 0.8])
                config["acceptance"]["mrstft_band"] = [band[0], value]

        return config

    def _run_evaluation(
        self,
        hyperparams: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, list[float]], dict[str, dict]]:
        """Generate test samples and compute metrics.

        Connects to the real pipeline: model inference -> DAC decode -> metrics.
        Falls back to empty results if no checkpoint is available (for unit tests).

        Args:
            hyperparams: Current hyperparameters.
            iteration: Current iteration number.

        Returns:
            Tuple of (eval_metrics dict, audio_paths dict).
        """
        logger.info("Running evaluation with hyperparams: %s", hyperparams)

        # Check if checkpoint exists — if not, return empty (unit test mode)
        checkpoint_path = Path(
            self.batch_config.get("model_checkpoint", "checkpoints/best.pt")
        )
        if not checkpoint_path.exists():
            logger.warning(
                "Checkpoint not found at %s — returning empty evaluation",
                checkpoint_path,
            )
            return {}, {}

        # Initialize pipeline lazily (loads model + DAC on first call)
        self._init_evaluation_pipeline()

        # Update the pipeline's config with current hyperparams
        base_config_path = self.batch_config.get(
            "base_config", "configs/default.yaml",
        )
        base_config = load_config(base_config_path)
        self._eval_pipeline.config = self._apply_hyperparams_to_config(
            base_config, hyperparams,
        )

        # Run evaluation
        iteration_output_dir = self.outputs_dir / f"iteration-{iteration:03d}"
        k_candidates = hyperparams.get(
            "k_candidates",
            self._eval_pipeline.config.get("sampling", {}).get("k_candidates", 8),
        )

        eval_metrics, audio_paths = self._eval_pipeline.evaluate_samples(
            self._dev_samples,
            output_dir=iteration_output_dir,
            k_candidates=k_candidates,
        )

        return eval_metrics, audio_paths

    def _get_claude_update(self, report: dict[str, Any]) -> dict[str, Any] | None:
        """Get config update from Claude API.

        Args:
            report: Current iteration report.

        Returns:
            Config update dict, or None if invalid/rejected.
        """
        system_prompt = build_system_prompt(self.allowed_params, self.step_size_limits)
        user_prompt = build_iteration_prompt(
            report,
            previous_reports=self.iteration_history[:-1],
        )

        try:
            response = call_claude(
                system_prompt, user_prompt,
                log_dir=self.claude_log_dir,
            )
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return None

        try:
            new_config, reasoning = parse_config_response(response)
        except ValueError as e:
            logger.warning("Failed to parse Claude response: %s", e)
            return None

        report["claude_reasoning"] = reasoning

        valid, errors = validate_config_update(
            new_config,
            {"masking": {"p_tail": self.current_hyperparams.get("mask_p_tail"),
                         "p_attack": self.current_hyperparams.get("mask_p_attack")},
             "sampling": {"temperature": self.current_hyperparams.get("temperature"),
                          "top_p": self.current_hyperparams.get("top_p"),
                          "k_candidates": self.current_hyperparams.get("k_candidates")}},
            self.allowed_params,
            self.step_size_limits,
        )

        if not valid:
            logger.warning("Claude config update rejected: %s", errors)
            return None

        logger.info("Claude config update accepted: %s", new_config)
        return new_config

    def _check_stopping(
        self,
        iteration_history: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Check if the loop should stop.

        Args:
            iteration_history: All iteration reports so far.

        Returns:
            Tuple of (should_stop, reason).
        """
        # Iteration cap
        if len(iteration_history) >= self.max_iterations:
            return True, "iteration_cap"

        # Stagnation
        if self.stagnation_counter >= self.stagnation_limit:
            return True, "stagnation"

        return False, ""

    def _check_regression(
        self,
        current_metrics: dict[str, list[float]],
        thresholds: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Check if current metrics have regressed past hard thresholds.

        Args:
            current_metrics: Current evaluation metrics.
            thresholds: Dict of metric_key -> threshold value.

        Returns:
            Tuple of (regressed, list_of_reasons).
        """
        reasons: list[str] = []

        for threshold_key, threshold_value in thresholds.items():
            # Parse threshold key format: "metric_stat" e.g. "mrstft_to_input_p95"
            # For simplicity, check mean of metric values
            metric_name = threshold_key.rsplit("_", 1)[0] if "_" in threshold_key else threshold_key
            stat = threshold_key.rsplit("_", 1)[-1] if "_" in threshold_key else "mean"

            values = current_metrics.get(metric_name, [])
            if not values:
                continue

            if stat == "p95":
                current_value = float(np.percentile(values, 95))
            elif stat == "min":
                current_value = float(np.min(values))
            elif stat == "mean":
                current_value = float(np.mean(values))
            else:
                current_value = float(np.mean(values))

            # For "min" thresholds (like acceptance_rate_min), below is bad
            if "min" in threshold_key:
                if current_value < threshold_value:
                    reasons.append(
                        f"{threshold_key}: {current_value:.4f} < {threshold_value}"
                    )
            else:
                # For upper thresholds, above is bad
                if current_value > threshold_value:
                    reasons.append(
                        f"{threshold_key}: {current_value:.4f} > {threshold_value}"
                    )

        return len(reasons) > 0, reasons

    def _rollback_config(
        self,
        iteration_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Revert to the last known good config.

        Args:
            iteration_history: All iteration reports.

        Returns:
            Reverted config dict.
        """
        # Find the most recent non-rolled-back config
        for report in reversed(iteration_history):
            if not report.get("rolled_back", False):
                logger.info(
                    "Rolling back to config from iteration %d",
                    report["iteration_id"],
                )
                return dict(report["config"])

        # If all are rolled back, use the starting config
        if iteration_history:
            return dict(iteration_history[0]["config"])
        return dict(self.current_hyperparams)

    def _update_stagnation(self, eval_metrics: dict[str, list[float]]) -> None:
        """Update stagnation counter based on metric improvement.

        Args:
            eval_metrics: Current evaluation metrics.
        """
        # Use mean MR-STFT as primary improvement indicator
        mrstft_values = eval_metrics.get("mrstft", [])
        if not mrstft_values:
            return

        current_mean = float(np.mean(mrstft_values))

        if self.best_metrics is None:
            self.best_metrics = {"mrstft_mean": current_mean}
            self.stagnation_counter = 0
            return

        best_mean = self.best_metrics.get("mrstft_mean", float("inf"))
        # "Improvement" means the metric moved closer to the target band
        # For simplicity, any decrease in distance counts as improvement
        if current_mean < best_mean * 0.98:  # 2% improvement threshold
            self.best_metrics["mrstft_mean"] = current_mean
            self.stagnation_counter = 0
            logger.info("Improvement detected: mrstft %.4f -> %.4f", best_mean, current_mean)
        else:
            self.stagnation_counter += 1
            logger.info(
                "No improvement (stagnation %d/%d): mrstft %.4f vs best %.4f",
                self.stagnation_counter, self.stagnation_limit,
                current_mean, best_mean,
            )
