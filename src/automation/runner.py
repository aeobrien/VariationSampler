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
from src.utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)


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

        # State
        self.iteration_history: list[dict[str, Any]] = []
        self.stagnation_counter = 0
        self.best_metrics: dict[str, float] | None = None

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
                report = generate_iteration_report(
                    iteration_id=iteration,
                    config=dict(self.current_hyperparams),
                    eval_metrics=eval_metrics,
                    audio_paths=audio_paths,
                    previous_report=previous_report,
                )
                save_iteration_report(report, self.reports_dir)
                self.iteration_history.append(report)

                # 5-6. Send to Claude API, get config update
                if not self.dry_run:
                    new_config = self._get_claude_update(report)
                    if new_config is not None:
                        report["claude_config_update"] = new_config
                else:
                    new_config = None
                    logger.info("Dry run: skipping Claude API call")

                # 7. Check regression -> rollback if needed
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
                elif new_config:
                    # Apply validated config
                    self.current_hyperparams.update(new_config)
                    logger.info("Applied config update: %s", new_config)

                # 8. Check stopping conditions
                should_stop, reason = self._check_stopping(self.iteration_history)
                if should_stop:
                    stop_reason = reason
                    logger.info("Stopping: %s", reason)
                    break

                # Update stagnation tracking
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

    def _run_evaluation(
        self,
        hyperparams: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, list[float]], dict[str, dict]]:
        """Generate test samples and compute metrics.

        This is a placeholder that should be overridden or connected to
        the actual generation + evaluation pipeline.

        Args:
            hyperparams: Current hyperparameters.
            iteration: Current iteration number.

        Returns:
            Tuple of (eval_metrics dict, audio_paths dict).
        """
        logger.info("Running evaluation with hyperparams: %s", hyperparams)

        # Placeholder: subclass or monkey-patch for real evaluation
        # In production, this would:
        # 1. Load model checkpoint
        # 2. Generate K candidates per dev sample
        # 3. Decode through DAC
        # 4. Apply postprocessing
        # 5. Compute all metrics
        # 6. Return metrics + audio paths

        return {}, {}

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
