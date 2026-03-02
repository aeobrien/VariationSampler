"""Iteration report generator for the automation loop."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.baselines import MetricDistribution, is_in_band

logger = logging.getLogger(__name__)


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _summarise_metric(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of metric values."""
    if not values:
        return {"mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "p5": _percentile(values, 5),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "std": float(np.std(values)),
        "n": len(values),
    }


def _compute_trend(
    current_mean: float,
    previous_mean: float | None,
    threshold: float = 0.05,
) -> str:
    """Determine trend direction between current and previous value.

    Args:
        current_mean: Current metric mean.
        previous_mean: Previous iteration's mean (None if first iteration).
        threshold: Relative change threshold for "stable".

    Returns:
        One of "improving", "stable", "regressing", or "first" (no previous).
    """
    if previous_mean is None:
        return "first"
    if previous_mean == 0.0:
        return "stable" if current_mean == 0.0 else "regressing"
    relative_change = (current_mean - previous_mean) / abs(previous_mean)
    if abs(relative_change) < threshold:
        return "stable"
    # For distance metrics, lower is generally "improving" (closer to source).
    # But this is metric-dependent. We report direction; consumer interprets.
    if relative_change < 0:
        return "improving"
    return "regressing"


def _composite_score(metrics: dict[str, float]) -> float:
    """Compute a composite quality score from per-sample metrics. Lower is better."""
    return (
        metrics.get("mrstft", 0.0)
        + metrics.get("mfcc", 0.0) / 10.0
        + abs(metrics.get("hf_energy_delta_db", 0.0)) / 6.0
    )


def generate_iteration_report(
    iteration_id: int,
    config: dict[str, Any],
    eval_metrics: dict[str, list[float]],
    audio_paths: dict[str, dict[str, Any]],
    baseline_dists: dict[str, MetricDistribution] | None = None,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured iteration report.

    Args:
        iteration_id: Iteration number.
        config: Full config snapshot for this iteration.
        eval_metrics: Dict of metric_name -> list of values (one per test sample).
        audio_paths: Dict of sample_name -> {"source": path, "variations": [paths]}.
        baseline_dists: Optional dict of metric_name -> MetricDistribution for
            ground-truth comparison.
        previous_report: Previous iteration's report dict for trend computation.

    Returns:
        Structured report dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Summarise each metric
    metric_summaries: dict[str, dict] = {}
    for metric_name, values in eval_metrics.items():
        metric_summaries[metric_name] = _summarise_metric(values)

    # Baseline comparison
    baseline_comparison: dict[str, dict] = {}
    if baseline_dists:
        for metric_name, dist in baseline_dists.items():
            if metric_name in metric_summaries:
                mean_val = metric_summaries[metric_name]["mean"]
                baseline_comparison[metric_name] = {
                    "value": mean_val,
                    "baseline_median": dist.median,
                    "baseline_q25": dist.q25,
                    "baseline_q75": dist.q75,
                    "in_band": is_in_band(mean_val, dist),
                }

    # Acceptance rate
    acceptance_values = eval_metrics.get("accepted", [])
    if acceptance_values:
        acceptance_rate = float(np.mean(acceptance_values))
    else:
        acceptance_rate = None

    # Trends vs previous iteration
    trends: dict[str, str] = {}
    if previous_report and "metrics" in previous_report:
        prev_metrics = previous_report["metrics"]
        for metric_name, summary in metric_summaries.items():
            prev_mean = None
            if metric_name in prev_metrics:
                prev_mean = prev_metrics[metric_name].get("mean")
            trends[metric_name] = _compute_trend(summary["mean"], prev_mean)
    else:
        for metric_name in metric_summaries:
            trends[metric_name] = "first"

    # Best/worst samples by composite score
    sample_scores: list[tuple[str, float]] = []
    for sample_name, paths_info in audio_paths.items():
        # Build per-sample metrics from the eval_metrics lists
        # This assumes eval_metrics values are ordered same as audio_paths
        sample_metrics = {}
        idx = list(audio_paths.keys()).index(sample_name)
        for metric_name, values in eval_metrics.items():
            if idx < len(values):
                sample_metrics[metric_name] = values[idx]
        score = _composite_score(sample_metrics)
        sample_scores.append((sample_name, score))

    sample_scores.sort(key=lambda x: x[1])
    best_samples = [{"name": name, "score": score} for name, score in sample_scores[:3]]
    worst_samples = [{"name": name, "score": score} for name, score in sample_scores[-3:]]

    report = {
        "iteration_id": iteration_id,
        "timestamp": timestamp,
        "config": config,
        "metrics": metric_summaries,
        "baseline_comparison": baseline_comparison,
        "acceptance_rate": acceptance_rate,
        "trends": trends,
        "best_samples": best_samples,
        "worst_samples": worst_samples,
        "audio_paths": {k: v for k, v in audio_paths.items()},
        "n_samples": len(audio_paths),
    }

    logger.info(
        "Generated iteration report #%d: %d metrics, %d samples, acceptance=%.2f",
        iteration_id,
        len(metric_summaries),
        len(audio_paths),
        acceptance_rate if acceptance_rate is not None else -1.0,
    )

    return report


def save_iteration_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Save an iteration report as JSON.

    Args:
        report: Report dict from generate_iteration_report().
        output_dir: Directory to write the report to.

    Returns:
        Path to the written report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_id = report["iteration_id"]
    path = output_dir / f"iteration-{iteration_id:03d}.json"

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Saved iteration report: %s", path)
    return path
