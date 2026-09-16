"""Ground-truth baseline metric distributions from real round-robin data."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from src.eval.metrics import multi_resolution_stft_distance, mfcc_distance, token_change_rate

logger = logging.getLogger(__name__)


@dataclass
class MetricDistribution:
    """Summary statistics for a metric computed over real round-robin pairs."""
    metric_name: str
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values)) if self.values else 0.0

    @property
    def q25(self) -> float:
        return float(np.percentile(self.values, 25)) if self.values else 0.0

    @property
    def q75(self) -> float:
        return float(np.percentile(self.values, 75)) if self.values else 0.0

    def summary(self) -> dict:
        """Return summary statistics as a dictionary."""
        return {
            "metric_name": self.metric_name,
            "n": len(self.values),
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "q25": self.q25,
            "q75": self.q75,
        }


def compute_pairwise_distances(
    audio_pairs: list[tuple[np.ndarray, np.ndarray]],
    metric_fn: str = "mrstft",
) -> MetricDistribution:
    """Compute pairwise distances for a set of audio pairs.

    Args:
        audio_pairs: List of (audio_a, audio_b) 1D mono arrays.
        metric_fn: Which metric to use: "mrstft" or "mfcc".

    Returns:
        MetricDistribution with all pairwise distances.
    """
    dist = MetricDistribution(metric_name=metric_fn)

    for audio_a, audio_b in audio_pairs:
        if metric_fn == "mrstft":
            d = multi_resolution_stft_distance(audio_a, audio_b)
        elif metric_fn == "mfcc":
            d = mfcc_distance(audio_a, audio_b)
        else:
            raise ValueError(f"Unknown metric: {metric_fn}")
        dist.values.append(d)

    logger.info(
        "Computed %d pairwise distances (%s): median=%.4f, IQR=[%.4f, %.4f]",
        len(dist.values), metric_fn, dist.median, dist.q25, dist.q75,
    )
    return dist


def compute_baseline_for_family(
    family_audio: dict[str, list[np.ndarray]],
    metric_fn: str = "mrstft",
) -> MetricDistribution:
    """Compute baseline distribution for an instrument family.

    For each round-robin set, computes all pairwise distances and
    aggregates into a single distribution.

    Args:
        family_audio: Mapping from set_id to list of mono audio arrays.
        metric_fn: Metric function name.

    Returns:
        Aggregated MetricDistribution across all sets.
    """
    all_values: list[float] = []

    for set_id, audios in family_audio.items():
        pairs = [(a, b) for i, a in enumerate(audios) for j, b in enumerate(audios) if i != j]
        if not pairs:
            continue

        dist = compute_pairwise_distances(pairs, metric_fn)
        all_values.extend(dist.values)
        logger.debug("Set '%s': %d pairs, median=%.4f", set_id, len(pairs), dist.median)

    result = MetricDistribution(metric_name=metric_fn, values=all_values)
    logger.info(
        "Family baseline (%s): %d total pairs, median=%.4f",
        metric_fn, len(all_values), result.median,
    )
    return result


def is_in_band(value: float, distribution: MetricDistribution, margin: float = 0.5) -> bool:
    """Check if a metric value falls within the distribution's IQR ± margin.

    Args:
        value: Metric value to check.
        distribution: Reference distribution.
        margin: Multiplier on IQR width for the acceptance band.

    Returns:
        True if value is within the acceptance band.
    """
    iqr = distribution.q75 - distribution.q25
    lower = distribution.q25 - margin * iqr
    upper = distribution.q75 + margin * iqr
    return lower <= value <= upper


def save_baseline(distribution: MetricDistribution, path: Path) -> None:
    """Save a baseline distribution to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": distribution.summary(),
        "values": distribution.values,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved baseline: %s", path)


def load_baseline(path: Path) -> MetricDistribution:
    """Load a baseline distribution from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    dist = MetricDistribution(
        metric_name=data["summary"]["metric_name"],
        values=data["values"],
    )
    logger.info("Loaded baseline: %s (%d values)", path, len(dist.values))
    return dist
