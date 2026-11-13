"""Acceptance filter for generated drum sample variations."""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.eval.metrics import (
    multi_resolution_stft_distance,
    mfcc_distance,
    token_change_rate,
    attack_smear_score,
    transient_cross_correlation,
    high_frequency_energy_delta,
    spectral_peak_divergence,
)

logger = logging.getLogger(__name__)


@dataclass
class AcceptanceResult:
    """Result of evaluating a single candidate variation."""
    accepted: bool
    metrics: dict = field(default_factory=dict)
    reject_reasons: list[str] = field(default_factory=list)


def evaluate_candidate(
    audio_source: np.ndarray,
    audio_candidate: np.ndarray,
    codegram_source: np.ndarray,
    codegram_candidate: np.ndarray,
    config: dict,
) -> AcceptanceResult:
    """Evaluate a candidate variation against all acceptance criteria.

    Args:
        audio_source: 1D mono source audio, float32.
        audio_candidate: 1D mono candidate audio, float32.
        codegram_source: Source codegram [NQ, T].
        codegram_candidate: Candidate codegram [NQ, T].
        config: Acceptance config dict with threshold keys.

    Returns:
        AcceptanceResult with all metric values and rejection reasons.
    """
    acc = config.get("acceptance", config)
    attack_ms = acc.get("attack_ms", 30)

    metrics = {}
    reject_reasons = []

    # MR-STFT distance
    mrstft = multi_resolution_stft_distance(audio_source, audio_candidate)
    metrics["mrstft"] = mrstft
    mrstft_band = acc.get("mrstft_band", [0.1, 0.8])
    if mrstft < mrstft_band[0]:
        reject_reasons.append(f"mrstft {mrstft:.4f} below band {mrstft_band[0]}")
    elif mrstft > mrstft_band[1]:
        reject_reasons.append(f"mrstft {mrstft:.4f} above band {mrstft_band[1]}")

    # MFCC distance
    mfcc = mfcc_distance(audio_source, audio_candidate)
    metrics["mfcc"] = mfcc
    mfcc_band = acc.get("mfcc_band", [2.0, 15.0])
    if mfcc < mfcc_band[0]:
        reject_reasons.append(f"mfcc {mfcc:.4f} below band {mfcc_band[0]}")
    elif mfcc > mfcc_band[1]:
        reject_reasons.append(f"mfcc {mfcc:.4f} above band {mfcc_band[1]}")

    # Token change rate
    tcr = token_change_rate(codegram_source, codegram_candidate)
    metrics["token_change_rate"] = tcr
    min_tcr = acc.get("min_token_change_rate", 0.01)
    max_tcr = acc.get("max_token_change_rate", 0.30)
    if tcr < min_tcr:
        reject_reasons.append(f"token_change_rate {tcr:.4f} below {min_tcr}")
    elif tcr > max_tcr:
        reject_reasons.append(f"token_change_rate {tcr:.4f} above {max_tcr}")

    # Attack smear
    smear = attack_smear_score(audio_source, audio_candidate, attack_ms=attack_ms)
    metrics["attack_smear"] = smear
    min_smear = acc.get("min_attack_smear", 0.85)
    if smear < min_smear:
        reject_reasons.append(f"attack_smear {smear:.4f} below {min_smear}")

    # Transient cross-correlation
    xcorr = transient_cross_correlation(audio_source, audio_candidate, attack_ms=attack_ms)
    metrics["transient_xcorr"] = xcorr
    min_xcorr = acc.get("min_transient_xcorr", 0.95)
    if xcorr < min_xcorr:
        reject_reasons.append(f"transient_xcorr {xcorr:.4f} below {min_xcorr}")

    # HF energy delta
    hf_delta = high_frequency_energy_delta(audio_source, audio_candidate)
    metrics["hf_energy_delta_db"] = hf_delta
    max_hf = acc.get("max_hf_energy_delta_db", 6.0)
    if abs(hf_delta) > max_hf:
        reject_reasons.append(f"hf_energy_delta |{hf_delta:.2f}| dB exceeds {max_hf}")

    # Spectral peak divergence
    spd = spectral_peak_divergence(audio_source, audio_candidate)
    metrics["spectral_peak_divergence"] = spd
    max_spd = acc.get("max_spectral_peak_divergence", 3)
    if spd > max_spd:
        reject_reasons.append(f"spectral_peak_divergence {spd} exceeds {max_spd}")

    accepted = len(reject_reasons) == 0

    if accepted:
        logger.info("Candidate accepted: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    else:
        logger.info("Candidate rejected: %s", reject_reasons)

    return AcceptanceResult(accepted=accepted, metrics=metrics, reject_reasons=reject_reasons)


def _composite_score(result: AcceptanceResult) -> float:
    """Compute a composite quality score for sorting accepted candidates.

    Lower is better (closer to source in all metrics).
    """
    m = result.metrics
    return (
        m.get("mrstft", 0.0)
        + m.get("mfcc", 0.0) / 10.0
        + abs(m.get("hf_energy_delta_db", 0.0)) / 6.0
    )


def filter_candidates(
    audio_source: np.ndarray,
    candidates: list[np.ndarray],
    codegram_source: np.ndarray,
    codegram_candidates: list[np.ndarray],
    config: dict,
) -> list[tuple[int, AcceptanceResult]]:
    """Evaluate K candidates and return accepted ones, sorted by quality.

    Args:
        audio_source: 1D mono source audio, float32.
        candidates: List of candidate audio arrays.
        codegram_source: Source codegram [NQ, T].
        codegram_candidates: List of candidate codegrams.
        config: Config dict containing acceptance thresholds.

    Returns:
        List of (candidate_index, AcceptanceResult) for accepted candidates,
        sorted by composite quality score (best first).
    """
    accepted = []

    for i, (audio_cand, cg_cand) in enumerate(zip(candidates, codegram_candidates)):
        result = evaluate_candidate(
            audio_source, audio_cand, codegram_source, cg_cand, config,
        )
        if result.accepted:
            accepted.append((i, result))

    accepted.sort(key=lambda x: _composite_score(x[1]))

    logger.info(
        "Accepted %d / %d candidates", len(accepted), len(candidates),
    )
    return accepted
