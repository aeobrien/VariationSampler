"""Tests for strategy comparison pipeline components."""

import json

import numpy as np
import pytest
import torch

from src.eval.acceptance import AcceptanceResult

# Import functions under test — the script uses sys.path manipulation,
# so we import from the scripts module after adding to path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.strategy_comparison import (
    build_mask_with_buffer,
    attack_quality_score,
    select_by_attack_score,
    select_by_composite,
    select_by_strict_acceptance,
    count_strict_passing,
    compute_selection_stats,
    make_strict_config,
    STRICT_THRESHOLDS,
    DEFAULT_FAMILY_RAMP_WIDTHS,
    LOW_ACCEPTANCE_THRESHOLD,
)


@pytest.fixture
def masking_config():
    """Minimal config dict for masking tests."""
    return {
        "model": {
            "edit_codebooks": [3, 4, 5, 6, 7, 8],
            "t_max": 30,
        },
        "masking": {
            "p_tail": 0.32,
            "p_attack": 0.02,
            "attack_frames": 3,
            "codebook_multipliers": {
                3: 0.24, 4: 0.48, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
            },
        },
    }


@pytest.fixture
def acceptance_config():
    """Config with acceptance thresholds."""
    return {
        "acceptance": {
            "mrstft_band": [0.1, 1.5],
            "mfcc_band": [2.0, 25.0],
            "min_token_change_rate": 0.01,
            "max_token_change_rate": 0.30,
            "min_attack_smear": 0.85,
            "min_transient_xcorr": 0.80,
            "max_hf_energy_delta_db": 6.0,
            "max_spectral_peak_divergence": 50,
            "attack_ms": 30,
        },
    }


def _make_candidate(index: int, xcorr: float, smear: float,
                     hf_delta: float, mrstft: float = 0.3,
                     mfcc: float = 5.0) -> dict:
    """Create a synthetic candidate dict for testing selection logic."""
    metrics = {
        "transient_xcorr": xcorr,
        "attack_smear": smear,
        "hf_energy_delta_db": hf_delta,
        "mrstft": mrstft,
        "mfcc": mfcc,
        "token_change_rate": 0.1,
        "spectral_peak_divergence": 2,
    }
    result = AcceptanceResult(
        accepted=True,
        metrics=metrics,
        reject_reasons=[],
    )
    return {
        "index": index,
        "codegram": np.zeros((9, 20), dtype=np.int32),
        "audio": np.zeros(1000, dtype=np.float32),
        "acceptance": result,
        "metrics": metrics,
    }


# ---- Test buffer zone mask ramp ----

class TestBufferZoneMask:
    """Tests for build_mask_with_buffer()."""

    def test_ramp_probabilities(self, masking_config):
        """Verify ramp probabilities at frames +1, +2, +3 after attack."""
        # Use a large batch to get stable empirical probabilities
        batch_size = 10000
        t_max = masking_config["model"]["t_max"]
        ramp_width = 3

        gen = torch.Generator().manual_seed(42)
        mask = build_mask_with_buffer(
            batch_size, t_max, masking_config,
            ramp_width=ramp_width, generator=gen,
        )
        # mask shape: [B, n_edit, t_max]

        attack_frames = masking_config["masking"]["attack_frames"]
        p_tail = masking_config["masking"]["p_tail"]

        # Check codebook 5 (index 2, mult=1.0) for clean expected values
        cb_idx = 2  # edit_codebooks[2] = 5, mult = 1.0

        # Ramp frame 1: p_tail * 0.25 * mult = 0.32 * 0.25 = 0.08
        frame_1 = attack_frames  # +0 in 0-indexed ramp → attack_frames + 1 - 1
        empirical_1 = mask[:, cb_idx, frame_1].float().mean().item()
        expected_1 = p_tail * (1 / (ramp_width + 1))  # 0.25
        assert abs(empirical_1 - expected_1) < 0.02, (
            f"Ramp frame 1: expected ~{expected_1:.3f}, got {empirical_1:.3f}"
        )

        # Ramp frame 2: p_tail * 0.50 * mult = 0.32 * 0.50 = 0.16
        frame_2 = attack_frames + 1
        empirical_2 = mask[:, cb_idx, frame_2].float().mean().item()
        expected_2 = p_tail * (2 / (ramp_width + 1))  # 0.50
        assert abs(empirical_2 - expected_2) < 0.02, (
            f"Ramp frame 2: expected ~{expected_2:.3f}, got {empirical_2:.3f}"
        )

        # Ramp frame 3: p_tail * 0.75 * mult = 0.32 * 0.75 = 0.24
        frame_3 = attack_frames + 2
        empirical_3 = mask[:, cb_idx, frame_3].float().mean().item()
        expected_3 = p_tail * (3 / (ramp_width + 1))  # 0.75
        assert abs(empirical_3 - expected_3) < 0.02, (
            f"Ramp frame 3: expected ~{expected_3:.3f}, got {empirical_3:.3f}"
        )

    def test_attack_frames_unchanged(self, masking_config):
        """Attack frames should still use p_attack * mult."""
        batch_size = 10000
        t_max = masking_config["model"]["t_max"]
        gen = torch.Generator().manual_seed(123)
        mask = build_mask_with_buffer(batch_size, t_max, masking_config, generator=gen)

        p_attack = masking_config["masking"]["p_attack"]
        # Codebook 5 (index 2, mult=1.0)
        cb_idx = 2
        empirical = mask[:, cb_idx, 0].float().mean().item()
        assert abs(empirical - p_attack) < 0.01, (
            f"Attack frame 0: expected ~{p_attack:.3f}, got {empirical:.3f}"
        )

    def test_tail_frames_normal(self, masking_config):
        """Frames past ramp should use normal p_tail * mult."""
        batch_size = 10000
        t_max = masking_config["model"]["t_max"]
        ramp_width = 3
        gen = torch.Generator().manual_seed(456)
        mask = build_mask_with_buffer(
            batch_size, t_max, masking_config,
            ramp_width=ramp_width, generator=gen,
        )

        attack_frames = masking_config["masking"]["attack_frames"]
        p_tail = masking_config["masking"]["p_tail"]

        # Frame past ramp: attack_frames + ramp_width
        tail_frame = attack_frames + ramp_width
        cb_idx = 2  # mult = 1.0
        empirical = mask[:, cb_idx, tail_frame].float().mean().item()
        assert abs(empirical - p_tail) < 0.02, (
            f"Tail frame {tail_frame}: expected ~{p_tail:.3f}, got {empirical:.3f}"
        )

    def test_multiplier_applied_to_ramp(self, masking_config):
        """Codebook multiplier should scale ramp probabilities."""
        batch_size = 10000
        t_max = masking_config["model"]["t_max"]
        ramp_width = 3
        gen = torch.Generator().manual_seed(789)
        mask = build_mask_with_buffer(
            batch_size, t_max, masking_config,
            ramp_width=ramp_width, generator=gen,
        )

        attack_frames = masking_config["masking"]["attack_frames"]
        p_tail = masking_config["masking"]["p_tail"]

        # Codebook 3 (index 0, mult=0.24)
        mult = 0.24
        frame_2 = attack_frames + 1  # ramp step 2
        empirical = mask[:, 0, frame_2].float().mean().item()
        expected = p_tail * (2 / (ramp_width + 1)) * mult
        assert abs(empirical - expected) < 0.015, (
            f"Codebook 3 ramp frame 2: expected ~{expected:.4f}, got {empirical:.4f}"
        )


# ---- Test attack score ranking ----

class TestAttackScoreRanking:

    def test_perfect_attack_scores_lowest(self):
        """Candidate with perfect attack metrics should score 0."""
        score = attack_quality_score({
            "transient_xcorr": 1.0,
            "attack_smear": 1.0,
            "hf_energy_delta_db": 0.0,
        })
        assert score == pytest.approx(0.0)

    def test_worse_attack_scores_higher(self):
        """Degraded attack should produce higher score."""
        good = attack_quality_score({
            "transient_xcorr": 0.98,
            "attack_smear": 0.97,
            "hf_energy_delta_db": 1.0,
        })
        bad = attack_quality_score({
            "transient_xcorr": 0.80,
            "attack_smear": 0.70,
            "hf_energy_delta_db": 5.0,
        })
        assert bad > good

    def test_select_by_attack_score_ordering(self):
        """select_by_attack_score should pick candidates with best attack."""
        candidates = [
            _make_candidate(0, xcorr=0.80, smear=0.70, hf_delta=5.0),  # bad
            _make_candidate(1, xcorr=0.99, smear=0.98, hf_delta=0.5),  # best
            _make_candidate(2, xcorr=0.95, smear=0.93, hf_delta=2.0),  # middle
            _make_candidate(3, xcorr=0.75, smear=0.65, hf_delta=4.0),  # worst
        ]
        selected = select_by_attack_score(candidates, 2)
        assert len(selected) == 2
        assert selected[0]["index"] == 1  # best attack
        assert selected[1]["index"] == 2  # second best


# ---- Test strict acceptance ----

class TestStrictAcceptance:

    def test_rejects_weak_xcorr(self, acceptance_config):
        """Candidates below strict xcorr threshold should be filtered."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.98, hf_delta=1.0),  # passes
            _make_candidate(1, xcorr=0.90, smear=0.98, hf_delta=1.0),  # fails xcorr
            _make_candidate(2, xcorr=0.96, smear=0.97, hf_delta=2.0),  # passes
        ]
        selected = select_by_strict_acceptance(candidates, 2, strict)
        indices = [c["index"] for c in selected]
        assert 0 in indices
        assert 2 in indices
        assert 1 not in indices

    def test_rejects_weak_smear(self, acceptance_config):
        """Candidates below strict smear threshold should be filtered."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.90, hf_delta=1.0),  # fails smear
            _make_candidate(1, xcorr=0.99, smear=0.98, hf_delta=1.0),  # passes
        ]
        selected = select_by_strict_acceptance(candidates, 1, strict)
        assert selected[0]["index"] == 1

    def test_rejects_high_hf_delta(self, acceptance_config):
        """Candidates with HF delta above strict threshold should be filtered."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.98, hf_delta=4.0),  # fails hf
            _make_candidate(1, xcorr=0.99, smear=0.98, hf_delta=2.0),  # passes
        ]
        selected = select_by_strict_acceptance(candidates, 1, strict)
        assert selected[0]["index"] == 1

    def test_fills_with_remaining_when_too_few_pass(self, acceptance_config):
        """If fewer pass strict thresholds than n, fill with best remaining."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.80, smear=0.70, hf_delta=5.0),  # fails all
            _make_candidate(1, xcorr=0.99, smear=0.98, hf_delta=1.0),  # passes
        ]
        selected = select_by_strict_acceptance(candidates, 2, strict)
        assert len(selected) == 2
        assert selected[0]["index"] == 1  # passing one first
        assert selected[1]["index"] == 0  # fallback

    def test_strict_thresholds_values(self):
        """Verify strict thresholds match specification."""
        assert STRICT_THRESHOLDS["min_transient_xcorr"] == 0.95
        assert STRICT_THRESHOLDS["min_attack_smear"] == 0.95
        assert STRICT_THRESHOLDS["max_hf_energy_delta_db"] == 3.0


# ---- Test manifest schema ----

class TestManifestSchema:

    def test_manifest_required_keys(self):
        """Verify a properly-formed manifest has all required keys."""
        manifest = {
            "generated_at": "2026-03-03T00:00:00+00:00",
            "config": {
                "checkpoint": "checkpoints/best.pt",
                "k_candidates": 32,
                "n_select": 8,
                "family_ramp_widths": {"CrossStick": 5, "SnareRim": 5},
                "config_file": "configs/default.yaml",
            },
            "strategies": {
                "A": "Baseline (standard masking + default scoring)",
                "B": "Buffer zone (ramp masking + default scoring)",
                "C": "Attack-scored (standard masking + attack quality scoring)",
                "D": "Strict acceptance (standard masking + tight attack thresholds)",
            },
            "samples": [
                {
                    "name": "CrossStick_Track1_CrossStick_v127",
                    "family": "CrossStick",
                    "ramp_width": 5,
                    "source_machinegun": "CrossStick_.../source_machinegun.wav",
                    "strategies": {
                        "A": "CrossStick_.../strategy_A_machinegun.wav",
                        "B": "CrossStick_.../strategy_B_machinegun.wav",
                        "C": "CrossStick_.../strategy_C_machinegun.wav",
                        "D": "CrossStick_.../strategy_D_machinegun.wav",
                    },
                    "metrics": {
                        "A": {"transient_xcorr": 0.92, "attack_smear": 0.90},
                        "B": {"transient_xcorr": 0.95, "attack_smear": 0.94},
                        "C": {"transient_xcorr": 0.97, "attack_smear": 0.96},
                        "D": {"transient_xcorr": 0.96, "attack_smear": 0.95},
                    },
                    "stats": {
                        "A": {"acceptance_rate": 1.0, "spectral_diversity": 0.5},
                        "B": {"acceptance_rate": 1.0, "spectral_diversity": 0.4},
                        "C": {"acceptance_rate": 1.0, "spectral_diversity": 0.3},
                        "D": {"acceptance_rate": 0.25, "spectral_diversity": 0.2},
                    },
                },
            ],
        }

        # Top-level keys
        assert "generated_at" in manifest
        assert "config" in manifest
        assert "strategies" in manifest
        assert "samples" in manifest

        # Config keys
        for key in ["checkpoint", "k_candidates", "n_select", "family_ramp_widths"]:
            assert key in manifest["config"], f"Missing config key: {key}"

        # All 4 strategies
        for s in ["A", "B", "C", "D"]:
            assert s in manifest["strategies"], f"Missing strategy: {s}"

        # Sample keys
        sample = manifest["samples"][0]
        for key in ["name", "family", "ramp_width", "source_machinegun",
                     "strategies", "metrics", "stats"]:
            assert key in sample, f"Missing sample key: {key}"

        # Each sample has all 4 strategies in paths, metrics, and stats
        for s in ["A", "B", "C", "D"]:
            assert s in sample["strategies"], f"Missing strategy path: {s}"
            assert s in sample["metrics"], f"Missing strategy metrics: {s}"
            assert s in sample["stats"], f"Missing strategy stats: {s}"

        # Stats contain required fields
        for s in ["A", "B", "C", "D"]:
            stats = sample["stats"][s]
            assert "acceptance_rate" in stats
            assert "spectral_diversity" in stats

    def test_manifest_roundtrip_json(self):
        """Manifest should survive JSON serialization."""
        manifest = {
            "generated_at": "2026-03-03T00:00:00+00:00",
            "config": {"checkpoint": "test.pt", "k_candidates": 32,
                       "n_select": 8, "family_ramp_widths": {}, "config_file": "test.yaml"},
            "strategies": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "samples": [],
        }
        serialized = json.dumps(manifest)
        deserialized = json.loads(serialized)
        assert deserialized == manifest


# ---- Test acceptance rate tracking ----

class TestAcceptanceRate:

    def test_count_strict_passing(self, acceptance_config):
        """count_strict_passing returns correct count."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.98, hf_delta=1.0),  # passes
            _make_candidate(1, xcorr=0.90, smear=0.98, hf_delta=1.0),  # fails
            _make_candidate(2, xcorr=0.96, smear=0.97, hf_delta=2.0),  # passes
            _make_candidate(3, xcorr=0.99, smear=0.80, hf_delta=1.0),  # fails
        ]
        assert count_strict_passing(candidates, strict) == 2

    def test_low_acceptance_threshold_value(self):
        """Low acceptance threshold should be 20%."""
        assert LOW_ACCEPTANCE_THRESHOLD == 0.20

    def test_compute_selection_stats_no_strict(self):
        """Without strict config, acceptance_rate is 1.0."""
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.98, hf_delta=1.0),
            _make_candidate(1, xcorr=0.95, smear=0.93, hf_delta=2.0),
        ]
        stats = compute_selection_stats(candidates, candidates, 32, strict_config=None)
        assert stats["acceptance_rate"] == 1.0
        assert stats["n_selected"] == 2

    def test_compute_selection_stats_with_strict(self, acceptance_config):
        """With strict config, acceptance_rate reflects passing count."""
        strict = make_strict_config(acceptance_config)
        candidates = [
            _make_candidate(0, xcorr=0.99, smear=0.98, hf_delta=1.0),  # passes
            _make_candidate(1, xcorr=0.90, smear=0.98, hf_delta=1.0),  # fails
            _make_candidate(2, xcorr=0.96, smear=0.97, hf_delta=2.0),  # passes
            _make_candidate(3, xcorr=0.80, smear=0.70, hf_delta=5.0),  # fails
        ]
        selected = [candidates[0], candidates[2]]
        stats = compute_selection_stats(candidates, selected, 4, strict_config=strict)
        assert stats["acceptance_rate"] == pytest.approx(0.5)
        assert stats["n_passing"] == 2
        assert stats["n_selected"] == 2


# ---- Test per-family ramp widths ----

class TestFamilyRampWidths:

    def test_default_ramp_widths(self):
        """CrossStick and SnareRim should get wider default buffers."""
        assert DEFAULT_FAMILY_RAMP_WIDTHS["CrossStick"] == 5
        assert DEFAULT_FAMILY_RAMP_WIDTHS["SnareRim"] == 5
        assert DEFAULT_FAMILY_RAMP_WIDTHS["HiHat"] == 3
        assert DEFAULT_FAMILY_RAMP_WIDTHS["Kick"] == 3

    def test_wider_ramp_reduces_early_frame_probability(self, masking_config):
        """Wider ramp should produce lower probability at the first ramp frame."""
        batch_size = 10000
        t_max = masking_config["model"]["t_max"]
        attack_frames = masking_config["masking"]["attack_frames"]
        p_tail = masking_config["masking"]["p_tail"]

        # Narrow ramp (width=3): first ramp frame gets p_tail * 1/4 = 0.08
        gen3 = torch.Generator().manual_seed(42)
        mask_narrow = build_mask_with_buffer(
            batch_size, t_max, masking_config, ramp_width=3, generator=gen3,
        )
        narrow_p = mask_narrow[:, 2, attack_frames].float().mean().item()

        # Wide ramp (width=5): first ramp frame gets p_tail * 1/6 ≈ 0.053
        gen5 = torch.Generator().manual_seed(42)
        mask_wide = build_mask_with_buffer(
            batch_size, t_max, masking_config, ramp_width=5, generator=gen5,
        )
        wide_p = mask_wide[:, 2, attack_frames].float().mean().item()

        assert wide_p < narrow_p, (
            f"Wider ramp should have lower first-frame prob: "
            f"wide={wide_p:.4f} vs narrow={narrow_p:.4f}"
        )
