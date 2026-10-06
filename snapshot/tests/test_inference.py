"""Tests for inference utilities."""

import torch
import pytest
import yaml
from pathlib import Path

from src.model.model import VariationTransformer
from src.model.inference import generate_variation, generate_k_candidates


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return {
        "model": {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "dropout": 0.0,
            "nq": 9,
            "codebook_size": 128,
            "t_max": 16,
            "edit_codebooks": [6, 7, 8],
        },
        "masking": {
            "p_tail": 0.3,
            "p_attack": 0.1,
            "attack_frames": 2,
            "codebook_multipliers": {6: 1.0, 7: 1.0, 8: 1.0},
        },
        "sampling": {"temperature": 0.9, "top_p": 0.95},
        "training": {
            "learning_rate": 0.001,
            "batch_size": 4,
            "grad_clip": 1.0,
            "change_rate_penalty_lambda": 0.1,
            "target_change_rate": 0.05,
        },
    }


@pytest.fixture
def small_model(small_config):
    """Small model for fast tests."""
    return VariationTransformer.from_config(small_config)


@pytest.fixture
def z_in(small_config):
    """Synthetic input codegram [nq, T]."""
    mc = small_config["model"]
    return torch.randint(0, mc["codebook_size"], (mc["nq"], mc["t_max"]))


class TestGenerateVariation:

    def test_output_shape(self, small_model, z_in, small_config):
        """Output has same shape as input [nq, T]."""
        z_out = generate_variation(small_model, z_in, small_config)
        assert z_out.shape == z_in.shape

    def test_output_dtype(self, small_model, z_in, small_config):
        """Output is long tensor."""
        z_out = generate_variation(small_model, z_in, small_config)
        assert z_out.dtype == torch.long

    def test_output_value_range(self, small_model, z_in, small_config):
        """Output values are in valid codebook range."""
        z_out = generate_variation(small_model, z_in, small_config)
        v = small_config["model"]["codebook_size"]
        assert z_out.min() >= 0
        assert z_out.max() < v

    def test_non_edit_codebooks_preserved(self, small_model, z_in, small_config):
        """Codebooks 0-5 are unchanged (only 6-8 are editable)."""
        z_out = generate_variation(small_model, z_in, small_config)
        edit_codebooks = small_config["model"]["edit_codebooks"]
        for cb in range(small_config["model"]["nq"]):
            if cb not in edit_codebooks:
                assert torch.equal(z_out[cb], z_in[cb]), f"Codebook {cb} was modified"

    def test_no_grad_context(self, small_model, z_in, small_config):
        """Inference does not accumulate gradients."""
        generate_variation(small_model, z_in, small_config)
        for p in small_model.parameters():
            assert p.grad is None


class TestGenerateKCandidates:

    def test_returns_k_candidates(self, small_model, z_in, small_config):
        """Returns exactly k candidates."""
        k = 4
        candidates = generate_k_candidates(small_model, z_in, k, small_config)
        assert len(candidates) == k

    def test_candidate_shapes(self, small_model, z_in, small_config):
        """Each candidate has correct shape."""
        candidates = generate_k_candidates(small_model, z_in, 3, small_config)
        for c in candidates:
            assert c.shape == z_in.shape

    def test_candidates_not_all_identical(self, small_model, z_in, small_config):
        """K candidates are not all the same (random masks + sampling)."""
        candidates = generate_k_candidates(small_model, z_in, 8, small_config)
        # At least 2 should differ from each other
        all_same = all(torch.equal(candidates[0], c) for c in candidates[1:])
        assert not all_same, "All 8 candidates are identical"
