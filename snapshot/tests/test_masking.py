"""Tests for mask generation."""

import torch
import pytest
import yaml
from pathlib import Path

from src.model.masking import build_mask


@pytest.fixture
def config():
    """Load default config."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class TestBuildMask:

    def test_output_shape(self, config):
        """Mask has correct shape [B, n_edit, T]."""
        b, t = 8, config["model"]["t_max"]
        n_edit = len(config["model"]["edit_codebooks"])
        mask = build_mask(b, t, config)
        assert mask.shape == (b, n_edit, t)

    def test_output_dtype(self, config):
        """Mask is boolean."""
        mask = build_mask(4, config["model"]["t_max"], config)
        assert mask.dtype == torch.bool

    def test_seed_reproducibility(self, config):
        """Same generator seed produces same mask."""
        t = config["model"]["t_max"]
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        m1 = build_mask(4, t, config, generator=g1)
        m2 = build_mask(4, t, config, generator=g2)
        assert torch.equal(m1, m2)

    def test_different_seeds_differ(self, config):
        """Different seeds produce different masks."""
        t = config["model"]["t_max"]
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        m1 = build_mask(4, t, config, generator=g1)
        m2 = build_mask(4, t, config, generator=g2)
        assert not torch.equal(m1, m2)

    def test_codebook_gradient(self, config):
        """Earlier codebooks (lower index) have lower mask rate than later ones.

        Codebook 3 (mult=0.06) should have much fewer masks than codebook 8 (mult=1.0).
        """
        # Large batch for statistical reliability
        t = config["model"]["t_max"]
        mask = build_mask(1000, t, config)

        edit_codebooks = config["model"]["edit_codebooks"]
        cb3_idx = edit_codebooks.index(3)
        cb8_idx = edit_codebooks.index(8)

        rate_cb3 = mask[:, cb3_idx, :].float().mean().item()
        rate_cb8 = mask[:, cb8_idx, :].float().mean().item()
        assert rate_cb3 < rate_cb8, f"cb3 rate {rate_cb3} >= cb8 rate {rate_cb8}"

    def test_attack_lower_than_tail(self, config):
        """Attack frames have lower mask rate than tail frames.

        Only test on codebook 8 (mult=1.0) for clearest signal.
        """
        t = config["model"]["t_max"]
        attack_frames = config["masking"]["attack_frames"]
        mask = build_mask(2000, t, config)

        edit_codebooks = config["model"]["edit_codebooks"]
        cb8_idx = edit_codebooks.index(8)

        attack_rate = mask[:, cb8_idx, :attack_frames].float().mean().item()
        tail_rate = mask[:, cb8_idx, attack_frames:].float().mean().item()
        assert attack_rate < tail_rate, (
            f"Attack rate {attack_rate} >= tail rate {tail_rate}"
        )

    def test_zero_probability_gives_empty_mask(self):
        """All-zero probabilities produce empty mask."""
        config = {
            "model": {"edit_codebooks": [3, 4, 5, 6, 7, 8]},
            "masking": {
                "p_tail": 0.0,
                "p_attack": 0.0,
                "attack_frames": 2,
                "codebook_multipliers": {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
            },
        }
        mask = build_mask(4, 64, config)
        assert mask.sum() == 0

    def test_full_probability_gives_full_mask(self):
        """p=1.0 with multiplier=1.0 masks everything."""
        config = {
            "model": {"edit_codebooks": [7, 8]},
            "masking": {
                "p_tail": 1.0,
                "p_attack": 1.0,
                "attack_frames": 2,
                "codebook_multipliers": {7: 1.0, 8: 1.0},
            },
        }
        mask = build_mask(4, 64, config)
        assert mask.all()
