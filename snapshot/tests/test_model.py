"""Tests for VariationTransformer model."""

import torch
import pytest
import yaml
from pathlib import Path

from src.model.model import VariationTransformer


@pytest.fixture
def config():
    """Load default config."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(config):
    """Create a small VariationTransformer for testing."""
    return VariationTransformer.from_config(config)


@pytest.fixture
def batch(config):
    """Synthetic batch of codegrams and masks."""
    b, nq, t = 4, config["model"]["nq"], config["model"]["t_max"]
    n_edit = len(config["model"]["edit_codebooks"])
    z_in = torch.randint(0, 1024, (b, nq, t))
    mask = torch.rand(b, n_edit, t) > 0.9
    return z_in, mask


class TestVariationTransformer:

    def test_output_shape(self, model, batch, config):
        """Forward pass produces correct output shape."""
        z_in, mask = batch
        logits = model(z_in, mask)
        b = z_in.shape[0]
        n_edit = len(config["model"]["edit_codebooks"])
        t = config["model"]["t_max"]
        v = config["model"]["codebook_size"]
        assert logits.shape == (b, n_edit, t, v)

    def test_output_dtype(self, model, batch):
        """Logits are float32."""
        z_in, mask = batch
        logits = model(z_in, mask)
        assert logits.dtype == torch.float32

    def test_logits_finite(self, model, batch):
        """All logits are finite (no NaN/Inf)."""
        z_in, mask = batch
        logits = model(z_in, mask)
        assert torch.isfinite(logits).all()

    def test_different_mask_different_output(self, model, config):
        """Different masks produce different logits."""
        b, nq, t = 2, config["model"]["nq"], config["model"]["t_max"]
        n_edit = len(config["model"]["edit_codebooks"])
        z_in = torch.randint(0, 1024, (b, nq, t))

        mask1 = torch.zeros(b, n_edit, t, dtype=torch.bool)
        mask1[:, :, 10:20] = True

        mask2 = torch.zeros(b, n_edit, t, dtype=torch.bool)
        mask2[:, :, 50:60] = True

        logits1 = model(z_in, mask1)
        logits2 = model(z_in, mask2)
        assert not torch.allclose(logits1, logits2)

    def test_gradient_flows(self, model, batch):
        """Gradients flow through the model."""
        z_in, mask = batch
        logits = model(z_in, mask)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_from_config(self, config):
        """from_config constructs model with correct dimensions."""
        model = VariationTransformer.from_config(config)
        assert model.d_model == config["model"]["d_model"]
        assert model.n_layers == config["model"]["n_layers"]
        assert model.n_heads == config["model"]["n_heads"]
        assert model.edit_codebooks == config["model"]["edit_codebooks"]

    def test_custom_dimensions(self):
        """Model works with non-default dimensions."""
        model = VariationTransformer(
            d_model=64, n_layers=2, n_heads=2,
            nq=9, codebook_size=512, t_max=32,
            edit_codebooks=[6, 7, 8],
        )
        z_in = torch.randint(0, 512, (2, 9, 32))
        mask = torch.rand(2, 3, 32) > 0.8
        logits = model(z_in, mask)
        assert logits.shape == (2, 3, 32, 512)

    def test_no_mask_still_works(self, model, config):
        """Model produces output even with all-False mask."""
        b, nq, t = 2, config["model"]["nq"], config["model"]["t_max"]
        n_edit = len(config["model"]["edit_codebooks"])
        z_in = torch.randint(0, 1024, (b, nq, t))
        mask = torch.zeros(b, n_edit, t, dtype=torch.bool)
        logits = model(z_in, mask)
        assert torch.isfinite(logits).all()
