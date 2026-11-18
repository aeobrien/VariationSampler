"""Tests for training loop."""

import torch
import pytest
import yaml
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from src.model.model import VariationTransformer
from src.model.train import compute_loss, train_one_epoch, save_checkpoint, load_checkpoint
from src.model.masking import build_mask


@pytest.fixture
def config():
    """Load default config."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


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
def small_batch(small_config):
    """Synthetic batch for small model."""
    mc = small_config["model"]
    b, nq, t = 4, mc["nq"], mc["t_max"]
    z_a = torch.randint(0, mc["codebook_size"], (b, nq, t))
    z_b = torch.randint(0, mc["codebook_size"], (b, nq, t))
    return z_a, z_b


class TestComputeLoss:

    def test_loss_finite_and_positive(self, small_model, small_batch, small_config):
        """Loss is a finite, positive scalar."""
        z_a, z_b = small_batch
        mask = build_mask(z_a.shape[0], z_a.shape[2], small_config)
        loss, metrics = compute_loss(small_model, z_a, z_b, mask, small_config)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_metrics_keys(self, small_model, small_batch, small_config):
        """Metrics dict has expected keys."""
        z_a, z_b = small_batch
        mask = build_mask(z_a.shape[0], z_a.shape[2], small_config)
        _, metrics = compute_loss(small_model, z_a, z_b, mask, small_config)
        expected_keys = {"ce_loss", "change_rate_penalty", "change_rate", "n_masked", "total_loss"}
        assert set(metrics.keys()) == expected_keys

    def test_change_rate_penalty_activates(self, small_model, small_config):
        """Change rate penalty is nonzero when change rate differs from target."""
        mc = small_config["model"]
        b, nq, t = 4, mc["nq"], mc["t_max"]
        z_a = torch.randint(0, mc["codebook_size"], (b, nq, t))
        z_b = torch.randint(0, mc["codebook_size"], (b, nq, t))

        # Use mask with many positions to get measurable change rate
        mask = torch.ones(b, len(mc["edit_codebooks"]), t, dtype=torch.bool)
        _, metrics = compute_loss(small_model, z_a, z_b, mask, small_config)
        assert metrics["change_rate_penalty"] > 0

    def test_loss_backward(self, small_model, small_batch, small_config):
        """Loss supports backpropagation."""
        z_a, z_b = small_batch
        mask = build_mask(z_a.shape[0], z_a.shape[2], small_config)
        loss, _ = compute_loss(small_model, z_a, z_b, mask, small_config)
        loss.backward()
        grads = [p.grad for p in small_model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestTrainOneEpoch:

    def test_one_step_reduces_loss(self, small_model, small_config):
        """One training step should reduce loss."""
        mc = small_config["model"]
        b, nq, t = 8, mc["nq"], mc["t_max"]
        z_a = torch.randint(0, mc["codebook_size"], (b, nq, t))
        z_b = z_a.clone()  # Same input/target — model should learn identity

        dataset = TensorDataset(z_a, z_b)
        loader = DataLoader(dataset, batch_size=4)

        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.01)

        # Get initial loss
        mask = build_mask(b, t, small_config)
        initial_loss, _ = compute_loss(small_model, z_a, z_b, mask, small_config)
        initial_loss_val = initial_loss.item()

        # Train several steps
        for _ in range(5):
            train_one_epoch(small_model, loader, optimizer, None, small_config)

        # Check loss decreased
        mask = build_mask(b, t, small_config)
        final_loss, _ = compute_loss(small_model, z_a, z_b, mask, small_config)
        assert final_loss.item() < initial_loss_val


class TestCheckpoint:

    def test_save_load_roundtrip(self, small_model, small_config, tmp_path):
        """Checkpoint save/load preserves model weights."""
        optimizer = torch.optim.Adam(small_model.parameters())
        ckpt_path = tmp_path / "test.pt"

        save_checkpoint(small_model, optimizer, epoch=7, path=ckpt_path)
        assert ckpt_path.exists()

        # Create fresh model with same config
        model2 = VariationTransformer.from_config(small_config)
        optimizer2 = torch.optim.Adam(model2.parameters())

        epoch = load_checkpoint(ckpt_path, model2, optimizer2)
        assert epoch == 7

        # Compare weights
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weight mismatch: {n1}"
