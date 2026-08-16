"""Training loop for the VariationTransformer."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.model import VariationTransformer
from src.model.masking import build_mask
from src.model.sampling import sample_tokens

logger = logging.getLogger(__name__)


def compute_loss(
    model: VariationTransformer,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    mask: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    """Compute masked cross-entropy loss with optional change rate penalty.

    Args:
        model: VariationTransformer instance.
        z_a: Input codegram [B, nq, T] long.
        z_b: Target codegram [B, nq, T] long.
        mask: Boolean mask [B, n_edit, T].
        config: Full config dict.

    Returns:
        (total_loss, metrics_dict) where metrics_dict contains 'ce_loss',
        'change_rate_penalty', 'change_rate', 'n_masked'.
    """
    edit_codebooks = config["model"]["edit_codebooks"]
    train_cfg = config["training"]
    lam = train_cfg["change_rate_penalty_lambda"]
    target_rate = train_cfg["target_change_rate"]

    logits = model(z_a, mask)  # [B, n_edit, T, V]

    # Build target from z_b at editable codebooks
    target = torch.stack(
        [z_b[:, cb, :] for cb in edit_codebooks], dim=1
    )  # [B, n_edit, T]

    # Masked cross-entropy: only at masked positions
    b, n_edit, t, v = logits.shape
    logits_flat = logits.reshape(-1, v)  # [B*n_edit*T, V]
    target_flat = target.reshape(-1)     # [B*n_edit*T]
    mask_flat = mask.reshape(-1)         # [B*n_edit*T]

    ce_all = nn.functional.cross_entropy(logits_flat, target_flat, reduction="none")
    n_masked = mask_flat.sum().clamp(min=1)
    ce_loss = (ce_all * mask_flat.float()).sum() / n_masked

    # Change rate penalty
    with torch.no_grad():
        sampled = sample_tokens(logits, temperature=0)  # greedy for penalty calc
        z_a_edit = torch.stack(
            [z_a[:, cb, :] for cb in edit_codebooks], dim=1
        )  # [B, n_edit, T]
        changed = (sampled != z_a_edit) & mask
        change_rate = changed.float().sum() / n_masked

    change_rate_penalty = lam * torch.abs(change_rate - target_rate)
    total_loss = ce_loss + change_rate_penalty

    metrics = {
        "ce_loss": ce_loss.item(),
        "change_rate_penalty": change_rate_penalty.item(),
        "change_rate": change_rate.item(),
        "n_masked": n_masked.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


def train_one_epoch(
    model: VariationTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: dict,
) -> dict:
    """Train for one epoch.

    Args:
        model: VariationTransformer instance.
        dataloader: DataLoader yielding (z_a, z_b) pairs.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler (step per batch).
        config: Full config dict.

    Returns:
        Dict of averaged metrics over the epoch.
    """
    model.train()
    grad_clip = config["training"]["grad_clip"]
    t_max = config["model"]["t_max"]

    epoch_metrics: dict[str, float] = {}
    n_batches = 0

    for z_a, z_b in dataloader:
        device = next(model.parameters()).device
        z_a = z_a.to(device)
        z_b = z_b.to(device)

        mask = build_mask(z_a.shape[0], z_a.shape[2], config).to(device)

        optimizer.zero_grad()
        loss, metrics = compute_loss(model, z_a, z_b, mask, config)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        n_batches += 1

    # Average
    for k in epoch_metrics:
        epoch_metrics[k] /= max(n_batches, 1)

    logger.info("Train epoch: %s", epoch_metrics)
    return epoch_metrics


@torch.no_grad()
def evaluate(
    model: VariationTransformer,
    dataloader: DataLoader,
    config: dict,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: VariationTransformer instance.
        dataloader: DataLoader yielding (z_a, z_b) pairs.
        config: Full config dict.

    Returns:
        Dict of averaged metrics.
    """
    model.eval()
    t_max = config["model"]["t_max"]

    epoch_metrics: dict[str, float] = {}
    n_batches = 0

    for z_a, z_b in dataloader:
        device = next(model.parameters()).device
        z_a = z_a.to(device)
        z_b = z_b.to(device)

        mask = build_mask(z_a.shape[0], z_a.shape[2], config).to(device)
        _, metrics = compute_loss(model, z_a, z_b, mask, config)

        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        n_batches += 1

    for k in epoch_metrics:
        epoch_metrics[k] /= max(n_batches, 1)

    logger.info("Eval: %s", epoch_metrics)
    return epoch_metrics


def save_checkpoint(
    model: VariationTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    """Save model and optimizer state.

    Args:
        model: VariationTransformer instance.
        optimizer: Optimizer.
        epoch: Current epoch number.
        path: Output file path (.pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    logger.info("Saved checkpoint: %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: Path,
    model: VariationTransformer,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load model and optimizer state from checkpoint.

    Args:
        path: Checkpoint file path (.pt).
        model: VariationTransformer to load weights into.
        optimizer: Optional optimizer to restore state.

    Returns:
        Epoch number from checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    logger.info("Loaded checkpoint: %s (epoch %d)", path, epoch)
    return epoch
