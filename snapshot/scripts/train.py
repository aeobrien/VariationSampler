#!/usr/bin/env python3
"""Training script for the VariationTransformer.

Usage:
    # Dry run (verify pipeline locally, ~30s)
    python scripts/train.py --config configs/default.yaml --dry-run

    # Full training
    python scripts/train.py --config configs/default.yaml

    # With W&B logging
    python scripts/train.py --config configs/default.yaml --wandb --wandb-project variation-sampler

    # Override device
    python scripts/train.py --config configs/default.yaml --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.dataset import CodegramPairDataset
from src.model.model import VariationTransformer
from src.model.train import (
    compute_loss,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)
from src.model.masking import build_mask

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def detect_device(override: str | None = None) -> torch.device:
    """Auto-detect best available device."""
    if override:
        device = torch.device(override)
        logger.info("Device override: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using CUDA: %s (%.1f GB)", name, mem)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def normalize_path(raw_path: str) -> str:
    """Normalize a path from split files: strip absolute prefixes, make relative to PROJECT_ROOT.

    Handles both relative paths (data/processed/...) and absolute paths
    (/Users/.../VariationSampler/data/processed/...) written by older split generation.
    """
    p = Path(raw_path)
    if p.is_absolute():
        # Try to find the 'data/' component and make relative from there
        parts = p.parts
        for i, part in enumerate(parts):
            if part == "data":
                return str(Path(*parts[i:]))
        # Fallback: try relative to PROJECT_ROOT
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return raw_path
    return raw_path


def wav_path_to_codegram_path(wav_path: str) -> str:
    """Convert processed WAV path to cached codegram .npy path.

    data/processed/pass-02/X/hit_01.wav -> data/codegrams/pass-02/X/hit_01.npy
    """
    normalized = normalize_path(wav_path)
    p = Path(normalized)
    parts = list(p.parts)
    try:
        idx = parts.index("processed")
        parts[idx] = "codegrams"
    except ValueError:
        raise ValueError(f"Expected 'processed' in path: {wav_path}")
    return str(Path(*parts).with_suffix(".npy"))


def load_pairs(pairs_path: Path) -> list[tuple[str, str]]:
    """Load training pairs JSON and convert to codegram paths."""
    with open(pairs_path) as f:
        raw_pairs = json.load(f)

    # Detect and warn about absolute paths
    if raw_pairs and Path(raw_pairs[0][0]).is_absolute():
        logger.warning(
            "Split file contains absolute paths — stripping prefixes. "
            "Regenerate splits to fix permanently."
        )

    codegram_pairs = []
    missing_paths: dict[str, int] = {}
    convert_errors = 0

    for pair in raw_pairs:
        try:
            a = wav_path_to_codegram_path(pair[0])
            b = wav_path_to_codegram_path(pair[1])
            a_full = PROJECT_ROOT / a
            b_full = PROJECT_ROOT / b
            a_exists = a_full.exists()
            b_exists = b_full.exists()
            if a_exists and b_exists:
                codegram_pairs.append((str(a_full), str(b_full)))
            else:
                if not a_exists:
                    # Track missing by group dir for useful reporting
                    group = str(Path(a).parent)
                    missing_paths[group] = missing_paths.get(group, 0) + 1
                if not b_exists:
                    group = str(Path(b).parent)
                    missing_paths[group] = missing_paths.get(group, 0) + 1
        except (ValueError, IndexError):
            convert_errors += 1

    total_skipped = len(raw_pairs) - len(codegram_pairs)
    if total_skipped > 0:
        logger.warning(
            "Skipped %d / %d pairs (%d convert errors, %d missing codegrams)",
            total_skipped, len(raw_pairs), convert_errors,
            total_skipped - convert_errors,
        )
        if missing_paths:
            # Show top missing groups
            sorted_missing = sorted(missing_paths.items(), key=lambda x: -x[1])
            logger.warning("Top missing codegram groups:")
            for group, count in sorted_missing[:10]:
                logger.warning("  %s: %d missing files", group, count)
            if len(sorted_missing) > 10:
                logger.warning("  ... and %d more groups", len(sorted_missing) - 10)

    return codegram_pairs


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_summary(
    config: dict,
    train_pairs: int,
    dev_pairs: int,
    model: VariationTransformer,
    device: torch.device,
    dry_run: bool,
) -> None:
    """Print training summary at startup."""
    n_params = count_parameters(model)
    tc = config["training"]
    batch_size = tc["batch_size"]
    max_epochs = tc["max_epochs"]
    steps_per_epoch = (train_pairs + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * max_epochs

    print("\n" + "=" * 60)
    print("  VariationTransformer — Training Summary")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Model:           d={config['model']['d_model']}, "
          f"layers={config['model']['n_layers']}, "
          f"heads={config['model']['n_heads']}")
    print(f"  Edit codebooks:  {config['model']['edit_codebooks']}")
    print(f"  Train pairs:     {train_pairs:,}")
    print(f"  Dev pairs:       {dev_pairs:,}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Steps/epoch:     {steps_per_epoch:,}")
    print(f"  Max epochs:      {max_epochs}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  LR:              {tc['learning_rate']}")
    print(f"  Warmup steps:    {tc['warmup_steps']}")
    if dry_run:
        print(f"  Mode:            DRY RUN (3 steps)")
    print("=" * 60 + "\n")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay scheduler."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    config = load_config(Path(args.config))
    device = detect_device(args.device)
    tc = config["training"]

    # Load data pairs
    splits_dir = DATA_DIR / "splits"
    train_pairs = load_pairs(splits_dir / "train_pairs.json")
    dev_pairs = load_pairs(splits_dir / "dev_pairs.json")

    if not train_pairs:
        logger.error("No training pairs found. Run import_training_data.py splits + encode first.")
        sys.exit(1)

    # Dry run: use tiny subset
    if args.dry_run:
        train_pairs = train_pairs[:10]
        dev_pairs = dev_pairs[:10] if dev_pairs else train_pairs[:5]
        config["training"]["max_epochs"] = 1
        config["training"]["batch_size"] = min(4, len(train_pairs))

    # Datasets and loaders
    train_dataset = CodegramPairDataset(train_pairs, max_frames=config["model"]["t_max"])
    dev_dataset = CodegramPairDataset(dev_pairs, max_frames=config["model"]["t_max"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = VariationTransformer.from_config(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tc["max_epochs"]
    scheduler = build_scheduler(optimizer, tc["warmup_steps"], total_steps)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer) + 1
            logger.info("Resuming from epoch %d", start_epoch)
        else:
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)

    # Print summary
    print_summary(config, len(train_pairs), len(dev_pairs), model, device, args.dry_run)

    # W&B init
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=config,
                name=args.wandb_name,
                tags=["dry-run"] if args.dry_run else None,
            )
            logger.info("W&B run: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    # Training loop
    checkpoint_every = tc.get("checkpoint_every_epochs", 5)
    eval_every = tc.get("eval_every_epochs", 1)
    global_step = start_epoch * steps_per_epoch

    best_dev_loss = float("inf")
    early_stopping_patience = tc.get("early_stopping_patience")
    patience_counter = 0

    # On resume, run a dev eval to restore best_dev_loss so early stopping works
    if args.resume and start_epoch > 0:
        logger.info("Running dev eval to establish baseline for early stopping...")
        baseline_metrics = evaluate(model, dev_loader, config)
        best_dev_loss = baseline_metrics.get("total_loss", float("inf"))
        logger.info("Resumed best_dev_loss: %.4f", best_dev_loss)

    for epoch in range(start_epoch, tc["max_epochs"]):
        epoch_start = time.time()
        model.train()

        epoch_metrics: dict[str, float] = {}
        n_batches = 0

        for batch_idx, (z_a, z_b) in enumerate(train_loader):
            step_start = time.time()
            z_a = z_a.to(device)
            z_b = z_b.to(device)

            mask = build_mask(z_a.shape[0], z_a.shape[2], config).to(device)

            optimizer.zero_grad()
            loss, metrics = compute_loss(model, z_a, z_b, mask, config)
            loss.backward()

            if tc["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])

            optimizer.step()
            scheduler.step()

            global_step += 1
            step_time = time.time() - step_start

            # Accumulate epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            n_batches += 1

            # Per-step logging
            current_lr = optimizer.param_groups[0]["lr"]
            if batch_idx % max(1, steps_per_epoch // 20) == 0 or args.dry_run:
                logger.info(
                    "Epoch %d step %d/%d | loss=%.4f ce=%.4f cr=%.4f "
                    "lr=%.2e | %.2fs/step",
                    epoch, batch_idx, steps_per_epoch,
                    metrics["total_loss"], metrics["ce_loss"],
                    metrics["change_rate"], current_lr, step_time,
                )

            if wandb_run:
                import wandb
                wandb.log({
                    "train/loss": metrics["total_loss"],
                    "train/ce_loss": metrics["ce_loss"],
                    "train/change_rate": metrics["change_rate"],
                    "train/change_rate_penalty": metrics["change_rate_penalty"],
                    "train/lr": current_lr,
                    "train/step_time": step_time,
                }, step=global_step)

            # Dry run: stop after 3 steps
            if args.dry_run and batch_idx >= 2:
                logger.info("Dry run: stopping after 3 training steps")
                break

        # Average epoch metrics
        epoch_time = time.time() - epoch_start
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        logger.info(
            "Epoch %d complete | avg_loss=%.4f avg_ce=%.4f avg_cr=%.4f | %s",
            epoch, epoch_metrics.get("total_loss", 0),
            epoch_metrics.get("ce_loss", 0),
            epoch_metrics.get("change_rate", 0),
            format_time(epoch_time),
        )

        # Dev evaluation
        if (epoch + 1) % eval_every == 0 or args.dry_run:
            dev_metrics = evaluate(model, dev_loader, config)

            if wandb_run:
                import wandb
                wandb.log({
                    f"dev/{k}": v for k, v in dev_metrics.items()
                }, step=global_step)

            dev_loss = dev_metrics.get("total_loss", float("inf"))
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_path = CHECKPOINT_DIR / "best.pt"
                save_checkpoint(model, optimizer, epoch, best_path)
                logger.info("New best dev loss: %.4f", dev_loss)
                patience_counter = 0
            elif early_stopping_patience is not None:
                patience_counter += 1
                logger.info(
                    "No dev improvement (%d/%d patience)",
                    patience_counter, early_stopping_patience,
                )
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d evals)",
                        epoch, early_stopping_patience,
                    )
                    if wandb_run:
                        import wandb
                        wandb.log({"early_stopped_epoch": epoch}, step=global_step)
                    break

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)

        if args.dry_run:
            break

    # Final checkpoint
    if not args.dry_run:
        final_path = CHECKPOINT_DIR / "final.pt"
        save_checkpoint(model, optimizer, tc["max_epochs"] - 1, final_path)

    # Dry run verification
    if args.dry_run:
        logger.info("--- Dry Run Verification ---")
        logger.info("  Data loading:    OK (%d train, %d dev pairs)", len(train_pairs), len(dev_pairs))
        logger.info("  Forward pass:    OK")
        logger.info("  Loss + backward: OK")
        logger.info("  Checkpoint:      OK (%s)", CHECKPOINT_DIR / "best.pt")
        logger.info("  Dev evaluation:  OK")

        # Verify checkpoint roundtrip
        model2 = VariationTransformer.from_config(config).to(device)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=tc["learning_rate"])
        loaded_epoch = load_checkpoint(CHECKPOINT_DIR / "best.pt", model2, opt2)
        logger.info("  Checkpoint load: OK (epoch %d)", loaded_epoch)
        logger.info("--- Dry Run Complete ---")

    if wandb_run:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the VariationTransformer.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 3 training steps on 10 pairs to verify pipeline.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda, mps, cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pt file to resume from.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader num_workers (default: 0 for main process).",
    )

    # W&B options
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="variation-sampler",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None,
        help="W&B run name.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.dry_run else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    train(args)


if __name__ == "__main__":
    main()
