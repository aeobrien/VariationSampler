#!/usr/bin/env python3
"""Diagnostic: compare token change rates across checkpoints."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.codegram_cache import load_codegram
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.model.inference import generate_variation
from src.model.masking import build_mask

config = load_config("configs/default.yaml")
print(f"p_tail: {config['masking']['p_tail']}")
print(f"p_attack: {config['masking']['p_attack']}")
print(f"multipliers: {config['masking']['codebook_multipliers']}")

device = "cpu"

# Find a codegram
codegrams_dir = Path("data/codegrams/pass-02")
npy_files = sorted(codegrams_dir.glob("**/*.npy"))
if not npy_files:
    print(f"ERROR: No .npy files found under {codegrams_dir}")
    sys.exit(1)
print(f"Found {len(npy_files)} codegram files")
print(f"Using: {npy_files[0]}")

codegram = load_codegram(npy_files[0])
z_in = torch.from_numpy(codegram).long().to(device)

# Pad to t_max if needed
t_max = config["model"]["t_max"]
if z_in.shape[1] < t_max:
    padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=z_in.device)
    padded[:, :z_in.shape[1]] = z_in
    z_in = padded

print(f"Input shape: {z_in.shape}")
print()

# Test each checkpoint
checkpoints = [
    "checkpoints/epoch_0000.pt",
    "checkpoints/epoch_0001.pt",
    "checkpoints/epoch_0002.pt",
    "checkpoints/best.pt",
]

for ckpt_path in checkpoints:
    if not Path(ckpt_path).exists():
        print(f"SKIP {ckpt_path} (not found)")
        continue

    model = VariationTransformer.from_config(config).to(device)
    epoch = load_checkpoint(ckpt_path, model)
    model.eval()

    print(f"=== {ckpt_path} (epoch {epoch}) ===")

    # Generate 3 variations, average the change rates
    total_rates = []
    cb_rates = {cb: [] for cb in config["model"]["edit_codebooks"]}

    for i in range(3):
        with torch.no_grad():
            z_out = generate_variation(model, z_in, config)
        diff = (z_in != z_out)
        total_rates.append(diff.float().mean().item())
        for cb in config["model"]["edit_codebooks"]:
            cb_rates[cb].append((z_in[cb] != z_out[cb]).float().mean().item())

    avg_total = sum(total_rates) / len(total_rates)
    print(f"  Avg token change rate: {avg_total:.4f}")
    for cb in config["model"]["edit_codebooks"]:
        avg_cb = sum(cb_rates[cb]) / len(cb_rates[cb])
        print(f"    Codebook {cb}: {avg_cb:.4f}")
    print()
