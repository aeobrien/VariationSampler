#!/usr/bin/env python3
"""Strategy A vs B comparison at 8x mask rate / T=2.0.

Generates machine-gun WAVs for:
  A — Standard masking (baseline)
  B — Buffer zone masking (ramp after attack frames)

Across all instrument families found in the codegrams directory.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.audio import save_wav, SAMPLE_RATE
from src.data.codegram_cache import load_codegram, load_dac_model, decode_codegram
from src.model.model import VariationTransformer
from src.model.train import load_checkpoint
from src.model.inference import generate_variation, generate_k_candidates
from src.model.masking import build_mask
from src.eval.machine_gun_proxy import render_machine_gun
from src.utils.instrument_families import infer_family

# Import buffer zone masking from strategy_comparison
from scripts.strategy_comparison import (
    build_mask_with_buffer,
    generate_k_candidates_with_buffer,
)

config = load_config("configs/default.yaml")
device = "cpu"

# Per-family ramp widths (from HANDOVER.md)
FAMILY_RAMP_WIDTHS = {
    "CrossStick": 5,
    "SnareRim": 5,
    "Snare": 5,
    "HiHat": 3,
    "Kick": 3,
    "Rimshot": 3,
}
DEFAULT_RAMP_WIDTH = 3

N_VARIATIONS = 6
K_CANDIDATES = 8  # generate more, pick best

output_dir = Path("outputs/strategy_ab")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading DAC model...")
dac_model = load_dac_model()

print("Loading model...")
model = VariationTransformer.from_config(config).to(device)
load_checkpoint("checkpoints/best.pt", model)
model.eval()

# Find one sample per family
codegrams_dir = Path("data/codegrams/pass-02")
families_found = {}
for group_dir in sorted(codegrams_dir.iterdir()):
    if not group_dir.is_dir():
        continue
    family = infer_family(group_dir.name)
    if family is None or family in families_found:
        continue
    npy_files = sorted(group_dir.glob("*.npy"))
    if npy_files:
        families_found[family] = {
            "path": npy_files[0],
            "group": group_dir.name,
        }

print(f"Found families: {sorted(families_found.keys())}")


def decode_and_trim(z_out: torch.Tensor, actual_t: int) -> np.ndarray:
    """Decode a codegram, truncating padding first."""
    z_trimmed = z_out[:, :actual_t]
    return decode_codegram(dac_model, z_trimmed.numpy())[0]


for family in sorted(families_found.keys()):
    info = families_found[family]
    sample_path = info["path"]
    print(f"\n{'='*60}")
    print(f"  {family} ({info['group']})")
    print(f"{'='*60}")

    codegram = load_codegram(sample_path)
    z_in = torch.from_numpy(codegram).long().to(device)
    actual_t = z_in.shape[1]

    t_max = config["model"]["t_max"]
    if z_in.shape[1] < t_max:
        padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=z_in.device)
        padded[:, :z_in.shape[1]] = z_in
        z_in = padded

    family_dir = output_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)

    # Source
    source_audio = decode_codegram(dac_model, codegram)[0]
    mg = render_machine_gun([source_audio] * 8, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(mg.reshape(1, -1).astype(np.float32), family_dir / "source_machinegun.wav")
    save_wav(source_audio.reshape(1, -1), family_dir / "source.wav")

    # Strategy A: standard masking
    print(f"  Strategy A (standard masking)...")
    a_audios = []
    a_change_rates = []
    for i in range(N_VARIATIONS):
        with torch.no_grad():
            z_out = generate_variation(model, z_in, config)
        a_change_rates.append((z_in != z_out).float().mean().item())
        a_audios.append(decode_and_trim(z_out, actual_t))

    mg_a = render_machine_gun(a_audios, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(mg_a.reshape(1, -1).astype(np.float32), family_dir / "A_machinegun.wav")
    avg_cr_a = sum(a_change_rates) / len(a_change_rates)

    # Strategy B: buffer zone masking
    ramp_width = FAMILY_RAMP_WIDTHS.get(family, DEFAULT_RAMP_WIDTH)
    print(f"  Strategy B (buffer zone, ramp_width={ramp_width})...")
    b_candidates = generate_k_candidates_with_buffer(
        model, z_in, K_CANDIDATES, config, ramp_width=ramp_width,
    )
    # Pick the first N_VARIATIONS (no acceptance filtering for this comparison)
    b_audios = []
    b_change_rates = []
    for z_out in b_candidates[:N_VARIATIONS]:
        b_change_rates.append((z_in != z_out).float().mean().item())
        b_audios.append(decode_and_trim(z_out, actual_t))

    mg_b = render_machine_gun(b_audios, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(mg_b.reshape(1, -1).astype(np.float32), family_dir / "B_machinegun.wav")
    avg_cr_b = sum(b_change_rates) / len(b_change_rates)

    print(f"  A: token change rate {avg_cr_a:.4f}")
    print(f"  B: token change rate {avg_cr_b:.4f}")

print(f"\nAll files in: {output_dir}")
print("For each family, compare:")
print("  source_machinegun.wav — identical hits (reference)")
print("  A_machinegun.wav     — standard masking")
print("  B_machinegun.wav     — buffer zone masking (should have cleaner attacks)")
