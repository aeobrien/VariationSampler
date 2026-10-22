#!/usr/bin/env python3
"""Temperature sweep at 8x mask rate."""
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
from src.model.inference import generate_variation
from src.eval.machine_gun_proxy import render_machine_gun

base_config = load_config("configs/default.yaml")
device = "cpu"

output_dir = Path("outputs/temp_sweep_8x")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading DAC model...")
dac_model = load_dac_model()

model = VariationTransformer.from_config(base_config).to(device)
load_checkpoint("checkpoints/best.pt", model)
model.eval()

codegrams_dir = Path("data/codegrams/pass-02")
npy_files = sorted(codegrams_dir.glob("**/*.npy"))
print(f"Using: {npy_files[0]}")

codegram = load_codegram(npy_files[0])
z_in = torch.from_numpy(codegram).long().to(device)
actual_t = z_in.shape[1]

t_max = base_config["model"]["t_max"]
if z_in.shape[1] < t_max:
    padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=z_in.device)
    padded[:, :z_in.shape[1]] = z_in
    z_in = padded

# Source
source_audio = decode_codegram(dac_model, codegram)[0]
mg = render_machine_gun([source_audio] * 8, bpm=120.0, sr=SAMPLE_RATE)
save_wav(mg.reshape(1, -1).astype(np.float32), output_dir / "source_machinegun.wav")
print("Saved source")

# 8x mask rate = 2.0 * default p_tail
mask_mult = 2.0
temperatures = [0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
n_variations = 6

for temp in temperatures:
    config = copy.deepcopy(base_config)
    config["masking"]["p_tail"] = base_config["masking"]["p_tail"] * mask_mult
    config["masking"]["p_attack"] = base_config["masking"]["p_attack"] * mask_mult
    config["sampling"]["temperature"] = temp

    label = f"8x_t{temp:.1f}"
    print(f"\n=== {label} ===")

    variation_audios = []
    change_rates = []
    for i in range(n_variations):
        with torch.no_grad():
            z_out = generate_variation(model, z_in, config)
        change_rates.append((z_in != z_out).float().mean().item())
        z_out_trimmed = z_out[:, :actual_t]
        audio_out = decode_codegram(dac_model, z_out_trimmed.numpy())[0]
        variation_audios.append(audio_out)

    mg = render_machine_gun(variation_audios, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(
        mg.reshape(1, -1).astype(np.float32),
        output_dir / f"{label}_machinegun.wav",
    )
    avg_cr = sum(change_rates) / len(change_rates)
    print(f"  Token change rate: {avg_cr:.4f}, saved {label}_machinegun.wav")

print(f"\nAll files in: {output_dir}")
