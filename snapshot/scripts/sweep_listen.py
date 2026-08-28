#!/usr/bin/env python3
"""Generate machine guns at different mask rates and temperatures for listening comparison."""
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

output_dir = Path("outputs/sweep_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Load DAC
print("Loading DAC model...")
dac_model = load_dac_model()

# Load model (use best/epoch 10)
model = VariationTransformer.from_config(base_config).to(device)
load_checkpoint("checkpoints/best.pt", model)
model.eval()

# Load codegram
codegrams_dir = Path("data/codegrams/pass-02")
npy_files = sorted(codegrams_dir.glob("**/*.npy"))
print(f"Using: {npy_files[0]}")

codegram = load_codegram(npy_files[0])
z_in = torch.from_numpy(codegram).long().to(device)
actual_t = z_in.shape[1]
print(f"Original length: {actual_t} frames")

t_max = base_config["model"]["t_max"]
if z_in.shape[1] < t_max:
    padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=z_in.device)
    padded[:, :z_in.shape[1]] = z_in
    z_in = padded

# Decode and save source
source_audio = decode_codegram(dac_model, codegram)[0]
save_wav(source_audio.reshape(1, -1), output_dir / "source.wav")
mg = render_machine_gun([source_audio] * 8, bpm=120.0, sr=SAMPLE_RATE)
save_wav(mg.reshape(1, -1).astype(np.float32), output_dir / "source_machinegun.wav")
print("Saved source")

# Sweep: (label, mask_multiplier, temperature)
# mask_multiplier scales p_tail and all codebook_multipliers relative to default.yaml (which is already 4x)
sweeps = [
    ("4x_t0.9",  1.0, 0.9),   # current default
    ("6x_t0.9",  1.5, 0.9),   # more masking, same temp
    ("8x_t0.9",  2.0, 0.9),   # even more masking
    ("4x_t1.1",  1.0, 1.1),   # same masking, higher temp
    ("4x_t1.3",  1.0, 1.3),   # same masking, much higher temp
    ("6x_t1.1",  1.5, 1.1),   # more masking + higher temp
    ("8x_t1.1",  2.0, 1.1),   # most masking + higher temp
]

n_variations = 6

for label, mask_mult, temp in sweeps:
    config = copy.deepcopy(base_config)

    # Scale mask rates
    config["masking"]["p_tail"] = base_config["masking"]["p_tail"] * mask_mult
    config["masking"]["p_attack"] = base_config["masking"]["p_attack"] * mask_mult
    for cb in config["masking"]["codebook_multipliers"]:
        config["masking"]["codebook_multipliers"][cb] = (
            base_config["masking"]["codebook_multipliers"][cb]
        )  # multipliers stay the same, p_tail scaling handles the rest

    # Set temperature
    config["sampling"]["temperature"] = temp

    effective_p_tail = config["masking"]["p_tail"]
    print(f"\n=== {label} (p_tail={effective_p_tail:.3f}, T={temp}) ===")

    variation_audios = []
    change_rates = []
    for i in range(n_variations):
        with torch.no_grad():
            z_out = generate_variation(model, z_in, config)

        diff = (z_in != z_out).float().mean().item()
        change_rates.append(diff)

        z_out_trimmed = z_out[:, :actual_t]
        audio_out = decode_codegram(dac_model, z_out_trimmed.numpy())[0]
        variation_audios.append(audio_out)

    mg = render_machine_gun(variation_audios, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(
        mg.reshape(1, -1).astype(np.float32),
        output_dir / f"{label}_machinegun.wav",
    )
    avg_cr = sum(change_rates) / len(change_rates)
    print(f"  Avg token change rate: {avg_cr:.4f}")
    print(f"  Saved {label}_machinegun.wav")

print(f"\nAll files in: {output_dir}")
