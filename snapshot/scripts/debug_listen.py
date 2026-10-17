#!/usr/bin/env python3
"""Generate listenable variations from different checkpoints for A/B comparison."""
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

config = load_config("configs/default.yaml")
device = "cpu"

output_dir = Path("outputs/checkpoint_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Load DAC
print("Loading DAC model (this takes a moment on CPU)...")
dac_model = load_dac_model()

# Load codegram
codegrams_dir = Path("data/codegrams/pass-02")
npy_files = sorted(codegrams_dir.glob("**/*.npy"))
print(f"Using: {npy_files[0]}")

codegram = load_codegram(npy_files[0])
z_in = torch.from_numpy(codegram).long().to(device)
actual_t = z_in.shape[1]  # remember original length before padding
print(f"Original codegram length: {actual_t} frames")

t_max = config["model"]["t_max"]
if z_in.shape[1] < t_max:
    padded = torch.zeros(z_in.shape[0], t_max, dtype=torch.long, device=z_in.device)
    padded[:, :z_in.shape[1]] = z_in
    z_in = padded

# Decode source
source_audio = decode_codegram(dac_model, codegram)[0]  # 1D
save_wav(source_audio.reshape(1, -1), output_dir / "source.wav")

# Render source machine gun
mg = render_machine_gun([source_audio] * 8, bpm=120.0, sr=SAMPLE_RATE)
save_wav(mg.reshape(1, -1).astype(np.float32), output_dir / "source_machinegun.wav")
print("Saved source + source machine gun")

checkpoints = [
    ("epoch_0000", "checkpoints/epoch_0000.pt"),
    ("epoch_0001", "checkpoints/epoch_0001.pt"),
    ("epoch_0002", "checkpoints/epoch_0002.pt"),
    ("best_ep10", "checkpoints/best.pt"),
]

n_variations = 6

for ckpt_name, ckpt_path in checkpoints:
    if not Path(ckpt_path).exists():
        print(f"SKIP {ckpt_path}")
        continue

    model = VariationTransformer.from_config(config).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()
    print(f"\n=== {ckpt_name} ===")

    variation_audios = []
    for i in range(n_variations):
        with torch.no_grad():
            z_out = generate_variation(model, z_in, config)

        # Truncate back to original length before decoding
        z_out_trimmed = z_out[:, :actual_t]
        audio_out = decode_codegram(dac_model, z_out_trimmed.numpy())[0]
        variation_audios.append(audio_out)

        # Save individual hit
        save_wav(
            audio_out.reshape(1, -1),
            output_dir / f"{ckpt_name}_var{i+1}.wav",
        )

    # Render machine gun with the variations
    mg = render_machine_gun(variation_audios, bpm=120.0, sr=SAMPLE_RATE)
    save_wav(
        mg.reshape(1, -1).astype(np.float32),
        output_dir / f"{ckpt_name}_machinegun.wav",
    )
    print(f"  Saved {n_variations} variations + machine gun")

print(f"\nAll files in: {output_dir}")
print("Compare: source_machinegun.wav vs epoch_0000_machinegun.wav vs best_ep10_machinegun.wav")
