#!/usr/bin/env python3
"""Extreme temperature sweep at 8x mask rate."""
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
from src.utils.instrument_families import infer_family

base_config = load_config("configs/default.yaml")
device = "cpu"

print("Loading DAC model...")
dac_model = load_dac_model()

model = VariationTransformer.from_config(base_config).to(device)
load_checkpoint("checkpoints/best.pt", model)
model.eval()

mask_mult = 2.0  # 8x
n_variations = 6


def find_samples_by_family(codegrams_dir: Path) -> dict[str, Path]:
    """Find one sample per instrument family."""
    found = {}
    for group_dir in sorted(codegrams_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        family = infer_family(group_dir.name)
        if family is None or family in found:
            continue
        npy_files = sorted(group_dir.glob("*.npy"))
        if npy_files:
            found[family] = npy_files[0]
    return found


def generate_for_sample(
    sample_path: Path,
    family: str,
    temperatures: list[float],
    output_dir: Path,
) -> None:
    """Generate machine guns at various temperatures for one sample."""
    output_dir.mkdir(parents=True, exist_ok=True)

    codegram = load_codegram(sample_path)
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

    for temp in temperatures:
        config = copy.deepcopy(base_config)
        config["masking"]["p_tail"] = base_config["masking"]["p_tail"] * mask_mult
        config["masking"]["p_attack"] = base_config["masking"]["p_attack"] * mask_mult
        config["sampling"]["temperature"] = temp

        label = f"8x_t{temp:.1f}"

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
        print(f"  {family} {label}: token change rate {avg_cr:.4f}")


# --- Phase 1: Extreme temps on CrossStick ---
print("\n=== CROSSSTICK — Extreme temperature sweep ===")
codegrams_dir = Path("data/codegrams/pass-02")
samples = find_samples_by_family(codegrams_dir)

crossstick_path = samples.get("CrossStick")
if crossstick_path:
    print(f"Using: {crossstick_path}")
    generate_for_sample(
        crossstick_path,
        "CrossStick",
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        Path("outputs/extreme_temp/CrossStick"),
    )

# --- Phase 2: Multi-family at a few key settings ---
print("\n=== MULTI-FAMILY — Key settings across instrument types ===")
# We'll generate these at the temperatures that seem interesting
# User can pick after listening to the extreme sweep
key_temps = [0.9, 1.1, 1.5, 2.0]

for family, sample_path in sorted(samples.items()):
    print(f"\n--- {family} ({sample_path.parent.name}) ---")
    generate_for_sample(
        sample_path,
        family,
        key_temps,
        Path(f"outputs/extreme_temp/{family}"),
    )

print("\n=== Done ===")
print("Files in: outputs/extreme_temp/<family>/")
print("Families found:", sorted(samples.keys()))
