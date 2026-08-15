"""Inference utilities for generating variations."""

import logging

import torch

from src.model.model import VariationTransformer
from src.model.masking import build_mask
from src.model.sampling import sample_tokens, apply_mask

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_variation(
    model: VariationTransformer,
    z_in: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Generate a single variation from an input codegram.

    Args:
        model: Trained VariationTransformer.
        z_in: Input codegram [nq, T] long.
        config: Full config dict.

    Returns:
        Output codegram [nq, T] long.
    """
    model.eval()
    device = next(model.parameters()).device

    # Add batch dimension
    z_batch = z_in.unsqueeze(0).to(device)  # [1, nq, T]
    t = z_in.shape[1]

    mask = build_mask(1, t, config).to(device)  # [1, n_edit, T]

    logits = model(z_batch, mask)  # [1, n_edit, T, V]

    sampling_cfg = config["sampling"]
    z_sampled = sample_tokens(
        logits,
        temperature=sampling_cfg["temperature"],
        top_p=sampling_cfg["top_p"],
    )  # [1, n_edit, T]

    edit_codebooks = config["model"]["edit_codebooks"]
    z_out = apply_mask(z_batch, z_sampled, mask, edit_codebooks)  # [1, nq, T]

    return z_out.squeeze(0)  # [nq, T]


@torch.no_grad()
def generate_k_candidates(
    model: VariationTransformer,
    z_in: torch.Tensor,
    k: int,
    config: dict,
) -> list[torch.Tensor]:
    """Generate k candidate variations with independent masks and sampling noise.

    Args:
        model: Trained VariationTransformer.
        z_in: Input codegram [nq, T] long.
        k: Number of candidates to generate.
        config: Full config dict.

    Returns:
        List of k codegrams, each [nq, T] long.
    """
    candidates = []
    for _ in range(k):
        z_out = generate_variation(model, z_in, config)
        candidates.append(z_out)
    return candidates
