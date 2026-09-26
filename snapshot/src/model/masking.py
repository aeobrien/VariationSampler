"""Mask generation for masked-token micro-inpainting."""

import logging

import torch

logger = logging.getLogger(__name__)


def build_mask(
    batch_size: int,
    t_max: int,
    config: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a random mask for editable codebooks.

    Per-codebook probability follows a gradient from config: each codebook
    has a base probability (p_tail or p_attack) scaled by its multiplier.
    Attack frames (first few) use a lower base probability.

    Args:
        batch_size: Batch dimension.
        t_max: Number of temporal frames.
        config: Full config dict with 'masking' and 'model' sections.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Boolean mask [batch_size, n_edit, t_max] — True at positions to mask.
    """
    masking_cfg = config["masking"]
    edit_codebooks = config["model"]["edit_codebooks"]
    n_edit = len(edit_codebooks)

    p_tail = masking_cfg["p_tail"]
    p_attack = masking_cfg["p_attack"]
    attack_frames = masking_cfg["attack_frames"]
    multipliers = masking_cfg["codebook_multipliers"]

    # Build probability tensor [n_edit, t_max]
    probs = torch.zeros(n_edit, t_max)
    for i, cb in enumerate(edit_codebooks):
        mult = multipliers[cb]
        probs[i, :attack_frames] = p_attack * mult
        probs[i, attack_frames:] = p_tail * mult

    # Expand for batch [B, n_edit, t_max]
    probs = probs.unsqueeze(0).expand(batch_size, -1, -1)

    # Bernoulli sampling
    mask = torch.bernoulli(probs, generator=generator).bool()

    return mask
