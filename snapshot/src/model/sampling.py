"""Token sampling and mask application for inference."""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Sample tokens from logits with temperature and nucleus (top-p) filtering.

    Args:
        logits: Model output [B, n_edit, T, codebook_size] float32.
        temperature: Sampling temperature. 0 = greedy argmax.
        top_p: Nucleus sampling threshold (1.0 = no filtering).

    Returns:
        Sampled token indices [B, n_edit, T] long, values 0..codebook_size-1.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    # Temperature scaling
    scaled_logits = logits / temperature

    # Reshape for efficient processing: merge B, n_edit, T
    orig_shape = scaled_logits.shape[:-1]  # [B, n_edit, T]
    flat_logits = scaled_logits.reshape(-1, scaled_logits.shape[-1])  # [N, V]

    if top_p < 1.0:
        flat_logits = _nucleus_filter(flat_logits, top_p)

    probs = F.softmax(flat_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [N]

    return sampled.reshape(orig_shape).long()


def _nucleus_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Sorts probabilities, computes cumulative sum, and masks out tokens
    below the top-p cumulative threshold.

    Args:
        logits: [N, V] float32.
        top_p: Cumulative probability threshold.

    Returns:
        Filtered logits [N, V] with excluded positions set to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative probability above threshold
    # Shift right so the token that crosses threshold is kept
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    output = torch.full_like(logits, float("-inf"))
    output.scatter_(1, sorted_indices, sorted_logits)

    return output


def apply_mask(
    z_in: torch.Tensor,
    z_sampled: torch.Tensor,
    mask: torch.Tensor,
    edit_codebooks: list[int],
) -> torch.Tensor:
    """Apply sampled tokens at masked positions, preserving unmasked.

    Args:
        z_in: Original codegram [B, nq, T] long.
        z_sampled: Sampled tokens [B, n_edit, T] long.
        mask: Boolean mask [B, n_edit, T].
        edit_codebooks: List of codebook indices corresponding to n_edit dim.

    Returns:
        Output codegram [B, nq, T] long with masked positions replaced.
    """
    z_out = z_in.clone()
    for i, cb in enumerate(edit_codebooks):
        z_out[:, cb, :] = torch.where(mask[:, i, :], z_sampled[:, i, :], z_in[:, cb, :])
    return z_out
