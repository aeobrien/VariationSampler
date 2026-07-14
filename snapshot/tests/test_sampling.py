"""Tests for token sampling and mask application."""

import torch
import pytest

from src.model.sampling import sample_tokens, apply_mask


class TestSampleTokens:

    def test_greedy_is_argmax(self):
        """temperature=0 produces argmax."""
        logits = torch.randn(2, 6, 32, 1024)
        sampled = sample_tokens(logits, temperature=0)
        expected = logits.argmax(dim=-1)
        assert torch.equal(sampled, expected)

    def test_output_shape(self):
        """Output shape matches [B, n_edit, T]."""
        logits = torch.randn(4, 6, 64, 1024)
        sampled = sample_tokens(logits, temperature=0.9, top_p=0.95)
        assert sampled.shape == (4, 6, 64)

    def test_output_dtype(self):
        """Output is long tensor."""
        logits = torch.randn(2, 6, 32, 1024)
        sampled = sample_tokens(logits, temperature=0.9)
        assert sampled.dtype == torch.long

    def test_output_value_range(self):
        """Sampled values are in [0, 1023]."""
        logits = torch.randn(4, 6, 64, 1024)
        sampled = sample_tokens(logits, temperature=0.9, top_p=0.95)
        assert sampled.min() >= 0
        assert sampled.max() <= 1023

    def test_top_p_1_allows_all_tokens(self):
        """top_p=1.0 doesn't filter any tokens."""
        # Uniform logits — all tokens should be reachable
        logits = torch.zeros(2, 3, 16, 1024)
        sampled = sample_tokens(logits, temperature=1.0, top_p=1.0)
        assert sampled.shape == (2, 3, 16)
        assert sampled.min() >= 0
        assert sampled.max() <= 1023

    def test_low_top_p_concentrates_sampling(self):
        """Low top_p should concentrate on high-probability tokens."""
        # Create logits with one dominant token
        logits = torch.full((1, 1, 100, 1024), -10.0)
        logits[:, :, :, 42] = 10.0  # token 42 is dominant
        sampled = sample_tokens(logits, temperature=0.5, top_p=0.1)
        # Most samples should be token 42
        assert (sampled == 42).float().mean() > 0.9


class TestApplyMask:

    def test_unmasked_preserved(self):
        """Unmasked positions remain identical to input."""
        b, nq, t = 2, 9, 32
        edit_codebooks = [3, 4, 5, 6, 7, 8]
        n_edit = len(edit_codebooks)

        z_in = torch.randint(0, 1024, (b, nq, t))
        z_sampled = torch.randint(0, 1024, (b, n_edit, t))
        mask = torch.zeros(b, n_edit, t, dtype=torch.bool)

        z_out = apply_mask(z_in, z_sampled, mask, edit_codebooks)
        assert torch.equal(z_out, z_in)

    def test_masked_positions_replaced(self):
        """Masked positions get replaced with sampled tokens."""
        b, nq, t = 1, 9, 16
        edit_codebooks = [7, 8]
        n_edit = 2

        z_in = torch.zeros(b, nq, t, dtype=torch.long)
        z_sampled = torch.ones(b, n_edit, t, dtype=torch.long) * 999
        mask = torch.ones(b, n_edit, t, dtype=torch.bool)

        z_out = apply_mask(z_in, z_sampled, mask, edit_codebooks)
        assert (z_out[:, 7, :] == 999).all()
        assert (z_out[:, 8, :] == 999).all()

    def test_non_edit_codebooks_unchanged(self):
        """Codebooks not in edit_codebooks are never modified."""
        b, nq, t = 2, 9, 32
        edit_codebooks = [6, 7, 8]
        n_edit = 3

        z_in = torch.randint(0, 1024, (b, nq, t))
        z_sampled = torch.randint(0, 1024, (b, n_edit, t))
        mask = torch.ones(b, n_edit, t, dtype=torch.bool)

        z_out = apply_mask(z_in, z_sampled, mask, edit_codebooks)
        for cb in range(nq):
            if cb not in edit_codebooks:
                assert torch.equal(z_out[:, cb, :], z_in[:, cb, :])

    def test_partial_mask(self):
        """Only masked positions in edit codebooks change."""
        b, nq, t = 1, 9, 8
        edit_codebooks = [8]
        n_edit = 1

        z_in = torch.zeros(b, nq, t, dtype=torch.long)
        z_sampled = torch.ones(b, n_edit, t, dtype=torch.long) * 500
        mask = torch.zeros(b, n_edit, t, dtype=torch.bool)
        mask[0, 0, 3] = True
        mask[0, 0, 7] = True

        z_out = apply_mask(z_in, z_sampled, mask, edit_codebooks)
        assert z_out[0, 8, 3] == 500
        assert z_out[0, 8, 7] == 500
        assert z_out[0, 8, 0] == 0
        assert z_out[0, 8, 4] == 0
