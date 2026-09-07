"""Masked-token micro-inpainting transformer for variation generation."""

import logging
import math

import torch
import torch.nn as nn

from src.utils.audio import NQ, CODEBOOK_SIZE, T_MAX

logger = logging.getLogger(__name__)


class VariationTransformer(nn.Module):
    """Bidirectional transformer that predicts replacement tokens for masked
    positions in editable codebooks.

    All 9 codebooks provide input context (summed embeddings). Only the
    editable subset (default: codebooks 3-8) gets prediction heads.

    Args:
        d_model: Hidden dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        nq: Total number of codebooks.
        codebook_size: Vocabulary size per codebook.
        t_max: Maximum temporal frames.
        edit_codebooks: List of codebook indices that are editable.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        nq: int = NQ,
        codebook_size: int = CODEBOOK_SIZE,
        t_max: int = T_MAX,
        edit_codebooks: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nq = nq
        self.codebook_size = codebook_size
        self.t_max = t_max
        self.edit_codebooks = edit_codebooks or [3, 4, 5, 6, 7, 8]
        self.n_edit = len(self.edit_codebooks)

        # Per-codebook embedding tables
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(nq)
        ])

        # Learned mask embedding — added at frames where any editable codebook is masked
        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        # Learned positional embedding
        self.pos_embedding = nn.Embedding(t_max, d_model)

        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Per-editable-codebook projection heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(self.n_edit)
        ])

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "VariationTransformer: d_model=%d, layers=%d, heads=%d, "
            "edit_codebooks=%s, params=%d",
            d_model, n_layers, n_heads, self.edit_codebooks, n_params,
        )

    def _init_weights(self) -> None:
        """Xavier uniform init for embeddings and linear layers."""
        for emb in self.codebook_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        z_in: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z_in: Input codegram [B, nq, T] long, values 0..codebook_size-1.
            mask: Boolean mask [B, n_edit, T] — True at positions to predict.

        Returns:
            Logits [B, n_edit, T, codebook_size] float32.
        """
        b, nq, t = z_in.shape

        # Sum embeddings from all codebooks to get per-frame representation
        h = torch.zeros(b, t, self.d_model, device=z_in.device, dtype=torch.float32)
        for i in range(nq):
            h = h + self.codebook_embeddings[i](z_in[:, i, :])  # [B, T, d_model]

        # Add positional embeddings
        positions = torch.arange(t, device=z_in.device)
        h = h + self.pos_embedding(positions).unsqueeze(0)  # broadcast over batch

        # Add mask embedding at frames where any editable codebook is masked
        frame_masked = mask.any(dim=1)  # [B, T]
        h = h + self.mask_embedding * frame_masked.unsqueeze(-1).float()

        # Transformer encoder
        h = self.transformer(h)  # [B, T, d_model]

        # Project to per-codebook logits
        logits = torch.stack(
            [head(h) for head in self.heads], dim=1
        )  # [B, n_edit, T, codebook_size]

        return logits

    @classmethod
    def from_config(cls, config: dict) -> "VariationTransformer":
        """Construct from a config dict (model section)."""
        mc = config["model"]
        return cls(
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            dropout=mc["dropout"],
            nq=mc["nq"],
            codebook_size=mc["codebook_size"],
            t_max=mc["t_max"],
            edit_codebooks=mc["edit_codebooks"],
        )
