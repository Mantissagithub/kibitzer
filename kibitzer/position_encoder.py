"""chess position encoder.

architecture: a learned per-square positional embedding plus a per-square piece
embedding, with global aux features projected and broadcast onto every square.
the 64 tokens go through ``n_layers`` of bidirectional pre-norm encoder blocks
(no rope — square ordering is arbitrary), then rmsnorm and mean pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kibitzer.transformer import EncoderBlock, RMSNorm


class PositionEncoder(nn.Module):
    """encode chess positions into ``(b, d_model)`` vectors."""

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.piece_emb = nn.Embedding(13, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        self.aux_proj = nn.Linear(7, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self, piece_idx: torch.Tensor, aux: torch.Tensor
    ) -> torch.Tensor:
        """encode ``piece_idx`` and ``aux`` tensors."""
        squares = torch.arange(64, device=piece_idx.device)
        x = self.piece_emb(piece_idx) + self.square_emb(squares)  # (B, 64, D)
        x = x + self.aux_proj(aux).unsqueeze(1)                    # (B, 64, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x.mean(dim=1)
