"""Encode one chess position into a dense vector."""

from __future__ import annotations

import torch
import torch.nn as nn

from kibitzer.blocks import EncoderBlock, RelativeEncoderBlock, RMSNorm


class PositionEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_layers: int,
        n_aux: int,
        pos_encoding: str = "absolute",
    ) -> None:
        super().__init__()
        if pos_encoding not in ("absolute", "shaw"):
            raise ValueError(f"unknown pos_encoding: {pos_encoding!r}")
        self.pos_encoding = pos_encoding
        self.piece_emb = nn.Embedding(13, dim)
        self.aux_proj = nn.Linear(n_aux, dim, bias=False)
        if pos_encoding == "absolute":
            self.square_emb = nn.Embedding(64, dim)
            self.blocks = nn.ModuleList([EncoderBlock(dim, n_heads) for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList(
                [RelativeEncoderBlock(dim, n_heads) for _ in range(n_layers)]
            )
        self.norm = RMSNorm(dim)

    def forward(self, piece_idx: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        x = self.piece_emb(piece_idx)
        if self.pos_encoding == "absolute":
            squares = torch.arange(64, device=piece_idx.device)
            x = x + self.square_emb(squares)
        x = x + self.aux_proj(aux).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)
