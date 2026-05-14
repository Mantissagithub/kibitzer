"""full kibitzer model: position encoder, causal trunk, policy/value heads.

composes a per-position encoder over 64 squares with causal transformer blocks
over the game timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kibitzer.position_encoder import PositionEncoder
from kibitzer.transformer import RMSNorm, TransformerBlock


@dataclass
class KibitzerConfig:
    """model hyperparameters."""

    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 8
    max_seq_len: int = 256
    encoder_layers: int = 3
    encoder_heads: int = 8
    vocab_pieces: int = 13
    n_squares: int = 64
    n_moves: int = 4672
    n_aux: int = 7


class Kibitzer(nn.Module):
    """position encoder plus causal trunk."""

    def __init__(self, config: KibitzerConfig | None = None) -> None:
        super().__init__()
        cfg = config if config is not None else KibitzerConfig()
        self.config = cfg

        self.position_encoder = PositionEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.encoder_heads,
            n_layers=cfg.encoder_layers,
        )

        self.trunk_blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.max_seq_len)
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.d_model)

        self.policy_head = nn.Linear(cfg.d_model, cfg.n_moves)

        self.value_head_1 = nn.Linear(cfg.d_model, cfg.d_model // 2)
        self.value_head_2 = nn.Linear(cfg.d_model // 2, 1)

    def forward(
        self,
        piece_idx: torch.Tensor,
        aux: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pad_mask  # part of the contract; consumed by the loss, not here

        B, T, _ = piece_idx.shape
        flat_p = piece_idx.reshape(B * T, self.config.n_squares)
        flat_a = aux.reshape(B * T, self.config.n_aux)
        h = self.position_encoder(flat_p, flat_a)        # (B*T, d_model)
        h = h.reshape(B, T, self.config.d_model)

        for block in self.trunk_blocks:
            h = block(h)
        h = self.final_norm(h)

        policy_logits = self.policy_head(h)               # (B, T, n_moves)

        v = F.gelu(self.value_head_1(h))
        value_pred = torch.tanh(self.value_head_2(v))     # (B, T, 1)

        return policy_logits, value_pred

    def num_params(self) -> int:
        """total parameter count."""
        return sum(p.numel() for p in self.parameters())
