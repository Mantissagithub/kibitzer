"""Full Kibitzer model: per-position encoder + causal trunk + policy/value heads.

Composes :class:`kibitzer.position_encoder.PositionEncoder` (over the 64
squares of one ply) with a stack of causal :class:`kibitzer.transformer.
TransformerBlock` layers (over the time/move axis of a game) and produces both
policy logits and a scalar value prediction at every timestep.
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
    """Hyperparameters for :class:`Kibitzer`."""

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
    """Composite chess model.

    Input is a batch of games as fixed-length sequences:

    * ``piece_idx`` (B, T, 64) long — per-ply board piece-token tensor
    * ``aux``       (B, T,  7) float — per-ply global auxiliary scalars
    * ``pad_mask``  (B, T)     bool  — ``True`` where the ply is padding
      (the game ended before this index)

    Forward returns ``(policy_logits, value_pred)``:

    * ``policy_logits`` (B, T, 4672) — **unmasked** logits over the AlphaZero
      action space. Apply :func:`kibitzer.masking.legal_move_mask` per-ply at
      loss/inference time.
    * ``value_pred``   (B, T, 1)     — scalar in ``[-1, 1]`` (tanh).

    ``pad_mask`` is part of the public contract for callers that already track
    it during batching, but the forward pass does not currently consume it:
    each ply's PositionEncoder output depends only on its own
    ``(piece_idx, aux)``, and the causal trunk only attends backwards. The
    training loop is responsible for excluding padded positions from the
    policy/value losses.
    """

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
        """Total parameter count (sum over ``self.parameters()``)."""
        return sum(p.numel() for p in self.parameters())
