"""Transformer + lightweight SSM hybrid chess model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kibitzer.blocks import CausalAttentionBlock, RMSNorm, SelectiveSSMBlock
from kibitzer.encoding import ACTION_SIZE, AUX_SIZE, NUM_PIECE_TOKENS
from kibitzer.position_encoder import PositionEncoder


@dataclass
class KibitzerConfig:
    d_model: int = 320
    n_heads: int = 8
    max_seq_len: int = 128
    encoder_layers: int = 3
    encoder_heads: int = 8
    trunk_layers: int = 10
    attention_every: int = 3
    ssm_state_dim: int = 8
    n_moves: int = ACTION_SIZE
    n_aux: int = AUX_SIZE
    n_squares: int = 64
    vocab_pieces: int = NUM_PIECE_TOKENS


class Kibitzer(nn.Module):
    """Position encoder plus alternating causal attention and SSM trunk."""

    def __init__(self, config: KibitzerConfig | None = None) -> None:
        super().__init__()
        cfg = config if config is not None else KibitzerConfig()
        self.config = cfg
        self.position_encoder = PositionEncoder(
            dim=cfg.d_model,
            n_heads=cfg.encoder_heads,
            n_layers=cfg.encoder_layers,
            n_aux=cfg.n_aux,
        )
        blocks: list[nn.Module] = []
        for i in range(cfg.trunk_layers):
            if i % cfg.attention_every == 0:
                blocks.append(CausalAttentionBlock(cfg.d_model, cfg.n_heads, cfg.max_seq_len))
            else:
                blocks.append(SelectiveSSMBlock(cfg.d_model, cfg.ssm_state_dim))
        self.trunk = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model)
        self.policy_head = nn.Linear(cfg.d_model, cfg.n_moves)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(
        self,
        piece_idx: torch.Tensor,
        aux: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = piece_idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}")
        if pad_mask is None:
            pad_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=piece_idx.device)

        flat_piece = piece_idx.reshape(batch * seq_len, self.config.n_squares)
        flat_aux = aux.reshape(batch * seq_len, self.config.n_aux)
        h = self.position_encoder(flat_piece, flat_aux)
        h = h.reshape(batch, seq_len, self.config.d_model)

        for block in self.trunk:
            h = block(h, pad_mask)
        h = self.norm(h)

        policy = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy, value

    def loss(
        self,
        piece_idx: torch.Tensor,
        aux: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        value_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, value = self(piece_idx, aux, pad_mask)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        active_shape = policy_target.shape[:2]
        active = torch.ones(active_shape, dtype=torch.bool, device=policy_target.device)
        if pad_mask is not None:
            active = ~pad_mask

        flat_logits = logits[active]
        flat_policy = policy_target[active]
        flat_value = value.squeeze(-1)[active]
        flat_value_target = value_target[active]

        if flat_policy.dtype == torch.long:
            policy_loss = F.cross_entropy(flat_logits, flat_policy)
        else:
            logp = F.log_softmax(flat_logits, dim=-1)
            policy_loss = -(flat_policy * logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(flat_value, flat_value_target)
        loss = policy_loss + value_weight * value_loss
        metrics = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
        }
        return loss, metrics

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
