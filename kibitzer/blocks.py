"""Small transformer and SSM blocks for the hybrid chess model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=0.0, batch_first=True, bias=False
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_mult * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.ffn(self.ffn_norm(x))


class CausalAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=0.0, batch_first=True, bias=False
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_mult * dim)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        h = self.attn_norm(x)
        attn, _ = self.attn(
            h,
            h,
            h,
            attn_mask=self.causal_mask[:seq_len, :seq_len],
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = x + attn
        return x + self.ffn(self.ffn_norm(x))


class SelectiveSSMBlock(nn.Module):
    """A compact Mamba-like recurrent block implemented without extra deps.

    This is not a drop-in Mamba kernel. It keeps the useful shape for this repo:
    causal scan, input-dependent gates, depthwise local mixing, and a cheap state
    update that can be trained with ordinary PyTorch.
    """

    def __init__(self, dim: int, state_dim: int = 16, conv_kernel: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=dim,
            bias=True,
        )
        self.decay = nn.Parameter(torch.empty(dim, state_dim))
        self.input_proj = nn.Linear(dim, dim * state_dim, bias=False)
        self.output_proj = nn.Linear(dim, dim * state_dim, bias=False)
        self.skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, 4 * dim)
        nn.init.normal_(self.decay, mean=-3.0, std=0.2)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)[:, : x.size(1)]
        h = F.silu(h)

        batch, seq_len, dim = h.shape
        a = torch.exp(-F.softplus(self.decay)).view(1, dim, self.state_dim)
        b = self.input_proj(h).view(batch, seq_len, dim, self.state_dim)
        c = self.output_proj(h).view(batch, seq_len, dim, self.state_dim)
        state = h.new_zeros(batch, dim, self.state_dim)
        outputs = []
        valid = None if pad_mask is None else (~pad_mask).to(h.dtype).view(batch, seq_len, 1, 1)

        for t in range(seq_len):
            candidate = a * state + b[:, t] * h[:, t].unsqueeze(-1)
            if valid is not None:
                state = candidate * valid[:, t] + state * (1.0 - valid[:, t])
            else:
                state = candidate
            y_t = (state * c[:, t]).sum(dim=-1) / math.sqrt(self.state_dim)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + h * self.skip
        x = residual + self.out_proj(y * F.silu(gate))
        return x + self.ffn(self.ffn_norm(x))
