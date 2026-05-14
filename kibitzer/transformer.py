"""reusable llama-style transformer blocks.

components, in order:

* :class:`rmsnorm` — root-mean-square layer norm.
* :func:`precompute_freqs_cis` / :func:`apply_rope` — rotary positional
  embeddings on q and k.
* :class:`swiglu` — gated mlp with silu activation.
* :class:`causalselfattention` — causal multi-head attention via
  ``torch.nn.functional.scaled_dot_product_attention``; auto-selects
  flashattention on supported gpus.
* :class:`transformerblock` — pre-norm causal block (used by the autoregressive
  head).
* :class:`encoderblock` — pre-norm bidirectional block (used by the position
  encoder over the 64 chess squares).

all linear layers use ``bias=false``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """root-mean-square layer norm with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = (x.float() * torch.rsqrt(var + self.eps)).type_as(x)
        return x_normed * self.weight


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """precompute complex rope frequencies for ``max_seq_len`` positions."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    )
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """apply rotary positional embeddings to a q or k tensor.

    adjacent feature pairs become complex numbers, get multiplied by
    ``freqs_cis``, and are converted back to the input dtype.
    """
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_out.type_as(x)


class SwiGLU(nn.Module):
    """gated mlp: ``w2(silu(w1 x) * w3 x)``."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _swiglu_hidden(dim: int, multiple_of: int = 64) -> int:
    """standard llama sizing: ``round(8/3 * dim)`` rounded up to ``multiple_of``."""
    h = int(2 * 4 * dim / 3)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


class CausalSelfAttention(nn.Module):
    """causal multi-head self-attention with rope on q/k."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # freqs_cis must stay complex64. registered buffers get downcast by
        # `model.to(torch.bfloat16)`, which strips the imaginary part.
        self._freqs_cis: torch.Tensor | None = None

    def _get_freqs_cis(self, device: torch.device) -> torch.Tensor:
        if self._freqs_cis is None or self._freqs_cis.device != device:
            self._freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_seq_len
            ).to(device)
        return self._freqs_cis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        freqs = self._get_freqs_cis(x.device)[:S]
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class _BidirectionalSelfAttention(nn.Module):
    """bidirectional multi-head self-attention, no rope or causal mask."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """pre-norm causal transformer block."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLU(dim, _swiglu_hidden(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class EncoderBlock(nn.Module):
    """pre-norm bidirectional encoder block."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = _BidirectionalSelfAttention(dim, n_heads)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLU(dim, _swiglu_hidden(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x
