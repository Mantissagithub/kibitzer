"""Reusable transformer building blocks (modern, llama-style).

Components, in order:

* :class:`RMSNorm` — root-mean-square layer norm.
* :func:`precompute_freqs_cis` / :func:`apply_rope` — rotary positional
  embeddings on Q and K.
* :class:`SwiGLU` — gated MLP with SiLU activation.
* :class:`CausalSelfAttention` — causal multi-head attention via
  ``torch.nn.functional.scaled_dot_product_attention``; auto-selects
  FlashAttention on supported GPUs.
* :class:`TransformerBlock` — pre-norm causal block (used by the autoregressive
  head).
* :class:`EncoderBlock` — pre-norm bidirectional block (used by the position
  encoder over the 64 chess squares).

All linear layers use ``bias=False``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Normalizes the last dimension of the input by its RMS, then scales by a
    learnable per-channel weight. No bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = (x.float() * torch.rsqrt(var + self.eps)).type_as(x)
        return x_normed * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute the complex-valued RoPE frequencies.

    Parameters
    ----------
    dim : int
        Per-head embedding size (the head dimension of the attention).
    max_seq_len : int
        Maximum sequence length the cache should cover.
    theta : float
        Base frequency. ``10000.0`` matches the original RoPE / llama paper.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``(max_seq_len, dim // 2)``, dtype ``complex64``.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    )
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to a Q or K tensor.

    Uses the adjacent-pair (llama-style) convention: pairs of consecutive
    features ``(x[..., 2i], x[..., 2i+1])`` are treated as a complex number
    and multiplied by ``freqs_cis`` to rotate them.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(B, H, S, head_dim)``. Any floating dtype; output preserves
        the input dtype.
    freqs_cis : torch.Tensor
        Complex tensor of shape ``(S, head_dim // 2)``. Typically a slice of
        the buffer returned by :func:`precompute_freqs_cis`.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as ``x``.
    """
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated MLP with SiLU activation: ``W2(silu(W1 x) * W3 x)``.

    No bias on any projection.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _swiglu_hidden(dim: int, multiple_of: int = 64) -> int:
    """Standard llama sizing: ``round(8/3 * dim)`` rounded up to ``multiple_of``."""
    h = int(2 * 4 * dim / 3)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE on Q/K.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` with
    ``is_causal=True`` so PyTorch can dispatch to the FlashAttention backend
    on supported GPUs.
    """

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

        # freqs_cis must stay complex64. Storing it as a registered buffer makes
        # `model.to(torch.bfloat16)` silently downcast it to bf16 (warning:
        # "Casting complex values to real discards the imaginary part") which
        # zeros out the rotation. Keep it as a plain attribute and lazily
        # (re)build it on the input device on first forward.
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
    """Bidirectional multi-head self-attention, no RoPE, no causal mask.

    Internal sibling of :class:`CausalSelfAttention` used by
    :class:`EncoderBlock`. Position information is expected to be supplied by
    a learned embedding outside this module (e.g. a square embedding for the
    64 chess squares).
    """

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


# ---------------------------------------------------------------------------
# Pre-norm blocks
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm causal transformer block.

    ``y = x + attn(rmsnorm(x)); out = y + swiglu(rmsnorm(y))``.
    """

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
    """Pre-norm bidirectional encoder block (no causal mask, no RoPE).

    Used by the chess-position encoder over the 64 squares; positional info
    comes from a learned square embedding upstream of this block.
    """

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
