from __future__ import annotations

import torch

from kibitzer.transformer import (
    CausalSelfAttention,
    EncoderBlock,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rope,
    precompute_freqs_cis,
)

DIM = 64
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS  # 16
MAX_SEQ_LEN = 128
B = 2
S = 32


def test_rmsnorm_shape() -> None:
    norm = RMSNorm(DIM)
    x = torch.randn(B, S, DIM)
    y = norm(x)
    assert y.shape == (B, S, DIM)
    assert y.dtype == x.dtype


def test_precompute_freqs_cis_shape() -> None:
    freqs = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN)
    assert freqs.shape == (MAX_SEQ_LEN, HEAD_DIM // 2)
    assert freqs.dtype == torch.complex64


def test_apply_rope_shape() -> None:
    freqs = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN)[:S]
    x = torch.randn(B, N_HEADS, S, HEAD_DIM)
    y = apply_rope(x, freqs)
    assert y.shape == (B, N_HEADS, S, HEAD_DIM)
    assert y.dtype == x.dtype


def test_swiglu_shape() -> None:
    mlp = SwiGLU(DIM, hidden_dim=192)
    x = torch.randn(B, S, DIM)
    y = mlp(x)
    assert y.shape == (B, S, DIM)


def test_causal_self_attention_shape() -> None:
    attn = CausalSelfAttention(DIM, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(B, S, DIM)
    y = attn(x)
    assert y.shape == (B, S, DIM)

    # shorter sequence exercises freqs_cis[:s] slicing.
    x_short = torch.randn(B, 16, DIM)
    y_short = attn(x_short)
    assert y_short.shape == (B, 16, DIM)


def test_transformer_block_shape() -> None:
    block = TransformerBlock(DIM, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(B, S, DIM)
    y = block(x)
    assert y.shape == (B, S, DIM)


def test_encoder_block_shape() -> None:
    block = EncoderBlock(DIM, N_HEADS)
    x = torch.randn(B, 64, DIM)  # the actual chess case: 64 squares
    y = block(x)
    assert y.shape == (B, 64, DIM)
