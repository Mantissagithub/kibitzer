"""Tests for kibitzer.position_encoder."""

from __future__ import annotations

import torch

from kibitzer.position_encoder import PositionEncoder

D_MODEL = 384
B = 4


def _random_inputs(batch: int = B) -> tuple[torch.Tensor, torch.Tensor]:
    piece_idx = torch.randint(0, 13, (batch, 64), dtype=torch.long)
    aux = torch.randn(batch, 7)
    return piece_idx, aux


def test_output_shape() -> None:
    enc = PositionEncoder()
    piece_idx, aux = _random_inputs()
    h = enc(piece_idx, aux)
    assert h.shape == (B, D_MODEL)


def test_no_nans() -> None:
    enc = PositionEncoder()
    piece_idx, aux = _random_inputs()
    h = enc(piece_idx, aux)
    assert not torch.isnan(h).any(), "output contains NaNs"
    assert not torch.isinf(h).any(), "output contains infs"


def test_param_count() -> None:
    enc = PositionEncoder()
    n = sum(p.numel() for p in enc.parameters())
    print(f"PositionEncoder params: {n:_}")
    # Generous range — catches gross arch mistakes without locking in a tight
    # number. With d_model=384, n_heads=8, n_layers=3 the actual count is
    # ~5.3M (12·D²·n_layers + small).
    assert 1_000_000 < n < 10_000_000, f"unexpected param count: {n}"
