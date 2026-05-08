"""Tests for kibitzer.model.Kibitzer."""

from __future__ import annotations

import torch

from kibitzer.model import Kibitzer, KibitzerConfig

B = 2
T = 10


def _random_inputs(batch: int = B, seq: int = T) -> tuple[torch.Tensor, ...]:
    piece_idx = torch.randint(0, 13, (batch, seq, 64), dtype=torch.long)
    aux = torch.randn(batch, seq, 7)
    pad_mask = torch.zeros(batch, seq, dtype=torch.bool)
    return piece_idx, aux, pad_mask


def test_forward_shapes() -> None:
    model = Kibitzer()
    piece_idx, aux, pad_mask = _random_inputs()
    policy, value = model(piece_idx, aux, pad_mask)
    assert policy.shape == (B, T, 4672)
    assert value.shape == (B, T, 1)


def test_value_range() -> None:
    model = Kibitzer()
    piece_idx, aux, pad_mask = _random_inputs()
    _, value = model(piece_idx, aux, pad_mask)
    assert value.max().item() <= 1.0
    assert value.min().item() >= -1.0


def test_no_nans() -> None:
    model = Kibitzer()
    piece_idx, aux, pad_mask = _random_inputs()
    policy, value = model(piece_idx, aux, pad_mask)
    assert not torch.isnan(policy).any()
    assert not torch.isinf(policy).any()
    assert not torch.isnan(value).any()
    assert not torch.isinf(value).any()


def test_param_count() -> None:
    model = Kibitzer()
    n = model.num_params()
    print(f"Kibitzer params: {n:_}")
    # Default spec (d_model=384, n_layers=12) lands ~28.5M; user accepted
    # this and chose a wider sanity bound rather than re-sizing.
    assert 20_000_000 < n < 40_000_000, f"unexpected param count: {n}"


def test_padding_doesnt_break() -> None:
    model = Kibitzer()
    piece_idx, aux, pad_mask = _random_inputs()
    pad_mask[:, 5:] = True
    piece_idx[:, 5:, :] = 0
    aux[:, 5:, :] = 0.0
    policy, value = model(piece_idx, aux, pad_mask)
    assert policy.shape == (B, T, 4672)
    assert value.shape == (B, T, 1)


def test_config_override() -> None:
    cfg = KibitzerConfig(d_model=64, n_layers=2, n_heads=4, encoder_layers=1, encoder_heads=4)
    small = Kibitzer(cfg)
    piece_idx, aux, pad_mask = _random_inputs()
    policy, value = small(piece_idx, aux, pad_mask)
    assert policy.shape == (B, T, 4672)
    assert value.shape == (B, T, 1)
