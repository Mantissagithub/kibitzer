from __future__ import annotations

import torch

from kibitzer.model import Kibitzer, KibitzerConfig


def _tiny_model() -> Kibitzer:
    cfg = KibitzerConfig(
        d_model=64,
        n_heads=4,
        max_seq_len=8,
        encoder_layers=1,
        encoder_heads=4,
        trunk_layers=3,
        attention_every=2,
        ssm_state_dim=4,
    )
    return Kibitzer(cfg)


def test_forward_shapes() -> None:
    model = _tiny_model()
    piece_idx = torch.randint(0, 13, (2, 4, 64))
    aux = torch.randn(2, 4, 7)
    policy, value = model(piece_idx, aux)
    assert policy.shape == (2, 4, 4672)
    assert value.shape == (2, 4, 1)
    assert value.min().item() >= -1.0
    assert value.max().item() <= 1.0


def test_hard_label_loss() -> None:
    model = _tiny_model()
    batch = {
        "piece_idx": torch.randint(0, 13, (2, 3, 64)),
        "aux": torch.randn(2, 3, 7),
        "policy_target": torch.randint(0, 4672, (2, 3)),
        "value_target": torch.zeros(2, 3),
        "legal_mask": torch.ones(2, 3, 4672, dtype=torch.bool),
    }
    loss, metrics = model.loss(**batch)
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "policy_loss", "value_loss"}


def test_dense_label_loss() -> None:
    model = _tiny_model()
    target = torch.zeros(2, 3, 4672)
    target[:, :, 10] = 0.7
    target[:, :, 20] = 0.3
    batch = {
        "piece_idx": torch.randint(0, 13, (2, 3, 64)),
        "aux": torch.randn(2, 3, 7),
        "policy_target": target,
        "value_target": torch.zeros(2, 3),
        "legal_mask": torch.ones(2, 3, 4672, dtype=torch.bool),
    }
    loss, _ = model.loss(**batch)
    assert torch.isfinite(loss)


def test_default_model_is_around_30m() -> None:
    model = Kibitzer()
    assert 25_000_000 <= model.num_params() <= 35_000_000
