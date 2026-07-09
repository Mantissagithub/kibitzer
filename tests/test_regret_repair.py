from __future__ import annotations

import torch
import pytest

from kibitzer.data import collate_positions
from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.train_regret_repair import (
    RegretDataset,
    anchored_loss,
    configure_trainable,
    labeled_regret,
)


def _tiny_model() -> Kibitzer:
    return Kibitzer(
        KibitzerConfig(
            d_model=32,
            n_heads=4,
            max_seq_len=2,
            encoder_layers=1,
            encoder_heads=4,
            trunk_layers=2,
            attention_every=1,
            ssm_state_dim=4,
        )
    )


def test_labeled_regret_uses_labeled_floor_for_unseen_move() -> None:
    scores = {10: 0.7, 20: 0.2, 30: -0.4}

    assert labeled_regret(scores, 10) == 0.0
    assert labeled_regret(scores, 20) == pytest.approx(0.5)
    assert labeled_regret(scores, 99) == pytest.approx(1.1)


def test_configure_trainable_defaults_to_heads_and_norm() -> None:
    model = _tiny_model()
    trainable = configure_trainable(model, unfreeze_last_trunk_blocks=0)
    trainable_ids = {id(parameter) for parameter in trainable}

    for name, parameter in model.named_parameters():
        expected = name.startswith(("policy_head.", "value_head.", "norm."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_configure_trainable_can_unfreeze_last_trunk_block() -> None:
    model = _tiny_model()
    trainable = configure_trainable(model, unfreeze_last_trunk_blocks=1)
    trainable_ids = {id(parameter) for parameter in trainable}

    for name, parameter in model.named_parameters():
        expected = name.startswith(("policy_head.", "value_head.", "norm.", "trunk.1."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_regret_dataset_and_anchored_loss_are_trainable() -> None:
    record = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "teacher_value": 0.25,
        "action_scores": {"528": 0.25, "530": 0.1},
    }
    model = _tiny_model()
    reference = _tiny_model().eval().requires_grad_(False)
    dataset = RegretDataset([record], temperature=0.1)
    batch = collate_positions([dataset[0]])

    loss, metrics = anchored_loss(
        model,
        reference,
        batch,
        value_weight=0.25,
        anchor_weight=0.5,
    )
    loss.backward()

    assert loss.isfinite()
    assert set(metrics) == {"loss", "policy_loss", "value_loss", "anchor_kl"}
    assert model.policy_head.weight.grad is not None
    assert torch.isfinite(model.policy_head.weight.grad).all()
