from __future__ import annotations

import pytest
import torch

from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.train_regret_start_az import (
    VisitDataset,
    anchored_visit_loss,
    collate_visits,
    configure_trainable,
    sample_move,
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


def test_sample_move_argmax_when_temperature_zero() -> None:
    import chess
    import random

    moves = {
        chess.Move.from_uci("e2e4"): 3,
        chess.Move.from_uci("d2d4"): 7,
    }

    assert sample_move(moves, random.Random(0), 0.0) == chess.Move.from_uci("d2d4")


def test_visit_dataset_builds_dense_visit_target() -> None:
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "visits": {"e7e5": 6, "c7c5": 2},
        "value": 0.25,
    }
    dataset = VisitDataset([row])
    batch = collate_visits([dataset[0]])

    assert batch["piece_idx"].shape == (1, 1, 64)
    assert batch["aux"].shape == (1, 1, 7)
    assert batch["target_policy"].sum() == pytest.approx(1.0)
    assert batch["value"].item() == pytest.approx(0.25)


def test_configure_trainable_defaults_to_heads_and_norm() -> None:
    model = _tiny_model()
    trainable = configure_trainable(model, unfreeze_last_trunk_blocks=0)
    trainable_ids = {id(parameter) for parameter in trainable}

    for name, parameter in model.named_parameters():
        expected = name.startswith(("policy_head.", "value_head.", "norm."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_anchored_visit_loss_backprops_policy_head() -> None:
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "visits": {"e7e5": 6, "c7c5": 2},
        "value": 0.0,
    }
    model = _tiny_model()
    reference = _tiny_model().eval().requires_grad_(False)
    batch = collate_visits([VisitDataset([row])[0]])

    loss, metrics = anchored_visit_loss(
        model,
        reference,
        batch,
        value_weight=0.0,
        anchor_weight=0.75,
    )
    loss.backward()

    assert loss.isfinite()
    assert set(metrics) == {"loss", "policy_loss", "value_loss", "anchor_kl"}
    assert model.policy_head.weight.grad is not None
    assert torch.isfinite(model.policy_head.weight.grad).all()
