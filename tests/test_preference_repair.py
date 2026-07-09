from __future__ import annotations

import pytest
import torch
import chess

from kibitzer.encoding import move_to_index
from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.train_preference_repair import (
    PreferenceDataset,
    choose_preference_pair,
    collate_preferences,
    configure_trainable,
    preference_loss,
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


def test_choose_preference_pair_prefers_policy_mistake() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = chess.Board(fen)
    good = move_to_index(chess.Move.from_uci("e7e5"), board)
    bad = move_to_index(chess.Move.from_uci("b8c6"), board)
    other = move_to_index(chess.Move.from_uci("g8f6"), board)
    action_scores = {good: 0.7, bad: 0.1, other: -0.2}
    policy_moves = [(bad, "b8c6", 0.4), (good, "e7e5", 0.2)]

    pair = choose_preference_pair(
        fen,
        action_scores,
        policy_moves,
        min_margin=0.2,
    )

    assert pair is not None
    assert pair["good_index"] == good
    assert pair["bad_index"] == bad
    assert pair["teacher_margin"] == pytest.approx(0.6)


def test_configure_trainable_policy_only_by_default() -> None:
    model = _tiny_model()
    trainable = configure_trainable(model, unfreeze_last_trunk_blocks=0)
    trainable_ids = {id(parameter) for parameter in trainable}

    for name, parameter in model.named_parameters():
        expected = name.startswith(("policy_head.", "norm."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_preference_dataset_and_loss_backprop_policy_head() -> None:
    record = {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "good_index": 528,
        "bad_index": 530,
        "teacher_margin": 0.6,
        "action_scores": {"528": 0.7, "530": 0.1},
    }
    model = _tiny_model()
    reference = _tiny_model().eval().requires_grad_(False)
    dataset = PreferenceDataset([record], temperature=0.1)
    batch = collate_preferences([dataset[0]])

    loss, metrics = preference_loss(
        model,
        reference,
        batch,
        beta=0.1,
        ce_weight=0.25,
        anchor_weight=0.05,
    )
    loss.backward()

    assert loss.isfinite()
    assert set(metrics) == {"loss", "dpo_loss", "ce_loss", "anchor_kl", "pair_acc", "pair_margin"}
    assert model.policy_head.weight.grad is not None
    assert torch.isfinite(model.policy_head.weight.grad).all()
