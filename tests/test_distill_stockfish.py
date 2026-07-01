from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.distill_stockfish import (
    calculate_value_metrics,
    collect_balanced_samples,
    configure_trainable_parameters,
    split_train_eval_by_game,
    training_loss,
    validate_args,
)
from kibitzer.data import PositionSample


def _tiny_model() -> Kibitzer:
    return Kibitzer(
        KibitzerConfig(
            d_model=32,
            n_heads=4,
            max_seq_len=2,
            encoder_layers=1,
            encoder_heads=4,
            trunk_layers=1,
            attention_every=1,
            ssm_state_dim=4,
        )
    )


def test_value_only_requires_initial_checkpoint() -> None:
    with pytest.raises(SystemExit, match="--value-only requires --init"):
        validate_args(Namespace(value_only=True, init=None))


def test_stockfish_workers_must_be_positive() -> None:
    with pytest.raises(SystemExit, match="--stockfish-workers must be at least 1"):
        validate_args(
            Namespace(
                value_only=False,
                init=None,
                eval_fraction=0.1,
                stockfish_workers=0,
            )
        )


def test_value_only_loss_updates_only_value_head() -> None:
    model = _tiny_model()
    trainable = configure_trainable_parameters(model, value_only=True)
    batch = {
        "piece_idx": torch.randint(0, 13, (2, 1, 64)),
        "aux": torch.randn(2, 1, 7),
        "policy_target": torch.randint(0, 4672, (2, 1)),
        "value_target": torch.tensor([[0.5], [-0.5]]),
        "legal_mask": torch.ones(2, 1, 4672, dtype=torch.bool),
    }

    loss, metrics = training_loss(model, batch, value_only=True)
    loss.backward()

    trainable_ids = {id(parameter) for parameter in trainable}
    assert trainable_ids == {id(parameter) for parameter in model.value_head.parameters()}
    assert set(metrics) == {"loss", "value_loss"}
    for name, parameter in model.named_parameters():
        if name.startswith("value_head."):
            assert parameter.requires_grad
            assert parameter.grad is not None
        else:
            assert not parameter.requires_grad
            assert parameter.grad is None


def test_eval_split_keeps_games_disjoint() -> None:
    samples = [
        PositionSample("fen", "e2e4", 0.0, game_id=game_id)
        for game_id in range(1, 6)
        for _ in range(2)
    ]
    labels = [({}, 0.0) for _ in samples]

    train_samples, _, eval_samples, _ = split_train_eval_by_game(
        samples,
        labels,
        eval_fraction=0.2,
        seed=42,
    )

    train_games = {sample.game_id for sample in train_samples}
    eval_games = {sample.game_id for sample in eval_samples}
    assert train_games.isdisjoint(eval_games)
    assert train_games | eval_games == {1, 2, 3, 4, 5}


def test_value_metrics_for_perfect_predictions() -> None:
    targets = torch.tensor([-0.8, -0.2, 0.3, 0.9])
    metrics = calculate_value_metrics(targets.clone(), targets)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["sign_accuracy"] == 1.0
    assert metrics["r2"] == 1.0


def test_balanced_collection_uses_each_pgn(tmp_path) -> None:
    template = """
[Event "{event}"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
""".strip()
    paths = []
    for index in range(2):
        path = tmp_path / f"games-{index}.pgn"
        path.write_text(template.format(event=index), encoding="utf-8")
        paths.append(path)

    samples = collect_balanced_samples(
        paths,
        max_games=None,
        max_positions=8,
        seed=42,
    )

    assert len(samples) == 8
    assert len({sample.game_id for sample in samples}) == 2
