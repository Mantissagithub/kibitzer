from __future__ import annotations

import copy
from argparse import Namespace

import pytest
import torch

from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.distill_stockfish import (
    calculate_binned_value_metrics,
    calculate_value_metrics,
    collect_balanced_samples,
    configure_trainable_parameters,
    label_cache_signature,
    load_or_create_labels,
    split_train_eval_by_game,
    training_loss,
    validate_args,
    value_repair_rank,
    value_sampling_weights,
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


def test_norm_repair_unfreezes_only_value_head_and_final_norm() -> None:
    model = _tiny_model()
    trainable = configure_trainable_parameters(
        model,
        value_only=True,
        unfreeze_final_norm=True,
    )

    trainable_ids = {id(parameter) for parameter in trainable}
    for name, parameter in model.named_parameters():
        expected = name.startswith(("value_head.", "norm."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_last_block_value_repair_keeps_policy_head_and_early_trunk_frozen() -> None:
    model = Kibitzer(
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
    trainable = configure_trainable_parameters(
        model,
        value_only=True,
        unfreeze_final_norm=True,
        unfreeze_last_trunk_blocks=1,
    )

    trainable_ids = {id(parameter) for parameter in trainable}
    for name, parameter in model.named_parameters():
        expected = name.startswith(("value_head.", "norm.", "trunk.1."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


def test_policy_anchor_kl_detects_shared_norm_drift() -> None:
    model = _tiny_model()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        model.norm.weight[0] = 2.0
    batch = {
        "piece_idx": torch.randint(0, 13, (2, 1, 64)),
        "aux": torch.randn(2, 1, 7),
        "policy_target": torch.randint(0, 4672, (2, 1)),
        "value_target": torch.tensor([[0.5], [-0.5]]),
        "legal_mask": torch.ones(2, 1, 4672, dtype=torch.bool),
    }

    loss, metrics = training_loss(
        model,
        batch,
        value_only=True,
        policy_reference=reference,
        policy_kl_weight=1.0,
    )

    assert metrics["policy_kl"] > 0.0
    assert loss > metrics["value_loss"]


def test_partial_joint_training_unfreezes_only_requested_scope() -> None:
    model = _tiny_model()
    trainable = configure_trainable_parameters(
        model,
        value_only=False,
        unfreeze_last_trunk_blocks=1,
    )

    trainable_ids = {id(parameter) for parameter in trainable}
    assert trainable_ids
    for name, parameter in model.named_parameters():
        expected = name.startswith(("trunk.0.", "norm.", "policy_head.", "value_head."))
        assert parameter.requires_grad == expected
        assert (id(parameter) in trainable_ids) == expected


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


def test_value_repair_metrics_and_rank_prioritize_decisive_sign() -> None:
    targets = torch.tensor([0.02, 0.10, 0.30, -0.80])
    predictions = torch.tensor([0.01, 0.08, -0.10, -0.70])
    metrics = calculate_value_metrics(predictions, targets)
    metrics.update(calculate_binned_value_metrics(predictions, targets))
    improved = dict(metrics)
    improved["decisive_sign_accuracy"] = 1.0

    assert metrics["decisive_count"] == 1
    assert metrics["won_sign_accuracy"] == 1.0
    assert value_repair_rank(improved) > value_repair_rank(metrics)


def test_inverse_frequency_sampling_is_capped() -> None:
    labels = [({}, 0.0)] * 16 + [({}, 0.1)] * 4 + [({}, 0.3)] * 2 + [({}, 0.8)]
    weights, counts = value_sampling_weights(labels, alpha=1.0, max_weight=4.0)

    assert counts == {"quiet": 16, "edge": 4, "decisive": 2, "won": 1}
    assert weights[0] == 1.0
    assert weights[-1] == 4.0
    assert float(weights.max()) == 4.0


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


def test_label_cache_reuses_matching_teacher_labels(tmp_path) -> None:
    pgn = tmp_path / "games.pgn"
    pgn.write_text(
        """
[Event "cache"]
[Result "1-0"]

1. e4 e5 1-0
""".strip(),
        encoding="utf-8",
    )
    samples = [PositionSample("fen", "e2e4", 1.0, game_id=1)]
    labels = [({1: 0.4, 2: 0.2}, 0.4)]
    signature = label_cache_signature(
        [pgn],
        depth=14,
        multipv=8,
        max_games=None,
        max_positions=1,
        seed=42,
    )
    cache = tmp_path / "labels.pt"
    torch.save({"signature": signature, "samples": samples, "labels": labels}, cache)

    loaded_samples, loaded_labels, cache_hit = load_or_create_labels(
        [pgn],
        cache_path=cache,
        stockfish_path="unused",
        workers=1,
        depth=14,
        multipv=8,
        max_games=None,
        max_positions=1,
        seed=42,
    )

    assert cache_hit
    assert loaded_samples == samples
    assert loaded_labels == labels
