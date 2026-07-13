from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from kibitzer.model import Kibitzer, KibitzerConfig
from kibitzer.oracle_rl import (
    add_returns_and_advantages,
    clipped_process_reward,
    configure_policy_scope,
    filter_training_records,
    oracle_policy_loss,
    signed_outcome,
)
from scripts import train_oracle_process_rl
from scripts.train_oracle_process_rl import split_by_group


def test_signed_outcome_uses_terminal_minus_one_zero_plus_one() -> None:
    assert signed_outcome(1.0) == 1.0
    assert signed_outcome(0.5) == 0.0
    assert signed_outcome(0.0) == -1.0


def test_process_reward_is_mover_relative_and_clipped() -> None:
    assert clipped_process_reward(0.4, 0.3, 0.5) == pytest.approx(-0.2)
    assert clipped_process_reward(0.2, 0.3, 0.5) == pytest.approx(0.2)
    assert clipped_process_reward(1.0, -1.0, 0.5) == -1.0


def test_returns_mix_process_credit_with_terminal_outcome() -> None:
    records = [
        {"game_id": 0, "group_id": 5, "model_ply": 0, "process_reward": 0.0, "terminal_reward": 1.0},
        {"game_id": 0, "group_id": 5, "model_ply": 1, "process_reward": -0.2, "terminal_reward": 1.0},
        {"game_id": 1, "group_id": 5, "model_ply": 0, "process_reward": -1.0, "terminal_reward": -1.0},
        {"game_id": 1, "group_id": 5, "model_ply": 1, "process_reward": -0.8, "terminal_reward": -1.0},
    ]
    shaped = add_returns_and_advantages(
        records,
        gamma=1.0,
        process_weight=0.25,
        terminal_weight=1.0,
    )
    by_game_ply = {(record["game_id"], record["model_ply"]): record for record in shaped}
    assert by_game_ply[(0, 1)]["return"] == 0.95
    assert by_game_ply[(1, 1)]["return"] == -1.2
    assert by_game_ply[(0, 0)]["advantage"] > 0.0
    assert by_game_ply[(1, 0)]["advantage"] < 0.0


def test_filter_requires_regret_and_nontrivial_signed_advantage() -> None:
    records = [
        {"regret": 0.2, "advantage": 0.8},
        {"regret": 0.2, "advantage": -0.7},
        {"regret": 0.01, "advantage": 1.0},
        {"regret": 0.5, "advantage": 0.01},
    ]
    kept = filter_training_records(records, min_regret=0.05, min_abs_advantage=0.1)
    assert kept == records[:2]


def test_policy_scope_freezes_trunk_and_value_head() -> None:
    model = Kibitzer(
        KibitzerConfig(
            d_model=32,
            n_heads=4,
            encoder_layers=1,
            encoder_heads=4,
            trunk_layers=1,
            attention_every=1,
        )
    )
    trainable = configure_policy_scope(model)
    trainable_ids = {id(parameter) for parameter in trainable}
    assert all(id(parameter) in trainable_ids for parameter in model.policy_head.parameters())
    assert all(id(parameter) in trainable_ids for parameter in model.norm.parameters())
    assert all(not parameter.requires_grad for parameter in model.value_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.trunk.parameters())


def test_oracle_gradient_rewards_good_sample_and_suppresses_bad_sample() -> None:
    logits = torch.zeros(2, 2, requires_grad=True)
    base_logits = torch.zeros(2, 2)
    legal = torch.ones(2, 2, dtype=torch.bool)
    rollout_policy = torch.full((2, 2), 0.5)
    action = torch.tensor([0, 1])
    advantage = torch.tensor([1.0, -1.0])
    loss, metrics = oracle_policy_loss(
        logits,
        base_logits,
        legal,
        rollout_policy,
        action,
        advantage,
        delta=0.2,
        beta=0.1,
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[1, 1] > 0.0
    assert metrics["keep_rate"].item() == 1.0


def test_group_split_keeps_rollout_groups_disjoint() -> None:
    records = [
        {"group_id": group_id, "row": row}
        for group_id in range(4)
        for row in range(3)
    ]
    train, evaluate = split_by_group(records, eval_fraction=0.25, seed=7)
    train_groups = {record["group_id"] for record in train}
    eval_groups = {record["group_id"] for record in evaluate}
    assert train_groups.isdisjoint(eval_groups)
    assert train_groups | eval_groups == {0, 1, 2, 3}


def test_run_resume_hands_existing_rollout_to_label_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    out_dir.mkdir()
    (out_dir / "on_policy_raw.jsonl").write_text("{}\n", encoding="utf-8")
    calls: dict[str, argparse.Namespace] = {}
    monkeypatch.setattr(train_oracle_process_rl, "command_generate", lambda args: pytest.fail("regenerated rollout"))
    monkeypatch.setattr(train_oracle_process_rl, "command_label", lambda args: calls.setdefault("label", args))
    monkeypatch.setattr(train_oracle_process_rl, "command_train", lambda args: calls.setdefault("train", args))
    args = argparse.Namespace(
        checkpoint=tmp_path / "base.pt",
        stockfish="stockfish",
        opponent_elo=2300,
        groups=8,
        group_size=4,
        sims=512,
        teacher_nodes=10_000,
        multipv=4,
        process_weight=0.25,
        terminal_weight=1.0,
        gamma=0.99,
        min_regret=0.05,
        min_abs_advantage=0.1,
        beta=0.1,
        lr=1e-5,
        epochs=2,
        workers=4,
        eval_fraction=0.2,
        batch_size=128,
        weight_decay=0.01,
        delta=0.1,
        max_tv_base=0.08,
        seed=31,
        device="cpu",
        out_dir=out_dir,
        report_dir=report_dir,
        reuse_rollout=True,
    )
    train_oracle_process_rl.command_run(args)
    assert calls["label"].input_jsonl == out_dir / "on_policy_raw.jsonl"
    assert calls["label"].workers == 4
    assert calls["train"].data == out_dir / "oracle_labeled.jsonl"
