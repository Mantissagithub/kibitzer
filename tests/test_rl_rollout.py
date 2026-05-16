from __future__ import annotations

import random

import torch

from kibitzer.rl_rollout import RolloutStep, pack_rollout_batch, sample_prev_checkpoint


def test_pack_rollout_batch_pads_to_chunk_len() -> None:
    step = RolloutStep(
        piece_idx=torch.zeros(64, dtype=torch.long),
        aux=torch.zeros(7),
        action=12,
        legal_mask=torch.zeros(4672, dtype=torch.bool),
        old_log_prob=-0.5,
        value_pred=0.1,
        reward=1.0,
        done=True,
        color=True,
        source="stockfish",
        opponent_label="stockfish-1320",
        stockfish_before=0.0,
        stockfish_after=0.1,
        value_before=0.0,
        value_after=0.1,
    )
    batches = pack_rollout_batch([step], chunk_len=4, source="stockfish")
    assert len(batches) == 1
    batch = batches[0]
    assert batch.piece_idx.shape == (1, 4, 64)
    assert batch.aux.shape == (1, 4, 7)
    assert batch.actions[0, 0].item() == 12
    assert batch.valid_mask[0, 0].item() is True
    assert batch.valid_mask[0, 1].item() is False
    assert batch.opponent_label == "stockfish-1320"


def test_pack_rollout_batch_splits_long_trajectory() -> None:
    steps = []
    for idx in range(5):
        steps.append(
            RolloutStep(
                piece_idx=torch.full((64,), idx, dtype=torch.long),
                aux=torch.zeros(7),
                action=idx,
                legal_mask=torch.zeros(4672, dtype=torch.bool),
                old_log_prob=-0.5,
                value_pred=0.1,
                reward=1.0,
                done=idx == 4,
                color=True,
                source="selfplay",
                opponent_label="prev",
                stockfish_before=0.0,
                stockfish_after=0.1,
                value_before=0.0,
                value_after=0.1,
                has_search_target=idx == 1,
                search_action=99,
                search_value_target=0.3,
            )
        )
    batches = pack_rollout_batch(steps, chunk_len=3, source="selfplay")
    assert len(batches) == 2
    assert batches[0].valid_mask.sum().item() == 3
    assert batches[1].valid_mask.sum().item() == 2
    assert batches[0].dones[0, 2].item() is False
    assert batches[1].dones[0, 1].item() is True
    assert batches[0].has_search_target[0, 1].item() is True
    assert batches[0].search_actions[0, 1].item() == 99


def test_sample_prev_checkpoint_prefers_weighted_sources() -> None:
    rng = random.Random(0)
    chosen = sample_prev_checkpoint(
        ["old-1", "old-2"],
        rng,
        "fallback",
        latest_checkpoint="latest",
        best_checkpoint="best",
        latest_weight=1.0,
        best_weight=0.0,
        older_weight=0.0,
    )
    assert chosen == "latest"
