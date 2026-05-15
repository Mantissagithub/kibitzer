from __future__ import annotations

import torch

from kibitzer.rl_rollout import RolloutStep, pack_rollout_batch


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
        stockfish_before=0.0,
        stockfish_after=0.1,
        value_before=0.0,
        value_after=0.1,
    )
    batch = pack_rollout_batch([step], chunk_len=4, source="stockfish")
    assert batch.piece_idx.shape == (1, 4, 64)
    assert batch.aux.shape == (1, 4, 7)
    assert batch.actions[0, 0].item() == 12
    assert batch.valid_mask[0, 0].item() is True
    assert batch.valid_mask[0, 1].item() is False
