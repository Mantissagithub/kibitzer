from __future__ import annotations

from pathlib import Path

import torch

from kibitzer.data import StreamingPositionDataset, dense_policy_from_scores, iter_pgn_samples


def test_iter_pgn_samples(tmp_path: Path) -> None:
    pgn = tmp_path / "mini.pgn"
    pgn.write_text(
        """
[Event "mini"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
""".strip(),
        encoding="utf-8",
    )
    samples = list(iter_pgn_samples([pgn]))
    assert [sample.move_uci for sample in samples] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert {sample.game_id for sample in samples} == {1}
    assert samples[0].value == 1.0
    assert samples[1].value == -1.0


def test_dense_policy_from_scores() -> None:
    target = dense_policy_from_scores({1: 0.5, 2: 0.0}, temperature=0.1)
    assert target.shape == (4672,)
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert target[1] > target[2]


def test_streaming_position_dataset(tmp_path: Path) -> None:
    pgn = tmp_path / "stream.pgn"
    pgn.write_text(
        """
[Event "stream"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
""".strip(),
        encoding="utf-8",
    )
    dataset = StreamingPositionDataset(
        [pgn],
        max_positions=3,
        shuffle_buffer_size=2,
        seed=42,
    )

    samples = list(dataset)
    assert len(samples) == 3
    assert all(sample["piece_idx"].shape == (64,) for sample in samples)
