from __future__ import annotations

from pathlib import Path

import torch

from kibitzer.data import dense_policy_from_scores, iter_pgn_samples


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
    assert samples[0].value == 1.0
    assert samples[1].value == -1.0


def test_dense_policy_from_scores() -> None:
    target = dense_policy_from_scores({1: 0.5, 2: 0.0}, temperature=0.1)
    assert target.shape == (4672,)
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert target[1] > target[2]
