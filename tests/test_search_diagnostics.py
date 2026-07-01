from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

import scripts.diagnose_search as diagnostics
from scripts.diagnose_search import Candidate, build_oracle, evaluate, game_split, load_tactics


def test_game_split_is_deterministic() -> None:
    assert game_split("month.pgn:42", 7) == game_split("month.pgn:42", 7)
    assert game_split("month.pgn:42", 7) in {"validation", "test"}


def test_mate_sanity_suite_has_thirty_positions() -> None:
    path = Path(__file__).parents[1] / "resources" / "mate_in_one_sanity.epd"
    tactics = load_tactics(path)

    assert len(tactics) == 30
    assert all(best_moves for _, best_moves in tactics)


def test_consumed_test_split_is_locked(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.pt"
    oracle.with_suffix(".pt.test.lock").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="already consumed"):
        evaluate(
            Namespace(
                oracle=oracle,
                split="test",
                output=tmp_path / "test.json",
            )
        )


def test_build_oracle_locks_every_split_and_bin(tmp_path: Path, monkeypatch) -> None:
    pgn = tmp_path / "unseen.pgn"
    pgn.write_text("placeholder", encoding="utf-8")
    centipawns = [0, 100, 300, 600]
    candidates = [
        Candidate(f"fen:{cp}", f"{split}:{cp}", split)
        for split in ("validation", "test")
        for cp in centipawns
    ]

    class FakePool:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def analyse(self, tasks, *, description: str):
            del description
            return [(int(fen.rsplit(":", 1)[1]), "e2e4") for fen, _ in tasks]

    monkeypatch.setattr(diagnostics, "StockfishPool", FakePool)
    monkeypatch.setattr(
        diagnostics,
        "iter_real_positions",
        lambda *args, **kwargs: iter(candidates),
    )
    output = tmp_path / "oracle.pt"
    build_oracle(
        Namespace(
            output=output,
            pgn=[str(pgn)],
            positions_per_bin=1,
            oversample=1,
            stockfish_path="unused",
            stockfish_workers=1,
            prescan_depth=1,
            oracle_depth=2,
            min_ply=0,
            position_stride=1,
            seed=42,
            prescan_batch_size=8,
        )
    )

    payload = torch.load(output, weights_only=False)
    assert len(payload["records"]) == 8
    assert {record["split"] for record in payload["records"]} == {
        "validation",
        "test",
    }
