"""Tests for kibitzer.data.LichessGameDataset / collate_games.

Hardcoded short PGNs cover the perspective + truncation + filter logic; the
multi-worker test patches torch.utils.data.get_worker_info to simulate two
workers without spinning up a real DataLoader.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from kibitzer.data import LichessGameDataset, collate_games


# Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7# — 7 plies, white wins.
SCHOLARS_MATE = """[Event "Test Blitz"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

# Fool's mate: 1.f3 e5 2.g4 Qh4# — 4 plies, black wins.
FOOLS_MATE = """[Event "Test Blitz"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "0-1"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. f3 e5 2. g4 Qh4# 0-1
"""

# Short "agreed draw": 8 plies, all zeros for value_target.
SHORT_DRAW = """[Event "Test Blitz"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1/2-1/2"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1/2-1/2
"""

# Same moves as Scholar's mate but low Elo — should be filtered out.
LOW_ELO = """[Event "Test Blitz"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Termination "Normal"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

ITALIAN_10_PLY = """[Event "Test Blitz"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1/2-1/2"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1/2-1/2
"""


def _write(tmp_path: Path, content: str, name: str = "game.pgn") -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_single_game_parses(tmp_path: Path) -> None:
    path = _write(tmp_path, SCHOLARS_MATE)
    ds = LichessGameDataset([path], min_plies=4, max_plies=20, shuffle_buffer_size=1)
    games = list(ds)
    assert len(games) == 1
    g = games[0]
    assert g["ply_count"] == 7  # Scholar's mate is 7 plies
    T = g["ply_count"]
    assert g["piece_idx"].shape == (T, 64)
    assert g["aux"].shape == (T, 7)
    assert g["move_idx"].shape == (T,)
    assert g["legal_mask"].shape == (T, 4672)
    assert g["value_target"].shape == (T,)


def test_value_target_perspective(tmp_path: Path) -> None:
    # White wins → at white's turn (even t) +1, at black's turn (odd t) -1.
    path_w = _write(tmp_path, SCHOLARS_MATE, "w.pgn")
    g = next(iter(LichessGameDataset([path_w], min_plies=4, shuffle_buffer_size=1)))
    assert g["value_target"].tolist() == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    # Black wins → flipped.
    path_b = _write(tmp_path, FOOLS_MATE, "b.pgn")
    g = next(iter(LichessGameDataset([path_b], min_plies=4, shuffle_buffer_size=1)))
    assert g["value_target"].tolist() == [-1.0, 1.0, -1.0, 1.0]

    # Draw → zeros throughout.
    path_d = _write(tmp_path, SHORT_DRAW, "d.pgn")
    g = next(iter(LichessGameDataset([path_d], min_plies=4, shuffle_buffer_size=1)))
    assert g["value_target"].tolist() == [0.0] * g["ply_count"]


def test_legal_mask_correct(tmp_path: Path) -> None:
    path = _write(tmp_path, SCHOLARS_MATE)
    ds = LichessGameDataset([path], min_plies=4, shuffle_buffer_size=1)
    g = next(iter(ds))
    # Standard starting position has 20 legal moves (16 pawn pushes + 4 knight).
    assert int(g["legal_mask"][0].sum()) == 20


def test_played_move_is_legal(tmp_path: Path) -> None:
    path = _write(tmp_path, SCHOLARS_MATE)
    ds = LichessGameDataset([path], min_plies=4, shuffle_buffer_size=1)
    g = next(iter(ds))
    for t in range(g["ply_count"]):
        idx = int(g["move_idx"][t])
        assert bool(g["legal_mask"][t, idx]), (
            f"played move {idx} at ply {t} is not in legal_mask"
        )


def test_filter_low_elo(tmp_path: Path) -> None:
    high = _write(tmp_path, SCHOLARS_MATE, "high.pgn")
    low = _write(tmp_path, LOW_ELO, "low.pgn")
    ds = LichessGameDataset(
        [high, low], min_elo=2400, min_plies=4, shuffle_buffer_size=1
    )
    games = list(ds)
    assert len(games) == 1


def test_truncation(tmp_path: Path) -> None:
    # Scholar's mate is 7 plies; truncate to 4.
    path = _write(tmp_path, SCHOLARS_MATE)
    ds = LichessGameDataset(
        [path], min_plies=4, max_plies=4, shuffle_buffer_size=1
    )
    g = next(iter(ds))
    assert g["ply_count"] == 4
    assert g["piece_idx"].shape == (4, 64)
    assert g["legal_mask"].shape == (4, 4672)


def test_collate_padding(tmp_path: Path) -> None:
    path_w = _write(tmp_path, SCHOLARS_MATE, "w.pgn")  # 7 plies
    path_b = _write(tmp_path, FOOLS_MATE, "b.pgn")  # 4 plies
    ds = LichessGameDataset([path_w, path_b], min_plies=4, shuffle_buffer_size=1)
    games = list(ds)
    assert len(games) == 2
    batch = collate_games(games)

    T_max = 7
    assert batch["piece_idx"].shape == (2, T_max, 64)
    assert batch["aux"].shape == (2, T_max, 7)
    assert batch["move_idx"].shape == (2, T_max)
    assert batch["legal_mask"].shape == (2, T_max, 4672)
    assert batch["value_target"].shape == (2, T_max)
    assert batch["loss_mask"].shape == (2, T_max)

    for b in range(2):
        plen = games[b]["ply_count"]
        assert batch["loss_mask"][b, :plen].all()
        assert not batch["loss_mask"][b, plen:].any()


def test_multi_worker_partitioning(tmp_path: Path) -> None:
    paths = []
    for i, content in enumerate(
        [SCHOLARS_MATE, FOOLS_MATE, SHORT_DRAW, ITALIAN_10_PLY]
    ):
        paths.append(_write(tmp_path, content, f"g{i}.pgn"))

    def _wi(worker_id: int, num_workers: int) -> MagicMock:
        wi = MagicMock()
        wi.id = worker_id
        wi.num_workers = num_workers
        return wi

    with patch("kibitzer.data.get_worker_info", return_value=_wi(0, 2)):
        games_w0 = list(
            LichessGameDataset(paths, min_plies=4, shuffle_buffer_size=1)
        )
    with patch("kibitzer.data.get_worker_info", return_value=_wi(1, 2)):
        games_w1 = list(
            LichessGameDataset(paths, min_plies=4, shuffle_buffer_size=1)
        )

    # paths[0::2] = [path0, path2]; paths[1::2] = [path1, path3].
    assert len(games_w0) == 2
    assert len(games_w1) == 2
    # Disjoint and complete: 4 distinct ply_counts across the two workers
    # ([7, 8] for worker 0, [4, 10] for worker 1).
    counts = sorted(
        [g["ply_count"] for g in games_w0] + [g["ply_count"] for g in games_w1]
    )
    assert counts == [4, 7, 8, 10]
