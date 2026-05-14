from __future__ import annotations

import random

import chess
import pytest
import torch

from kibitzer.encoding import ACTION_SIZE
from kibitzer.masking import legal_move_mask


def _random_positions(n: int, seed: int) -> list[chess.Board]:
    positions: list[chess.Board] = []
    rng = random.Random(seed)
    while len(positions) < n:
        b = chess.Board()
        for _ in range(200):
            if b.is_game_over(claim_draw=False):
                break
            positions.append(b.copy(stack=False))
            if len(positions) >= n:
                break
            b.push(rng.choice(list(b.legal_moves)))
    return positions[:n]


@pytest.fixture(scope="module")
def random_positions() -> list[chess.Board]:
    return _random_positions(n=500, seed=20240509)


def test_starting_position_mask() -> None:
    mask = legal_move_mask(chess.Board())
    assert mask.shape == (ACTION_SIZE,)
    assert mask.dtype == torch.bool
    # 8 pawns × 2 push options + 2 knights × 2 squares = 20 legal moves.
    assert mask.sum().item() == 20


def test_mask_count_matches(random_positions: list[chess.Board]) -> None:
    for b in random_positions:
        mask = legal_move_mask(b)
        assert mask.sum().item() == b.legal_moves.count(), (
            f"mask count mismatch at {b.fen()}: "
            f"mask={mask.sum().item()} legal={b.legal_moves.count()}"
        )


def test_checkmate_position() -> None:
    # fool's mate after 1.f3 e5 2.g4 qh4#: white is in check with no moves.
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    assert legal_move_mask(b).sum().item() == 0


def test_stalemate_position() -> None:
    # lone-king stalemate: black is not in check, but every square is covered.
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert b.is_stalemate()
    assert legal_move_mask(b).sum().item() == 0
