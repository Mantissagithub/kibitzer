from __future__ import annotations

import random
import shutil

import chess
import pytest
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer
from kibitzer.opponents import KibitzerOpponent, RandomOpponent, StockfishOpponent


def test_random_opponent() -> None:
    op = RandomOpponent(seed=0)
    b = chess.Board()
    for _ in range(20):
        if b.is_game_over():
            break
        m = op(b)
        assert m in b.legal_moves
        b.push(m)


def test_kibitzer_opponent_legal() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    engine = KibitzerEngine(Kibitzer(), device=device, dtype=dtype)
    op = KibitzerOpponent(engine, temperature=1.0)

    rng = random.Random(0)
    b = chess.Board()
    for _ in range(20):
        if b.is_game_over():
            break
        m = op(b)
        assert m in b.legal_moves
        # drive the board forward so each call sees a different position.
        # we're testing legality across many positions,
        # not training the engine's own play.
        b.push(rng.choice(list(b.legal_moves)))


@pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish not installed"
)
def test_stockfish_opponent() -> None:
    with StockfishOpponent(depth=1) as sf:
        b = chess.Board()
        m = sf(b)
        assert m in b.legal_moves
        b.push(m)
        m2 = sf(b)
        assert m2 in b.legal_moves


def test_stockfish_rejects_both_limits() -> None:
    # limit validation runs before path resolution, so this is env-independent.
    with pytest.raises(ValueError, match="exactly one"):
        StockfishOpponent(depth=10, time_ms=100)


def test_stockfish_rejects_no_limit() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        StockfishOpponent()
