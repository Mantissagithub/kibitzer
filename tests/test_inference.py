from __future__ import annotations

import math
import random

import chess
import numpy as np
import pytest
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


@pytest.fixture(scope="module")
def engine() -> KibitzerEngine:
    device, dtype = _device_dtype()
    model = Kibitzer()
    return KibitzerEngine(model, device=device, dtype=dtype)


def _fresh(engine: KibitzerEngine) -> KibitzerEngine:
    engine.reset()
    return engine


def test_engine_init(engine: KibitzerEngine) -> None:
    _fresh(engine)
    assert len(engine.history) == 1
    assert engine.history[0].fen() == chess.STARTING_FEN


def test_evaluate_starting_position(engine: KibitzerEngine) -> None:
    _fresh(engine)
    out = engine.evaluate()

    policy = out["policy"]
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (4672,)
    assert policy.dtype == np.float32
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert int((policy > 0).sum()) == 20  # 20 legal opening moves

    assert -1.0 <= out["value"] <= 1.0
    assert math.isfinite(out["value"])

    assert len(out["legal_moves"]) == 20
    assert len(out["move_probs"]) == 20
    # sorted descending.
    ps = [p for _, p in out["move_probs"]]
    assert all(ps[i] >= ps[i + 1] for i in range(len(ps) - 1))


def test_select_move_legal(engine: KibitzerEngine) -> None:
    _fresh(engine)
    legal = set(engine.history[-1].legal_moves)
    for _ in range(50):
        m = engine.select_move(temperature=1.0)
        assert m in legal


def test_temperature_zero_deterministic(engine: KibitzerEngine) -> None:
    _fresh(engine)
    first = engine.select_move(temperature=0.0)
    for _ in range(9):
        assert engine.select_move(temperature=0.0) == first


def test_temperature_high_varies(engine: KibitzerEngine) -> None:
    _fresh(engine)
    torch.manual_seed(0)
    seen: set[chess.Move] = set()
    for _ in range(50):
        seen.add(engine.select_move(temperature=2.0))
    assert len(seen) >= 3, f"only {len(seen)} distinct moves over 50 samples"


def test_push_move_updates_history(engine: KibitzerEngine) -> None:
    _fresh(engine)
    assert len(engine.history) == 1
    engine.push_move(chess.Move.from_uci("e2e4"))
    assert len(engine.history) == 2
    # side to move flipped to black.
    assert engine.history[-1].turn is chess.BLACK
    assert engine.history[0].turn is chess.WHITE  # original snapshot intact


def test_full_game_loop(engine: KibitzerEngine) -> None:
    _fresh(engine)
    rng = random.Random(42)
    for _ in range(20):
        if engine.history[-1].is_game_over():
            break
        out = engine.evaluate()
        assert abs(float(out["policy"].sum()) - 1.0) < 1e-5
        assert math.isfinite(out["value"])
        legal = list(engine.history[-1].legal_moves)
        engine.push_move(rng.choice(legal))


def test_evaluate_at(engine: KibitzerEngine) -> None:
    _fresh(engine)
    # build a board with moves on its move_stack.
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push(chess.Move.from_uci(uci))

    saved_len = len(engine.history)
    move = engine.evaluate_at(board)
    # returned move must be legal in the queried board.
    assert move in board.legal_moves
    # engine state must be restored.
    assert len(engine.history) == saved_len


def test_principal_variation(engine: KibitzerEngine) -> None:
    _fresh(engine)
    saved_len = len(engine.history)
    pv = engine.get_principal_variation(depth=5)
    assert len(pv) <= 5
    # history must be restored exactly.
    assert len(engine.history) == saved_len
    # every pv move must have been legal in its position.
    b = engine.history[-1].copy(stack=False)
    for m in pv:
        assert m in b.legal_moves
        b.push(m)


def test_context_window_trims_long_history() -> None:
    device, dtype = _device_dtype()
    engine = KibitzerEngine(Kibitzer(), device=device, dtype=dtype, context_window=4)
    board = chess.Board()
    moves = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
    ]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        board.push(move)
        engine.push_move(move)

    assert len(engine.history) == len(moves) + 1
    out = engine.evaluate()
    assert len(out["legal_moves"]) > 0
    assert abs(float(out["policy"].sum()) - 1.0) < 1e-5


def test_select_moves_batch_legal(engine: KibitzerEngine) -> None:
    _fresh(engine)
    board_a = chess.Board()
    board_b = chess.Board()
    for uci in ("d2d4", "d7d5", "c2c4"):
        board_b.push(chess.Move.from_uci(uci))

    moves = engine.select_moves([board_a, board_b], temperature=0.0)
    assert len(moves) == 2
    assert moves[0] in board_a.legal_moves
    assert moves[1] in board_b.legal_moves
