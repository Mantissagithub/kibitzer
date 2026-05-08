"""Tests for kibitzer.inference.KibitzerEngine."""

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
    # Sorted desc.
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
    # Side to move flipped to black.
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


def test_principal_variation(engine: KibitzerEngine) -> None:
    _fresh(engine)
    saved_len = len(engine.history)
    pv = engine.get_principal_variation(depth=5)
    assert len(pv) <= 5
    # History must be restored exactly.
    assert len(engine.history) == saved_len
    # Every PV move must have been legal in the position it was played from.
    b = engine.history[-1].copy(stack=False)
    for m in pv:
        assert m in b.legal_moves
        b.push(m)
