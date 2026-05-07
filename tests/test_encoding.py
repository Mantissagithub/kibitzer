"""Tests for kibitzer.encoding."""

from __future__ import annotations

import random

import chess
import pytest
import torch

from kibitzer.encoding import (
    ACTION_SIZE,
    board_to_tensor,
    index_to_move,
    move_to_index,
)


# ---------------------------------------------------------------------------
# Shared fixture: 1000 positions reached by random self-play (deterministic).
# ---------------------------------------------------------------------------


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
    return _random_positions(n=1000, seed=20240508)


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------


def test_board_to_tensor_shapes() -> None:
    t = board_to_tensor(chess.Board())
    assert set(t.keys()) == {"piece_idx", "aux"}
    assert t["piece_idx"].shape == (64,)
    assert t["piece_idx"].dtype == torch.long
    assert t["aux"].shape == (7,)
    assert t["aux"].dtype == torch.float32


def test_board_to_tensor_pieces() -> None:
    t = board_to_tensor(chess.Board())

    expected = torch.zeros(64, dtype=torch.long)
    expected[0:8] = torch.tensor([4, 2, 3, 5, 6, 3, 2, 4])   # white back rank
    expected[8:16] = 1                                        # white pawns
    # squares 16..47 stay 0 (empty)
    expected[48:56] = 7                                       # black pawns
    expected[56:64] = torch.tensor([10, 8, 9, 11, 12, 9, 8, 10])  # black back rank

    assert torch.equal(t["piece_idx"], expected), (
        f"piece_idx mismatch:\n  got: {t['piece_idx'].tolist()}\n  exp: {expected.tolist()}"
    )

    aux = t["aux"]
    assert aux[0].item() == 1.0          # white to move
    assert aux[1].item() == 1.0          # WK
    assert aux[2].item() == 1.0          # WQ
    assert aux[3].item() == 1.0          # BK
    assert aux[4].item() == 1.0          # BQ
    assert aux[5].item() == -1.0         # no en passant
    assert aux[6].item() == 0.0          # halfmove clock 0


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------


def test_move_roundtrip(random_positions: list[chess.Board]) -> None:
    for b in random_positions:
        for move in b.legal_moves:
            idx = move_to_index(move, b)
            decoded = index_to_move(idx, b)
            assert decoded == move, (
                f"round-trip failed at {b.fen()}: move={move.uci()} "
                f"idx={idx} decoded={decoded.uci()}"
            )


def test_move_index_in_range(random_positions: list[chess.Board]) -> None:
    for b in random_positions:
        for move in b.legal_moves:
            idx = move_to_index(move, b)
            assert 0 <= idx < ACTION_SIZE, (
                f"idx={idx} out of [0, {ACTION_SIZE}) at {b.fen()} for {move.uci()}"
            )


def test_promotion_encoding() -> None:
    # White pawn on e7 with no capture targets — only push promotions are legal.
    # Black king on a8 to keep the e-file clear so the pawn can actually push.
    b = chess.Board("k7/4P3/8/8/8/8/8/7K w - - 0 1")
    for piece, name in [
        (chess.QUEEN, "queen"),
        (chess.ROOK, "rook"),
        (chess.BISHOP, "bishop"),
        (chess.KNIGHT, "knight"),
    ]:
        move = chess.Move(chess.E7, chess.E8, promotion=piece)
        assert move in b.legal_moves, f"{name} promotion should be legal"

        idx = move_to_index(move, b)
        plane = idx % 73
        if piece == chess.QUEEN:
            assert plane < 56, f"queen promo plane={plane}, expected queen-like (<56)"
        else:
            assert 64 <= plane < 73, (
                f"{name} promo plane={plane}, expected underpromotion [64, 73)"
            )

        decoded = index_to_move(idx, b)
        assert decoded == move, f"{name} round-trip failed: {move} -> {idx} -> {decoded}"

    # Capture-promotion: black knights on d8 and f8 give the e7 pawn diagonal targets.
    b2 = chess.Board("3nkn2/4P3/8/8/8/8/8/7K w - - 0 1")
    for to_sq in (chess.D8, chess.F8):
        for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            move = chess.Move(chess.E7, to_sq, promotion=piece)
            assert move in b2.legal_moves
            idx = move_to_index(move, b2)
            assert 0 <= idx < ACTION_SIZE
            assert index_to_move(idx, b2) == move

    # Black underpromotion: ensure the rank delta is correctly recovered from board.turn.
    # White king on a1 (off the e-file) so the e2 pawn can push to e1.
    b3 = chess.Board("4k3/8/8/8/8/8/4p3/K7 b - - 0 1")
    for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        move = chess.Move(chess.E2, chess.E1, promotion=piece)
        assert move in b3.legal_moves
        idx = move_to_index(move, b3)
        assert index_to_move(idx, b3) == move


def test_castling_encoding() -> None:
    # White to move, both sides have all castling rights.
    bw = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    legal = {m.uci(): m for m in bw.legal_moves}
    for uci in ("e1g1", "e1c1"):
        assert uci in legal, f"{uci} should be a legal castling move"
        move = legal[uci]
        idx = move_to_index(move, bw)
        decoded = index_to_move(idx, bw)
        assert decoded == move, f"white castling round-trip failed: {move} -> {decoded}"

    # Black to move, same rights.
    bb = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    legal_b = {m.uci(): m for m in bb.legal_moves}
    for uci in ("e8g8", "e8c8"):
        assert uci in legal_b, f"{uci} should be a legal castling move"
        move = legal_b[uci]
        idx = move_to_index(move, bb)
        decoded = index_to_move(idx, bb)
        assert decoded == move, f"black castling round-trip failed: {move} -> {decoded}"


def test_en_passant() -> None:
    b = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "f7f5"):
        b.push_uci(uci)

    assert b.ep_square == chess.F6, f"expected ep_square=F6, got {b.ep_square}"

    ep_move = chess.Move(chess.E5, chess.F6)
    assert ep_move in b.legal_moves, "e5xf6 e.p. should be legal"

    idx = move_to_index(ep_move, b)
    assert 0 <= idx < ACTION_SIZE
    decoded = index_to_move(idx, b)
    assert decoded == ep_move, f"en-passant round-trip failed: {ep_move} -> {decoded}"

    aux = board_to_tensor(b)["aux"]
    assert aux[5].item() == float(chess.square_file(chess.F6))  # 5 (f-file)
