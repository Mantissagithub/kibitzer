from __future__ import annotations

import chess
import torch

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, index_to_move, legal_move_mask, move_to_index


def test_start_board_tensor() -> None:
    encoded = board_to_tensor(chess.Board())
    assert encoded["piece_idx"].shape == (64,)
    assert encoded["piece_idx"].dtype == torch.long
    assert encoded["aux"].shape == (7,)
    assert encoded["aux"][0].item() == 1.0


def test_legal_move_mask_matches_legal_moves() -> None:
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.sum().item() == board.legal_moves.count()


def test_opening_move_roundtrip() -> None:
    board = chess.Board()
    for uci in ["e2e4", "g1f3", "b1c3"]:
        move = chess.Move.from_uci(uci)
        index = move_to_index(move, board)
        assert 0 <= index < ACTION_SIZE
        assert index_to_move(index, board) == move


def test_promotion_roundtrip() -> None:
    board = chess.Board("k7/4P3/8/8/8/8/8/7K w - - 0 1")
    for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        move = chess.Move(chess.E7, chess.E8, promotion=piece)
        assert move in board.legal_moves
        assert index_to_move(move_to_index(move, board), board) == move
