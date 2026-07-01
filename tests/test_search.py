from __future__ import annotations

import chess

from kibitzer.inference import PositionEvaluation
from kibitzer.search import puct_search, terminal_value


class UniformEvaluator:
    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        legal_moves = list(board.legal_moves)
        probability = 1.0 / len(legal_moves)
        return PositionEvaluation(
            priors={move: probability for move in legal_moves},
            value=0.0,
        )


def test_search_returns_legal_move_and_preserves_board() -> None:
    board = chess.Board()
    original_fen = board.fen()

    result = puct_search(board, UniformEvaluator(), simulations=8)

    assert result.move in board.legal_moves
    assert sum(result.visits.values()) == 8
    assert board.fen() == original_fen


def test_terminal_value_uses_side_to_move_perspective() -> None:
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(move)

    assert board.is_checkmate()
    assert terminal_value(board) == -1.0


def test_zero_value_scale_follows_policy_prior() -> None:
    class PriorEvaluator:
        def evaluate(self, board: chess.Board) -> PositionEvaluation:
            legal_moves = list(board.legal_moves)
            priors = {move: 0.0 for move in legal_moves}
            if not board.move_stack:
                priors[chess.Move.from_uci("e2e4")] = 0.9
                priors[chess.Move.from_uci("d2d4")] = 0.1
            else:
                probability = 1.0 / len(legal_moves)
                priors = {move: probability for move in legal_moves}
            return PositionEvaluation(priors=priors, value=1.0)

    result = puct_search(
        chess.Board(),
        PriorEvaluator(),
        simulations=32,
        value_scale=0.0,
    )

    assert result.move == chess.Move.from_uci("e2e4")
