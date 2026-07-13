from __future__ import annotations

import chess
import pytest

from kibitzer.inference import PositionEvaluation
from kibitzer.search import adaptive_puct_search, puct_search, terminal_value


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
    assert result.simulations == 8
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


def test_search_can_play_a_claimable_draw_for_uci() -> None:
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(move)

    assert board.can_claim_threefold_repetition()
    assert not board.is_game_over(claim_draw=False)
    with pytest.raises(ValueError, match="terminal"):
        puct_search(board, UniformEvaluator(), simulations=4)

    result = puct_search(board, UniformEvaluator(), simulations=4, claim_draw=False)
    assert result.move in board.legal_moves


def test_adaptive_search_stops_early_on_a_decisive_root() -> None:
    class SharpEvaluator:
        def evaluate(self, board: chess.Board) -> PositionEvaluation:
            legal_moves = list(board.legal_moves)
            priors = {move: 0.0 for move in legal_moves}
            priors[legal_moves[0]] = 1.0
            return PositionEvaluation(priors=priors, value=0.0)

    result = adaptive_puct_search(
        chess.Board(),
        SharpEvaluator(),
        stages=(8, 16),
    )

    assert result.simulations == 8
    assert result.stop_reason == "stable"


def test_adaptive_search_spends_more_on_an_ambiguous_root() -> None:
    result = adaptive_puct_search(
        chess.Board(),
        UniformEvaluator(),
        stages=(8, 16),
    )

    assert result.simulations == 16
    assert result.stop_reason == "max"
    assert sum(result.visits.values()) == 16
