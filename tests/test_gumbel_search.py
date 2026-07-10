from __future__ import annotations

import random

import chess

from search_lab.gumbel import gumbel_search, sequential_halving_schedule
from kibitzer.inference import PositionEvaluation


class _UniformEvaluator:
    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        moves = list(board.legal_moves)
        probability = 1.0 / len(moves)
        return PositionEvaluation(
            priors={move: probability for move in moves},
            value=0.0,
        )


def test_sequential_halving_schedule_matches_rounds() -> None:
    assert sequential_halving_schedule(4, 16) == (
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
    )


def test_gumbel_search_preserves_board_and_budget() -> None:
    board = chess.Board()
    original_fen = board.fen()
    result = gumbel_search(
        board,
        _UniformEvaluator(),
        simulations=16,
        rng=random.Random(7),
    )
    assert result.move in board.legal_moves
    assert sum(result.visits.values()) == 16
    assert board.fen() == original_fen


def test_zero_scale_keeps_the_strong_policy_move() -> None:
    class _PriorEvaluator:
        def evaluate(self, board: chess.Board) -> PositionEvaluation:
            moves = list(board.legal_moves)
            priors = {move: 1e-4 for move in moves}
            if not board.move_stack:
                priors[chess.Move.from_uci("e2e4")] = 0.9
            total = sum(priors.values())
            return PositionEvaluation(
                priors={move: value / total for move, value in priors.items()},
                value=0.0,
            )

    result = gumbel_search(
        chess.Board(),
        _PriorEvaluator(),
        simulations=32,
        gumbel_scale=0.0,
        rng=random.Random(7),
    )
    assert result.move == chess.Move.from_uci("e2e4")
