"""Stockfish labeling helpers for phase-2 distillation."""

from __future__ import annotations

import chess
import chess.engine

from kibitzer.encoding import move_to_index


def pov_score_to_value(score: chess.engine.PovScore, turn: bool) -> float:
    cp = score.pov(turn).score(mate_score=100_000)
    if cp is None:
        return 0.0
    return max(-1.0, min(1.0, cp / 1000.0))


def analyze_actions(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    depth: int,
    multipv: int,
) -> tuple[dict[int, float], float]:
    limit = chess.engine.Limit(depth=depth)
    requested = min(multipv, board.legal_moves.count())
    infos = engine.analyse(board, limit, multipv=requested)
    if isinstance(infos, dict):
        infos = [infos]

    action_scores: dict[int, float] = {}
    value = 0.0
    for rank, info in enumerate(infos):
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        action_scores[move_to_index(move, board)] = pov_score_to_value(score, board.turn)
        if rank == 0:
            value = action_scores[move_to_index(move, board)]
    return action_scores, value
