"""reward shaping utilities for chess RL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import chess.engine

from kibitzer.rl_config import RewardMix


def _clip_centipawns(cp: int, limit: int = 1000) -> int:
    return max(-limit, min(limit, int(cp)))


def centipawns_to_scalar(cp: int, scale: float = 400.0) -> float:
    """map centipawns to a bounded scalar in [-1, 1]."""
    clipped = _clip_centipawns(cp)
    return math.tanh(clipped / scale)


def score_to_scalar(
    score: chess.engine.Score | chess.engine.PovScore,
    acting_color: bool,
) -> float:
    """convert a python-chess score to the acting side's bounded perspective."""
    if isinstance(score, chess.engine.PovScore):
        pov = score.pov(acting_color)
    else:
        pov = score if acting_color == chess.WHITE else -score

    mate = pov.mate()
    if mate is not None:
        if mate == 0:
            return 0.0
        return 1.0 if mate > 0 else -1.0

    cp = pov.score(mate_score=100000)
    if cp is None:
        return 0.0
    return centipawns_to_scalar(cp)


def terminal_reward(board: chess.Board, acting_color: bool) -> float:
    """terminal outcome in the acting side's perspective."""
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == acting_color else -1.0


@dataclass
class RewardBreakdown:
    stockfish_delta: float
    value_delta: float
    terminal: float
    total: float


def mix_rewards(
    reward_mix: RewardMix,
    stockfish_before: float,
    stockfish_after: float,
    value_before: float,
    value_after: float,
    terminal: float = 0.0,
) -> RewardBreakdown:
    """combine bounded potential differences into one scalar reward."""
    stockfish_delta = stockfish_after - stockfish_before
    value_delta = value_after - value_before
    total = (
        reward_mix.stockfish_delta_weight * stockfish_delta
        + reward_mix.value_delta_weight * value_delta
        + reward_mix.terminal_weight * terminal
    )
    return RewardBreakdown(
        stockfish_delta=stockfish_delta,
        value_delta=value_delta,
        terminal=terminal,
        total=total,
    )
