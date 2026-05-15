from __future__ import annotations

import chess
import chess.engine

from kibitzer.rl_config import RewardMix
from kibitzer.rl_reward import centipawns_to_scalar, mix_rewards, score_to_scalar, terminal_reward


def test_centipawns_to_scalar_is_bounded() -> None:
    assert -1.0 <= centipawns_to_scalar(-5000) <= 0.0
    assert 0.0 <= centipawns_to_scalar(5000) <= 1.0


def test_score_to_scalar_respects_pov() -> None:
    score = chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE)
    white = score_to_scalar(score, acting_color=chess.WHITE)
    black = score_to_scalar(score, acting_color=chess.BLACK)
    assert white > 0.0
    assert black < 0.0


def test_score_to_scalar_handles_mate() -> None:
    score = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
    assert score_to_scalar(score, acting_color=chess.WHITE) == 1.0
    assert score_to_scalar(score, acting_color=chess.BLACK) == -1.0


def test_terminal_reward_from_acting_side() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    assert terminal_reward(board, acting_color=chess.WHITE) == -1.0
    assert terminal_reward(board, acting_color=chess.BLACK) == 1.0


def test_mix_rewards_combines_deltas() -> None:
    mix = RewardMix(stockfish_delta_weight=0.25, value_delta_weight=0.1, terminal_weight=1.0)
    out = mix_rewards(
        mix,
        stockfish_before=0.1,
        stockfish_after=0.5,
        value_before=-0.2,
        value_after=0.2,
        terminal=1.0,
    )
    assert out.stockfish_delta == 0.4
    assert out.value_delta == 0.4
    assert abs(out.total - 1.14) < 1e-6
