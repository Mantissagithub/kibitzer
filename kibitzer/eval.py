"""Checkpoint evaluation: run a saved Kibitzer state-dict against a baseline.

Wraps :func:`kibitzer.match.play_match` with a fixed list of standard opening
positions (:data:`STARTING_FENS`) so different checkpoints are compared on the
same starts. Returns a small scorecard dict suitable for both human eyeballs
and machine consumption (logging to TensorBoard, JSON dumps, SPRT triggers).
"""

from __future__ import annotations

import math
import os
from typing import Literal

import chess
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.match import play_match
from kibitzer.model import Kibitzer
from kibitzer.opponents import RandomOpponent, StockfishOpponent


def _build_fens() -> list[str]:
    sequences: list[list[str]] = [
        [],                                                                 # startpos
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],                          # Italian
        ["e2e4", "c7c5"],                                                   # Sicilian
        ["e2e4", "e7e6"],                                                   # French
        ["d2d4", "d7d5", "c2c4", "e7e6"],                                   # QGD
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],                   # KID
    ]
    fens: list[str] = []
    for seq in sequences:
        b = chess.Board()
        for uci in seq:
            b.push_uci(uci)
        fens.append(b.fen())
    return fens


STARTING_FENS: list[str] = _build_fens()


def _load_model(path: str, device: str) -> Kibitzer:
    model = Kibitzer()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in (None, "model", "state_dict"):
        sd = ckpt if key is None else ckpt.get(key)
        if sd is None:
            continue
        try:
            model.load_state_dict(sd)
            return model
        except (TypeError, RuntimeError):
            continue
    raise RuntimeError(
        f"checkpoint at {path} did not match any known shape "
        "(raw state_dict / dict['model'] / dict['state_dict'])"
    )


class _TrackingOpponent:
    """Greedy Kibitzer wrapper that records ``|value|`` at every move."""

    def __init__(self, engine: KibitzerEngine) -> None:
        self.engine = engine
        self.abs_values: list[float] = []

    def __call__(self, board: chess.Board) -> chess.Move:
        saved = list(self.engine.history)
        try:
            if board.move_stack:
                self.engine.reset(board.root())
                for m in board.move_stack:
                    self.engine.push_move(m)
            else:
                self.engine.reset(board)
            out = self.engine.evaluate()
            self.abs_values.append(abs(out["value"]))
            return out["move_probs"][0][0]
        finally:
            self.engine.history = saved


def evaluate_checkpoint(
    checkpoint_path: str,
    opponent: Literal["random", "stockfish"] = "random",
    stockfish_skill: int = 0,
    stockfish_depth: int = 1,
    n_games: int = 20,
    device: str = "cuda",
    max_plies: int = 300,
    verbose: bool = False,
) -> dict:
    """Score a checkpoint against the chosen baseline; return a summary dict.

    The Kibitzer side plays greedy (temperature=0) for reproducibility. Games
    cycle through :data:`STARTING_FENS` and swap colors every other game.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = _load_model(checkpoint_path, device)
    engine = KibitzerEngine(model, device=device, dtype=dtype)
    tracker = _TrackingOpponent(engine)

    if opponent == "random":
        baseline = RandomOpponent(seed=0)
        baseline_desc = "random"
        result = play_match(
            tracker,
            baseline,
            n_games=n_games,
            max_plies=max_plies,
            starting_fens=STARTING_FENS,
            swap_colors=True,
            verbose=verbose,
        )
    elif opponent == "stockfish":
        baseline_desc = f"stockfish (skill={stockfish_skill}, depth={stockfish_depth})"
        with StockfishOpponent(
            depth=stockfish_depth, skill_level=stockfish_skill
        ) as sf:
            result = play_match(
                tracker,
                sf,
                n_games=n_games,
                max_plies=max_plies,
                starting_fens=STARTING_FENS,
                swap_colors=True,
                verbose=verbose,
            )
    else:
        raise ValueError(f"unknown opponent: {opponent!r}")

    wins = result["wins_a"]
    losses = result["wins_b"]
    draws = result["draws"]

    win_rate = (wins + 0.5 * draws) / n_games if n_games else 0.0
    wr_clamped = max(0.001, min(0.999, win_rate))
    approx_elo_diff = 400.0 * math.log10(wr_clamped / (1.0 - wr_clamped))

    avg_plies = (
        sum(g["plies"] for g in result["games"]) / n_games if n_games else 0.0
    )
    avg_value_pred = (
        sum(tracker.abs_values) / len(tracker.abs_values)
        if tracker.abs_values
        else 0.0
    )

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "opponent": baseline_desc,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "approx_elo_diff": approx_elo_diff,
        "avg_plies": avg_plies,
        "avg_value_pred": avg_value_pred,
    }
