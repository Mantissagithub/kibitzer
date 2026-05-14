"""evaluate a kibitzer checkpoint via cutechess-cli."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Callable, Literal

from kibitzer.cutechess_runner import run_match


_REPO_ROOT = Path(__file__).resolve().parent.parent
_UCI_SCRIPT = _REPO_ROOT / "scripts" / "uci.py"
_KIBITZER_CMD = shlex.join([sys.executable, str(_UCI_SCRIPT)])

OpponentName = Literal[
    "stockfish-elo-1320",
    "stockfish-elo-1500",
    "stockfish-elo-1800",
    "stockfish-full",
    "self-vs-prev",
]
_VALID_OPPONENTS = {
    "stockfish-elo-1320",
    "stockfish-elo-1500",
    "stockfish-elo-1800",
    "stockfish-full",
    "self-vs-prev",
}


def evaluate_checkpoint(
    checkpoint_path: str,
    opponent: OpponentName = "stockfish-elo-1320",
    prev_checkpoint: str | None = None,
    n_games: int = 20,
    time_per_move_ms: int = 200,
    stockfish_path: str = "stockfish",
    cutechess_path: str = "cutechess-cli",
    pgn_dir: str = "eval_pgns",
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """run cutechess and return a flat scorecard."""
    if opponent not in _VALID_OPPONENTS:
        raise ValueError(
            f"unknown opponent: {opponent!r}; "
            f"expected one of {sorted(_VALID_OPPONENTS)}"
        )

    a_cmd = _KIBITZER_CMD
    a_options: dict[str, str] = {"Checkpoint": checkpoint_path}

    if opponent == "self-vs-prev":
        if not prev_checkpoint:
            raise ValueError(
                "opponent='self-vs-prev' requires prev_checkpoint"
            )
        b_cmd = _KIBITZER_CMD
        b_options: dict[str, str] = {"Checkpoint": prev_checkpoint}
    elif opponent == "stockfish-full":
        b_cmd = stockfish_path
        b_options = {}
    else:
        elo = int(opponent.rsplit("-", 1)[1])
        b_cmd = stockfish_path
        b_options = {"UCI_LimitStrength": "true", "UCI_Elo": str(elo)}

    pgn_root = Path(pgn_dir)
    pgn_root.mkdir(parents=True, exist_ok=True)
    pgn_path = pgn_root / f"{Path(checkpoint_path).stem}_vs_{opponent}.pgn"

    raw = run_match(
        engine_a_cmd=a_cmd,
        engine_b_cmd=b_cmd,
        engine_a_options=a_options,
        engine_b_options=b_options,
        n_games=n_games,
        time_per_move_ms=time_per_move_ms,
        pgn_output=str(pgn_path),
        cutechess_path=cutechess_path,
        on_progress=on_progress,
    )

    wins = raw["wins"]
    losses = raw["losses"]
    draws = raw["draws"]
    score = wins + 0.5 * draws
    win_rate = score / n_games if n_games else 0.0

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "opponent": opponent,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "win_rate": win_rate,
        "elo_diff": raw["elo_diff"],
        "elo_err": raw["elo_err"],
        "pgn_path": raw["pgn_path"],
    }
