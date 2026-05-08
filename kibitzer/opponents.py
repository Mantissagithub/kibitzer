"""Engine wrappers compatible with :func:`kibitzer.match.play_match`.

Each opponent implements ``__call__(board) -> chess.Move`` so it can be passed
directly as one of the engine arguments to ``play_match``:

* :class:`RandomOpponent` — uniform-random legal move (seedable baseline).
* :class:`StockfishOpponent` — wraps ``chess.engine.SimpleEngine`` over a
  Stockfish subprocess; use as a context manager to manage the subprocess.
* :class:`KibitzerOpponent` — wraps a :class:`KibitzerEngine` so its
  history-driven API can be queried with a single board.
"""

from __future__ import annotations

import random
import shutil
from typing import Any

import chess
import chess.engine

from kibitzer.inference import KibitzerEngine


class RandomOpponent:
    """Picks a uniform-random legal move. Useful as a sanity baseline."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def __call__(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))


class StockfishOpponent:
    """Stockfish wrapper. Must be used as a context manager.

    Parameters
    ----------
    path : str | None
        Path to the Stockfish binary. If ``None``, falls back to
        ``shutil.which("stockfish")``; raises ``FileNotFoundError`` if no
        binary can be located.
    depth : int | None
        Search depth limit. Mutually exclusive with ``time_ms`` — exactly one
        must be set.
    time_ms : int | None
        Per-move time budget in milliseconds. Mutually exclusive with
        ``depth``.
    skill_level : int | None
        Optional Stockfish skill level in ``[0, 20]``.

    Example
    -------
    >>> with StockfishOpponent(depth=10, skill_level=0) as sf:
    ...     result = play_match(my_engine, sf, n_games=10)
    """

    def __init__(
        self,
        path: str | None = None,
        depth: int | None = None,
        time_ms: int | None = None,
        skill_level: int | None = None,
    ) -> None:
        if (depth is None) == (time_ms is None):
            raise ValueError(
                "exactly one of `depth` or `time_ms` must be set "
                f"(got depth={depth}, time_ms={time_ms})"
            )
        if skill_level is not None and not 0 <= skill_level <= 20:
            raise ValueError(
                f"skill_level must be in [0, 20], got {skill_level}"
            )

        resolved = path if path is not None else shutil.which("stockfish")
        if resolved is None:
            raise FileNotFoundError(
                "stockfish binary not found on PATH; install Stockfish or "
                "pass an explicit `path=...`"
            )

        self.path = resolved
        self.depth = depth
        self.time_ms = time_ms
        self.skill_level = skill_level
        self._engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "StockfishOpponent":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        if self.skill_level is not None:
            self._engine.configure({"Skill Level": self.skill_level})
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __call__(self, board: chess.Board) -> chess.Move:
        if self._engine is None:
            raise RuntimeError(
                "StockfishOpponent must be used as a context manager"
            )
        if self.depth is not None:
            limit = chess.engine.Limit(depth=self.depth)
        else:
            assert self.time_ms is not None  # validated in __init__
            limit = chess.engine.Limit(time=self.time_ms / 1000.0)
        result = self._engine.play(board, limit)
        if result.move is None:
            raise RuntimeError(
                f"Stockfish returned no move at FEN {board.fen()}"
            )
        return result.move


class KibitzerOpponent:
    """Adapt a :class:`KibitzerEngine` to the ``play_match`` callable interface.

    On each call, repositions the underlying engine (via
    :meth:`KibitzerEngine.evaluate_at`) and returns its chosen move.
    """

    def __init__(
        self, engine: KibitzerEngine, temperature: float = 0.0
    ) -> None:
        self.engine = engine
        self.temperature = temperature

    def __call__(self, board: chess.Board) -> chess.Move:
        return self.engine.evaluate_at(board, temperature=self.temperature)
