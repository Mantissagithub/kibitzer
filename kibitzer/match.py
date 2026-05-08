"""Engine-vs-engine match runner.

Each engine is a ``Callable[[chess.Board], chess.Move]`` — accepts a board,
returns its chosen move. Wrap a Kibitzer instance with
:meth:`kibitzer.inference.KibitzerEngine.evaluate_at`, or pass any custom
function (uniform random, Stockfish wrapper, etc.).

Games end on checkmate, stalemate, insufficient material, claimable threefold
repetition, claimable fifty-move rule, or ``max_plies`` (declared draw).
Outputs proper PGN with headers per game.
"""

from __future__ import annotations

from typing import Callable

import chess
import chess.pgn


EngineFn = Callable[[chess.Board], chess.Move]


def play_match(
    engine_white: EngineFn,
    engine_black: EngineFn,
    n_games: int = 1,
    max_plies: int = 400,
    starting_fens: list[str] | None = None,
    swap_colors: bool = True,
    verbose: bool = False,
) -> dict:
    """Play ``n_games`` between two engine callables and return aggregate stats.

    Parameters
    ----------
    engine_white, engine_black : EngineFn
        The two engines. Color assignment in game 0 follows the parameter
        names; with ``swap_colors=True`` they alternate every game.
    n_games : int
        Number of games to play.
    max_plies : int
        Hard cap; games still in progress at this length are scored as draws
        with ``termination="max plies"``.
    starting_fens : list[str] | None
        If given, cycles through these FENs as starting positions
        (game ``i`` uses ``starting_fens[i % len(...)]``). ``None`` means
        always start from the standard initial position.
    swap_colors : bool
        If True, swap engine colors every game so each engine plays each side
        equally often (assuming an even ``n_games``).
    verbose : bool
        If True, print a one-line summary per game to stdout.

    Returns
    -------
    dict
        ``wins_a`` / ``wins_b`` / ``draws`` are tallied against the engine
        passed *first* / *second*, regardless of which color they played in
        any given game. ``games`` is a list of per-game records:
        ``{"pgn": str, "result": "1-0"|"0-1"|"1/2-1/2", "plies": int,
        "termination": str}``.
    """
    wins_a = 0
    wins_b = 0
    draws = 0
    games: list[dict] = []

    for game_idx in range(n_games):
        if starting_fens:
            fen = starting_fens[game_idx % len(starting_fens)]
            board = chess.Board(fen)
        else:
            board = chess.Board()

        swapped = swap_colors and (game_idx % 2 == 1)
        white_engine = engine_black if swapped else engine_white
        black_engine = engine_white if swapped else engine_black

        game = chess.pgn.Game()
        if starting_fens:
            game.setup(board)
        game.headers["Event"] = "Kibitzer match"
        game.headers["Round"] = str(game_idx + 1)
        game.headers["White"] = "B" if swapped else "A"
        game.headers["Black"] = "A" if swapped else "B"

        node = game
        plies = 0
        termination: str | None = None

        while plies < max_plies:
            if board.is_checkmate():
                termination = "checkmate"
                break
            if board.is_stalemate():
                termination = "stalemate"
                break
            if board.is_insufficient_material():
                termination = "insufficient material"
                break
            if board.can_claim_threefold_repetition():
                termination = "threefold repetition"
                break
            if board.can_claim_fifty_moves():
                termination = "fifty-move rule"
                break

            engine = white_engine if board.turn == chess.WHITE else black_engine
            move = engine(board)
            if move not in board.legal_moves:
                raise RuntimeError(
                    f"engine returned illegal move {move} at FEN {board.fen()}"
                )
            board.push(move)
            node = node.add_variation(move)
            plies += 1

        if termination is None:
            termination = "max plies"

        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
        else:
            result = "1/2-1/2"
        game.headers["Result"] = result

        if result == "1-0":
            if swapped:
                wins_b += 1
            else:
                wins_a += 1
        elif result == "0-1":
            if swapped:
                wins_a += 1
            else:
                wins_b += 1
        else:
            draws += 1

        games.append(
            {
                "pgn": str(game),
                "result": result,
                "plies": plies,
                "termination": termination,
            }
        )

        if verbose:
            print(f"game {game_idx + 1}: {result} ({termination}, {plies} plies)")

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "games": games,
    }
