"""engine-vs-engine match runner."""

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
    """play games between two engine callables and return aggregate stats."""
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
