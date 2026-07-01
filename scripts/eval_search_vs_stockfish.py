from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
from tqdm import trange

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


OPENINGS = [
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6"),
    ("g1f3", "d7d5", "g2g3", "g8f6"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PUCT search against Stockfish.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--stockfish-elo", type=int, default=1320)
    parser.add_argument("--stockfish-time", type=float, default=0.05)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    if args.simulations < 1:
        raise SystemExit("--simulations must be at least 1")
    if args.stockfish_time <= 0.0:
        raise SystemExit("--stockfish-time must be positive")
    if args.max_plies < 1:
        raise SystemExit("--max-plies must be at least 1")


def starting_board(game_index: int) -> chess.Board:
    board = chess.Board()
    for move_uci in OPENINGS[(game_index // 2) % len(OPENINGS)]:
        board.push_uci(move_uci)
    return board


def play_game(
    *,
    game_index: int,
    evaluator: ModelEvaluator,
    engine: chess.engine.SimpleEngine,
    simulations: int,
    c_puct: float,
    stockfish_time: float,
    max_plies: int,
) -> tuple[chess.pgn.Game, str]:
    board = starting_board(game_index)
    network_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Kibitzer PUCT vs Stockfish"
    game.headers["Round"] = str(game_index + 1)
    game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = "Kibitzer" if network_color == chess.BLACK else "Stockfish"
    node = game.end()

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            move = puct_search(
                board,
                evaluator,
                simulations=simulations,
                c_puct=c_puct,
            ).move
        else:
            result = engine.play(board, chess.engine.Limit(time=stockfish_time))
            if result.move is None:
                raise RuntimeError("Stockfish returned no move for a non-terminal board")
            move = result.move
        board.push(move)
        node = node.add_variation(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    result_text = outcome.result() if outcome is not None else "1/2-1/2"
    game.headers["Result"] = result_text
    game.headers["Termination"] = (
        outcome.termination.name if outcome is not None else "max plies adjudication"
    )
    return game, result_text


def network_result(result: str, network_is_white: bool) -> str:
    if result == "1/2-1/2":
        return "draw"
    network_won = (result == "1-0") == network_is_white
    return "win" if network_won else "loss"


def main() -> None:
    args = parse_args()
    validate_args(args)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"win": 0, "draw": 0, "loss": 0}

    with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
        with args.out.open("w", encoding="utf-8") as output:
            for game_index in trange(args.games, desc="games"):
                game, result = play_game(
                    game_index=game_index,
                    evaluator=evaluator,
                    engine=engine,
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    stockfish_time=args.stockfish_time,
                    max_plies=args.max_plies,
                )
                output.write(str(game))
                output.write("\n\n")
                output.flush()
                outcome = network_result(result, network_is_white=game_index % 2 == 0)
                counts[outcome] += 1

    score = (counts["win"] + 0.5 * counts["draw"]) / args.games
    summary = {
        "checkpoint": str(args.checkpoint),
        "games": args.games,
        "simulations": args.simulations,
        "stockfish_elo": args.stockfish_elo,
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score": score,
    }
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"PGN: {args.out}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
