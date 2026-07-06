# parallel gauntlet shard: play one stockfish level for N games from randomized
# openings (so parallel shards give distinct games, not replays), model via puct.
# writes one jsonl line per game (level, color, result, score) + a pgn of every
# game. a launcher runs many of these in parallel across levels/seeds; elo is
# fit afterward from the combined jsonl. built to be scp'd to a rented pod.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


def random_opening(rng: random.Random, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(moves))
    return board


def play_game(
    *,
    evaluator: ModelEvaluator,
    engine: chess.engine.SimpleEngine,
    network_color: bool,
    opening: chess.Board,
    simulations: int,
    stockfish_time: float,
    max_plies: int,
) -> tuple[chess.pgn.Game, str]:
    board = opening
    game = chess.pgn.Game.from_board(board)
    node = game.end()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            move = puct_search(board, evaluator, simulations=simulations).move
        else:
            played = engine.play(board, chess.engine.Limit(time=stockfish_time))
            if played.move is None:
                raise RuntimeError("stockfish returned no move")
            move = played.move
        board.push(move)
        node = node.add_variation(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome is not None else "1/2-1/2"
    return game, result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stockfish-elo", type=int, required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=256)
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--stockfish-time", type=float, default=0.05)
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=0, help="distinct per shard -> distinct games")
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-pgn", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jl = args.out_jsonl.open("w", encoding="utf-8")
    pgn_fh = args.out_pgn.open("w", encoding="utf-8")

    with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
        for i in range(args.games):
            opening = random_opening(rng, args.opening_plies)
            if opening.is_game_over(claim_draw=True):
                continue
            network_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            game, result = play_game(
                evaluator=evaluator,
                engine=engine,
                network_color=network_color,
                opening=opening,
                simulations=args.simulations,
                stockfish_time=args.stockfish_time,
                max_plies=args.max_plies,
            )
            # score from the model's pov
            if result == "1/2-1/2":
                score = 0.5
            else:
                score = 1.0 if (result == "1-0") == (network_color == chess.WHITE) else 0.0
            game.headers["Event"] = f"gauntlet vs SF-{args.stockfish_elo}"
            game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else f"SF-{args.stockfish_elo}"
            game.headers["Black"] = f"SF-{args.stockfish_elo}" if network_color == chess.WHITE else "Kibitzer"
            game.headers["Result"] = result
            pgn_fh.write(str(game) + "\n\n")
            pgn_fh.flush()
            jl.write(json.dumps({
                "opp_elo": args.stockfish_elo,
                "network_white": network_color == chess.WHITE,
                "result": result,
                "score": score,
                "plies": len(list(game.mainline_moves())),
            }) + "\n")
            jl.flush()
    jl.close()
    pgn_fh.close()


if __name__ == "__main__":
    main()
