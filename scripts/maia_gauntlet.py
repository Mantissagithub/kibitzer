# gauntlet vs maia: our model plays via puct, the opponent is lc0 loaded with a
# maia weight at 1 node (maia is calibrated to real lichess elo when played as the
# raw policy, i.e. nodes=1 / no search). real human-calibrated strength, unlike
# stockfish's uci_elo handicap. maia caps at 1900, so a dominant score means
# ">1900 lichess" (a real lower bound). outputs per-game jsonl + a pgn.

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search

OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6 f1b5", "e2e4 e7e5 g1f3 b8c6 f1c4", "e2e4 e7e5 g1f3 g8f6",
    "e2e4 c7c5 g1f3 d7d6 d2d4", "e2e4 c7c5 g1f3 b8c6", "e2e4 c7c5 b1c3 b8c6",
    "e2e4 e7e6 d2d4 d7d5 b1c3", "e2e4 c7c6 d2d4 d7d5 b1c3", "e2e4 g7g6 d2d4 f8g7",
    "d2d4 d7d5 c2c4 e7e6 b1c3", "d2d4 d7d5 c2c4 c7c6 g1f3", "d2d4 g8f6 c2c4 e7e6",
    "d2d4 g8f6 c2c4 g7g6 b1c3", "d2d4 g8f6 g1f3 e7e6", "d2d4 f7f5 g2g3 g8f6",
    "c2c4 e7e5 b1c3 g8f6", "c2c4 g8f6 b1c3 e7e6", "g1f3 d7d5 d2d4 g8f6 c2c4",
    "g1f3 g8f6 c2c4 g7g6 b1c3", "d2d4 d7d5 g1f3 g8f6 c2c4",
]


def book_board(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for uci in rng.choice(OPENING_BOOK).split():
        board.push_uci(uci)
    return board


def open_maia(lc0_path: str, weights: str, backend: str) -> chess.engine.SimpleEngine:
    # one lc0 process per shard; maia plays deterministically at nodes=1 so no
    # extra options are needed beyond the weight + a cpu backend.
    return chess.engine.SimpleEngine.popen_uci(
        [lc0_path, f"--weights={weights}", f"--backend={backend}"]
    )


def play_game(*, evaluator, engine, network_color, opening, simulations, maia_nodes, max_plies):
    board = opening
    game = chess.pgn.Game.from_board(board)
    node = game.end()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            move = puct_search(board, evaluator, simulations=simulations).move
        else:
            played = engine.play(board, chess.engine.Limit(nodes=maia_nodes))
            if played.move is None:
                raise RuntimeError("maia returned no move")
            move = played.move
        board.push(move)
        node = node.add_variation(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    return game, (outcome.result() if outcome is not None else "1/2-1/2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--maia-weights", type=Path, required=True)
    p.add_argument("--maia-elo", type=int, required=True, help="just a label for outputs")
    p.add_argument("--lc0-path", required=True)
    p.add_argument("--backend", default="blas")
    p.add_argument("--maia-nodes", type=int, default=1)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--simulations", type=int, default=256)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
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
    engine = open_maia(args.lc0_path, str(args.maia_weights), args.backend)
    try:
        for i in range(args.games):
            opening = book_board(rng)
            if opening.is_game_over(claim_draw=True):
                continue
            network_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            game, result = play_game(
                evaluator=evaluator, engine=engine, network_color=network_color,
                opening=opening, simulations=args.simulations, maia_nodes=args.maia_nodes,
                max_plies=args.max_plies,
            )
            if result == "1/2-1/2":
                score = 0.5
            else:
                score = 1.0 if (result == "1-0") == (network_color == chess.WHITE) else 0.0
            game.headers["Event"] = f"gauntlet vs Maia-{args.maia_elo}"
            game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else f"Maia-{args.maia_elo}"
            game.headers["Black"] = f"Maia-{args.maia_elo}" if network_color == chess.WHITE else "Kibitzer"
            game.headers["Result"] = result
            pgn_fh.write(str(game) + "\n\n"); pgn_fh.flush()
            jl.write(json.dumps({"maia_elo": args.maia_elo, "network_white": network_color == chess.WHITE,
                                 "result": result, "score": score}) + "\n"); jl.flush()
    finally:
        engine.quit(); jl.close(); pgn_fh.close()


if __name__ == "__main__":
    main()
