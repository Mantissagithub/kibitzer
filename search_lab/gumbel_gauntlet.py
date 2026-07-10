from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from search_lab.gumbel import gumbel_search
from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search
from scripts.maia_gauntlet import (
    OPENING_BOOK,
    elo_delta_from_score,
    format_elo,
    open_maia,
)


def _book_board(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for uci in rng.choice(OPENING_BOOK).split():
        board.push_uci(uci)
    return board


def _play_game(
    *,
    evaluator,
    engine,
    network_color,
    opening,
    search,
    simulations,
    maia_nodes,
    max_plies,
    max_actions,
    gumbel_scale,
    search_rng,
):
    board = opening
    game = chess.pgn.Game.from_board(board)
    node = game.end()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            if search == "gumbel":
                move = gumbel_search(
                    board,
                    evaluator,
                    simulations=simulations,
                    max_num_considered_actions=max_actions,
                    gumbel_scale=gumbel_scale,
                    rng=search_rng,
                ).move
            else:
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
    return game, outcome.result() if outcome is not None else "1/2-1/2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", choices=("puct", "gumbel"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--maia-weights", type=Path, required=True)
    parser.add_argument("--maia-elo", type=int, default=2700)
    parser.add_argument("--lc0-path", required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--maia-nodes", type=int, default=1)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--gumbel-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-pgn", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    opening_rng = random.Random(args.seed)
    search_rng = random.Random(args.seed + 1_000_003)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_jsonl.open("w", encoding="utf-8")
    pgn = args.out_pgn.open("w", encoding="utf-8")
    engine = open_maia(args.lc0_path, str(args.maia_weights), args.backend)
    wins = draws = losses = 0
    score_sum = 0.0
    started = time.time()

    print("============================================================", flush=True)
    print(f" KIBITZER {args.search.upper()} SEARCH GATE", flush=True)
    print("============================================================", flush=True)
    print(f"checkpoint:       {args.checkpoint}", flush=True)
    print(f"search:           {args.search}", flush=True)
    print(f"games/sims/seed:  {args.games} / {args.simulations} / {args.seed}", flush=True)
    if args.search == "gumbel":
        print(f"gumbel config:    max_actions={args.max_actions} scale={args.gumbel_scale:g}", flush=True)
    print(f"opponent:         Leela-{args.maia_elo} nodes={args.maia_nodes} backend={args.backend}", flush=True)
    print(f"jsonl:            {args.out_jsonl}", flush=True)
    print(f"pgn:              {args.out_pgn}", flush=True)
    print("", flush=True)

    try:
        for index in range(args.games):
            opening = _book_board(opening_rng)
            opening_fen = opening.fen()
            network_color = chess.WHITE if index % 2 == 0 else chess.BLACK
            game, result = _play_game(
                evaluator=evaluator,
                engine=engine,
                network_color=network_color,
                opening=opening,
                search=args.search,
                simulations=args.simulations,
                maia_nodes=args.maia_nodes,
                max_plies=args.max_plies,
                max_actions=args.max_actions,
                gumbel_scale=args.gumbel_scale,
                search_rng=search_rng,
            )
            if result == "1/2-1/2":
                score = 0.5
            else:
                score = 1.0 if (result == "1-0") == (network_color == chess.WHITE) else 0.0
            wins += score == 1.0
            draws += score == 0.5
            losses += score == 0.0
            score_sum += score

            game.headers["Event"] = f"{args.search} search vs Leela-{args.maia_elo}"
            game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else f"Leela-{args.maia_elo}"
            game.headers["Black"] = f"Leela-{args.maia_elo}" if network_color == chess.WHITE else "Kibitzer"
            game.headers["Result"] = result
            pgn.write(str(game) + "\n\n")
            pgn.flush()
            jsonl.write(json.dumps({
                "game": index + 1,
                "search": args.search,
                "opening_fen": opening_fen,
                "network_white": network_color == chess.WHITE,
                "result": result,
                "score": score,
            }) + "\n")
            jsonl.flush()

            played = wins + draws + losses
            elapsed = (time.time() - started) / 60.0
            eta = elapsed / played * (args.games - played)
            rate = score_sum / played
            elo_delta = elo_delta_from_score(rate)
            color = "white" if network_color == chess.WHITE else "black"
            print(
                f"[{args.search} {played}/{args.games}] as {color:<5} result={result:<7} "
                f"W/D/L={wins}/{draws}/{losses} score={score_sum:.1f} rate={rate:.3f} "
                f"elo={format_elo(args.maia_elo + elo_delta)} "
                f"elapsed={elapsed:.1f}m eta={eta:.1f}m",
                flush=True,
            )
    finally:
        engine.quit()
        jsonl.close()
        pgn.close()


if __name__ == "__main__":
    main()
