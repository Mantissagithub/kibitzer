# gauntlet vs mlabonne/LFM2.5-230M-Chess (a 230M HF causal LM, not a UCI engine).
# both sides play with search: kibitzer uses its own PUCT, LFM uses the shallow
# negamax alpha-beta search (depth/topK/rootTopK) from its official browser demo,
# which is what gets it to its reported ~2004 Elo (raw one-pass policy is weaker).
# outputs per-game jsonl + a pgn, same shape as scripts/maia_gauntlet.py.

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.pgn
import torch

from kibitzer.inference import ModelEvaluator
from kibitzer.lfm_chess import LFMChessPlayer, SearchSettings
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


def position_key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def play_game(
    *,
    evaluator,
    lfm: LFMChessPlayer,
    network_color,
    opening: chess.Board,
    simulations: int,
    lfm_settings: SearchSettings,
    max_plies: int,
    value_scale: float = 1.0,
    batch_size: int = 1,
):
    board = opening
    game = chess.pgn.Game.from_board(board)
    node = game.end()
    plies = 0
    kibitzer_moves = 0
    kibitzer_seconds = 0.0
    lfm_moves = 0
    lfm_seconds = 0.0

    # full-game move history (uci, both colors) and repetition counts, exactly
    # as the LFM demo builds them: last-8-ply history token window + a
    # 0/1/2-capped repeat count of the position about to be searched.
    history: list[str] = [move.uci() for move in board.move_stack]
    rep_counts: dict[str, int] = {}
    for i in range(len(board.move_stack) + 1):
        replay = opening.root().copy(stack=True)
        for move in board.move_stack[:i]:
            replay.push(move)
        key = position_key(replay)
        rep_counts[key] = rep_counts.get(key, 0) + 1

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            started = time.perf_counter()
            searched = puct_search(
                board, evaluator, simulations=simulations,
                value_scale=value_scale, batch_size=batch_size,
            )
            kibitzer_seconds += time.perf_counter() - started
            kibitzer_moves += 1
            move = searched.move
        else:
            key = position_key(board)
            repetition = rep_counts.get(key, 1) - 1
            started = time.perf_counter()
            move = lfm.search_move(board, history, settings=lfm_settings, repetition=repetition)
            lfm_seconds += time.perf_counter() - started
            lfm_moves += 1
        board.push(move)
        history.append(move.uci())
        rep_counts[position_key(board)] = rep_counts.get(position_key(board), 0) + 1
        node = node.add_variation(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    stats = {
        "kibitzer_moves": kibitzer_moves,
        "kibitzer_seconds": kibitzer_seconds,
        "kibitzer_seconds_per_move": kibitzer_seconds / kibitzer_moves if kibitzer_moves else 0.0,
        "lfm_moves": lfm_moves,
        "lfm_seconds": lfm_seconds,
        "lfm_seconds_per_move": lfm_seconds / lfm_moves if lfm_moves else 0.0,
    }
    return game, (outcome.result() if outcome is not None else "1/2-1/2"), stats


def elo_delta_from_score(score_rate: float) -> float:
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def format_elo(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.0f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="kibitzer .pt checkpoint")
    p.add_argument("--lfm-model-id", default="mlabonne/LFM2.5-230M-Chess")
    p.add_argument("--lfm-elo", type=int, default=2004, help="just a label for outputs")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--simulations", type=int, default=128, help="kibitzer PUCT sims/move")
    p.add_argument("--lfm-depth", type=int, default=3, help="0 = raw one-pass policy, no search")
    p.add_argument("--lfm-top-k", type=int, default=6)
    p.add_argument("--lfm-root-top-k", type=int, default=12)
    p.add_argument("--value-scale", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32, help="kibitzer leaf-parallel search batch")
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
    lfm = LFMChessPlayer(args.lfm_model_id, device=args.device)
    lfm_settings = SearchSettings(depth=args.lfm_depth, top_k=args.lfm_top_k, root_top_k=args.lfm_root_top_k)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jl = args.out_jsonl.open("w", encoding="utf-8")
    pgn_fh = args.out_pgn.open("w", encoding="utf-8")

    wins = draws = losses = 0
    score_sum = 0.0
    started = time.time()
    print("============================================================", flush=True)
    print(" KIBITZER vs LFM2.5-230M-Chess", flush=True)
    print("============================================================", flush=True)
    print(f"checkpoint:  {args.checkpoint}", flush=True)
    print(f"lfm model:   {args.lfm_model_id}  device={args.device}", flush=True)
    print(f"kibitzer:    {args.simulations} PUCT sims/move", flush=True)
    print(f"lfm search:  depth={args.lfm_depth} topK={args.lfm_top_k} rootTopK={args.lfm_root_top_k}", flush=True)
    print(f"games/seed:  {args.games} / {args.seed}", flush=True)
    print(f"jsonl:       {args.out_jsonl}", flush=True)
    print(f"pgn:         {args.out_pgn}", flush=True)
    print("", flush=True)

    paired_opening = None
    try:
        for i in range(args.games):
            if i % 2 == 0 or paired_opening is None:
                paired_opening = book_board(rng)
            opening = paired_opening.copy(stack=True)
            if opening.is_game_over(claim_draw=True):
                continue
            network_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            game_started = time.perf_counter()
            game, result, stats = play_game(
                evaluator=evaluator, lfm=lfm, network_color=network_color,
                opening=opening, simulations=args.simulations, lfm_settings=lfm_settings,
                max_plies=args.max_plies, value_scale=args.value_scale, batch_size=args.batch_size,
            )
            game_seconds = time.perf_counter() - game_started
            if result == "1/2-1/2":
                score = 0.5
            else:
                score = 1.0 if (result == "1-0") == (network_color == chess.WHITE) else 0.0
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
            score_sum += score

            game.headers["Event"] = f"gauntlet vs LFM2.5-230M-Chess-{args.lfm_elo}"
            game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else "LFM2.5-230M-Chess"
            game.headers["Black"] = "LFM2.5-230M-Chess" if network_color == chess.WHITE else "Kibitzer"
            game.headers["Result"] = result
            pgn_fh.write(str(game) + "\n\n")
            pgn_fh.flush()
            jl.write(json.dumps({
                "game": i + 1,
                "pair": i // 2 + 1,
                "network_white": network_color == chess.WHITE,
                "result": result,
                "score": score,
                "game_seconds": game_seconds,
                "stats": stats,
            }) + "\n")
            jl.flush()

            played = wins + draws + losses
            elapsed = (time.time() - started) / 60.0
            eta = elapsed / played * (args.games - played) if played else 0.0
            color = "white" if network_color == chess.WHITE else "black"
            rate = score_sum / played
            elo_delta = elo_delta_from_score(rate)
            print(
                f"[game {played}/{args.games}] as {color:<5} result={result:<7} "
                f"W/D/L={wins}/{draws}/{losses} score={score_sum:.1f} "
                f"rate={rate:.3f} elo_delta={format_elo(elo_delta)} "
                f"elo={format_elo(args.lfm_elo + elo_delta)} "
                f"kibitzer={stats['kibitzer_seconds_per_move']:.2f}s/mv "
                f"lfm={stats['lfm_seconds_per_move']:.2f}s/mv "
                f"elapsed={elapsed:.1f}m eta={eta:.1f}m",
                flush=True,
            )
        print("", flush=True)
        final_rate = score_sum / max(wins + draws + losses, 1)
        final_delta = elo_delta_from_score(final_rate)
        print(
            f"done: {wins}W/{draws}D/{losses}L score={score_sum:.1f}/{args.games} "
            f"rate={final_rate:.3f} elo_delta={format_elo(final_delta)} "
            f"elo={format_elo(args.lfm_elo + final_delta)}",
            flush=True,
        )
    finally:
        jl.close()
        pgn_fh.close()


if __name__ == "__main__":
    main()
