# gauntlet vs maia: our model plays via puct, the opponent is lc0 loaded with a
# maia weight at 1 node (maia is calibrated to real lichess elo when played as the
# raw policy, i.e. nodes=1 / no search). real human-calibrated strength, unlike
# stockfish's uci_elo handicap. maia caps at 1900, so a dominant score means
# ">1900 lichess" (a real lower bound). outputs per-game jsonl + a pgn.

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from kibitzer.inference import ModelEvaluator
from kibitzer.search import adaptive_puct_search, puct_search

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


def parse_stages(value: str) -> tuple[int, ...]:
    stages = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not stages or stages[0] < 1 or any(a >= b for a, b in zip(stages, stages[1:])):
        raise argparse.ArgumentTypeError("stages must be strictly increasing positive integers")
    return stages


def open_maia(lc0_path: str, weights: str, backend: str) -> chess.engine.SimpleEngine:
    # one lc0 process per shard; maia plays deterministically at nodes=1 so no
    # extra options are needed beyond the weight + a cpu backend.
    return chess.engine.SimpleEngine.popen_uci(
        [lc0_path, f"--weights={weights}", f"--backend={backend}"]
    )


def play_game(
    *,
    evaluator,
    engine,
    network_color,
    opening,
    simulations,
    maia_nodes,
    max_plies,
    value_scale=1.0,
    adaptive_stages=None,
    entropy_threshold=0.55,
    top2_ratio_threshold=0.75,
    value_delta_threshold=0.10,
):
    board = opening
    game = chess.pgn.Game.from_board(board)
    node = game.end()
    plies = 0
    search_moves = 0
    search_seconds = 0.0
    total_simulations = 0
    stage_counts: dict[int, int] = {}
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            search_started = time.perf_counter()
            if adaptive_stages is None:
                searched = puct_search(
                    board,
                    evaluator,
                    simulations=simulations,
                    value_scale=value_scale,
                )
            else:
                searched = adaptive_puct_search(
                    board,
                    evaluator,
                    stages=adaptive_stages,
                    entropy_threshold=entropy_threshold,
                    top2_ratio_threshold=top2_ratio_threshold,
                    value_delta_threshold=value_delta_threshold,
                    value_scale=value_scale,
                )
            search_seconds += time.perf_counter() - search_started
            search_moves += 1
            total_simulations += searched.simulations
            stage_counts[searched.simulations] = stage_counts.get(searched.simulations, 0) + 1
            move = searched.move
        else:
            played = engine.play(board, chess.engine.Limit(nodes=maia_nodes))
            if played.move is None:
                raise RuntimeError("maia returned no move")
            move = played.move
        board.push(move)
        node = node.add_variation(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    stats = {
        "moves": search_moves,
        "total_simulations": total_simulations,
        "avg_simulations": total_simulations / search_moves if search_moves else 0.0,
        "seconds": search_seconds,
        "seconds_per_move": search_seconds / search_moves if search_moves else 0.0,
        "stage_counts": {str(stage): count for stage, count in sorted(stage_counts.items())},
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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--maia-weights", type=Path, required=True)
    p.add_argument("--maia-elo", type=int, required=True, help="just a label for outputs")
    p.add_argument("--lc0-path", required=True)
    p.add_argument("--backend", default="eigen")
    p.add_argument("--maia-nodes", type=int, default=1)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--simulations", type=int, default=256)
    p.add_argument("--adaptive-stages", type=parse_stages, default=None)
    p.add_argument("--entropy-threshold", type=float, default=0.55)
    p.add_argument("--top2-ratio-threshold", type=float, default=0.75)
    p.add_argument("--value-delta-threshold", type=float, default=0.10)
    p.add_argument("--value-scale", type=float, default=1.0, help="puct value-backup weight; <1 trusts the noisy value head less")
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
    wins = 0
    draws = 0
    losses = 0
    score_sum = 0.0
    started = time.time()
    print("============================================================", flush=True)
    print(" KIBITZER MAIA/LEELA EXTERNAL GATE", flush=True)
    print("============================================================", flush=True)
    print(f"checkpoint:       {args.checkpoint}", flush=True)
    print(f"maia weights:     {args.maia_weights}", flush=True)
    print(f"lc0:              {args.lc0_path} backend={args.backend} nodes={args.maia_nodes}", flush=True)
    print(f"games/sims/seed:  {args.games} / {args.simulations} / {args.seed}", flush=True)
    if args.adaptive_stages is not None:
        print(f"adaptive stages:  {','.join(str(x) for x in args.adaptive_stages)}", flush=True)
        print(
            f"uncertainty:      entropy>={args.entropy_threshold:g} "
            f"top2_ratio>={args.top2_ratio_threshold:g} value_delta>={args.value_delta_threshold:g}",
            flush=True,
        )
    print(f"jsonl:            {args.out_jsonl}", flush=True)
    print(f"pgn:              {args.out_pgn}", flush=True)
    print("", flush=True)
    paired_opening = None
    total_search_moves = 0
    total_search_simulations = 0
    total_search_seconds = 0.0
    all_stage_counts: dict[str, int] = {}
    try:
        for i in range(args.games):
            # one opening is reused with colors swapped, so fixed and adaptive runs
            # can be compared game-for-game instead of leaning on opening luck.
            if i % 2 == 0 or paired_opening is None:
                paired_opening = book_board(rng)
            opening = paired_opening.copy(stack=True)
            if opening.is_game_over(claim_draw=True):
                continue
            network_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            game_started = time.perf_counter()
            game, result, search_stats = play_game(
                evaluator=evaluator, engine=engine, network_color=network_color,
                opening=opening, simulations=args.simulations, maia_nodes=args.maia_nodes,
                max_plies=args.max_plies, value_scale=args.value_scale,
                adaptive_stages=args.adaptive_stages,
                entropy_threshold=args.entropy_threshold,
                top2_ratio_threshold=args.top2_ratio_threshold,
                value_delta_threshold=args.value_delta_threshold,
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
            total_search_moves += search_stats["moves"]
            total_search_simulations += search_stats["total_simulations"]
            total_search_seconds += search_stats["seconds"]
            for stage, count in search_stats["stage_counts"].items():
                all_stage_counts[stage] = all_stage_counts.get(stage, 0) + count
            game.headers["Event"] = f"gauntlet vs Maia-{args.maia_elo}"
            game.headers["White"] = "Kibitzer" if network_color == chess.WHITE else f"Maia-{args.maia_elo}"
            game.headers["Black"] = f"Maia-{args.maia_elo}" if network_color == chess.WHITE else "Kibitzer"
            game.headers["Result"] = result
            pgn_fh.write(str(game) + "\n\n"); pgn_fh.flush()
            jl.write(json.dumps({
                "game": i + 1,
                "pair": i // 2 + 1,
                "maia_elo": args.maia_elo,
                "maia_nodes": args.maia_nodes,
                "network_white": network_color == chess.WHITE,
                "result": result,
                "score": score,
                "game_seconds": game_seconds,
                "search": search_stats,
            }) + "\n"); jl.flush()
            played = wins + draws + losses
            elapsed = (time.time() - started) / 60.0
            eta = elapsed / played * (args.games - played) if played else 0.0
            color = "white" if network_color == chess.WHITE else "black"
            rate = score_sum / played
            elo_delta = elo_delta_from_score(rate)
            print(
                f"[gate {played}/{args.games}] as {color:<5} result={result:<7} "
                f"W/D/L={wins}/{draws}/{losses} score={score_sum:.1f} "
                f"rate={rate:.3f} elo_delta={format_elo(elo_delta)} "
                f"elo={format_elo(args.maia_elo + elo_delta)} "
                f"search={search_stats['avg_simulations']:.0f}sims/{search_stats['seconds_per_move']:.2f}s "
                f"elapsed={elapsed:.1f}m eta={eta:.1f}m",
                flush=True,
            )
        print("", flush=True)
        final_rate = score_sum / max(wins + draws + losses, 1)
        final_delta = elo_delta_from_score(final_rate)
        print(
            f"done: {wins}W/{draws}D/{losses}L score={score_sum:.1f}/{args.games} "
            f"rate={final_rate:.3f} elo_delta={format_elo(final_delta)} "
            f"elo={format_elo(args.maia_elo + final_delta)}",
            flush=True,
        )
        avg_simulations = total_search_simulations / max(total_search_moves, 1)
        seconds_per_move = total_search_seconds / max(total_search_moves, 1)
        print(
            f"search budget: {total_search_moves} moves avg={avg_simulations:.1f} sims "
            f"time={seconds_per_move:.3f}s/move stages={dict(sorted(all_stage_counts.items()))}",
            flush=True,
        )
    finally:
        engine.quit(); jl.close(); pgn_fh.close()


if __name__ == "__main__":
    main()
