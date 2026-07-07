# play one search variant against baseline_puct over the same net at an equal
# net-eval budget per move, from a fixed opening book. score > 0.5 means the
# variant raises the net's cap over vanilla puct. run per-variant so a bash
# launcher can parallelize the matchups.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess

from kibitzer.inference import ModelEvaluator
from variants import VARIANTS, CountingEvaluator

OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6",
    "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 d2d4 g8f6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]


def book_board(rng):
    board = chess.Board()
    for uci in rng.choice(OPENING_BOOK).split():
        board.push_uci(uci)
    return board


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--baseline", default="baseline_puct", choices=list(VARIANTS))
    p.add_argument("--budget", type=int, default=128, help="net evals per move")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    base_ev = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    ev = CountingEvaluator(base_ev)
    a_fn = VARIANTS[args.variant]
    b_fn = VARIANTS[args.baseline]

    w = d = l = 0
    ev_a = ev_b = moves_a = moves_b = 0
    for i in range(args.games):
        board = book_board(rng)
        a_white = i % 2 == 0  # variant A gets each color equally
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            is_a = (board.turn == chess.WHITE) == a_white
            ev.count = 0
            ev.budget = args.budget
            move = (a_fn if is_a else b_fn)(board, ev, args.budget)
            if is_a:
                ev_a += ev.count; moves_a += 1
            else:
                ev_b += ev.count; moves_b += 1
            board.push(move)
        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            d += 1
        elif (o.winner == chess.WHITE) == a_white:
            w += 1
        else:
            l += 1
        print(f"game {i+1}: {args.variant} {w}W/{d}D/{l}L score={(w+0.5*d)/(w+d+l):.3f}", flush=True)

    score = (w + 0.5 * d) / max(1, args.games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "variant": args.variant, "baseline": args.baseline, "budget": args.budget,
        "games": args.games, "w": w, "d": d, "l": l, "score": round(score, 4),
        "avg_evals_variant": round(ev_a / max(1, moves_a), 1),
        "avg_evals_baseline": round(ev_b / max(1, moves_b), 1),
    }, indent=2))
    print(f"{args.variant} vs {args.baseline}: {w}W/{d}D/{l}L score={score:.3f} "
          f"(avg evals {ev_a/max(1,moves_a):.0f} vs {ev_b/max(1,moves_b):.0f})")


if __name__ == "__main__":
    main()
