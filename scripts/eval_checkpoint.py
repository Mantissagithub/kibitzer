"""Evaluate a Kibitzer checkpoint against a baseline opponent.

Examples:
    # Random-init checkpoint vs random opponent (sanity check):
    uv run python scripts/eval_checkpoint.py \\
        --checkpoint /tmp/rand.pt --opponent random --n-games 4

    # Trained checkpoint vs Stockfish skill 0, depth 1:
    uv run python scripts/eval_checkpoint.py \\
        --checkpoint runs/best.pt --opponent stockfish \\
        --skill 0 --depth 1 --n-games 40 --output runs/best.eval.json
"""

from __future__ import annotations

import argparse
import json

import torch

from kibitzer.eval import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="random", choices=["random", "stockfish"])
    p.add_argument("--skill", type=int, default=0)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--output", default=None, help="optional JSON output path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        opponent=args.opponent,
        stockfish_skill=args.skill,
        stockfish_depth=args.depth,
        n_games=args.n_games,
        device=device,
        max_plies=args.max_plies,
        verbose=args.verbose,
    )

    print()
    for k in (
        "checkpoint",
        "opponent",
        "n_games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "approx_elo_diff",
        "avg_plies",
        "avg_value_pred",
    ):
        v = result[k]
        if isinstance(v, float):
            print(f"  {k:>17}: {v:+.4f}")
        else:
            print(f"  {k:>17}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
