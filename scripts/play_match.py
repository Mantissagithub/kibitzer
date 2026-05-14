"""run a head-to-head match between kibitzer and a baseline opponent.

examples:
    # random-init kibitzer vs random opponent — sanity check the harness:
    uv run python scripts/play_match.py --opponent random --n-games 4

    # trained kibitzer vs stockfish skill 0, depth 1, 20 games:
    uv run python scripts/play_match.py         --checkpoint runs/best.pt         --opponent stockfish --skill 0 --depth 1 --n-games 20 --verbose

    # stockfish at fixed time per move:
    uv run python scripts/play_match.py         --opponent stockfish --time-ms 100 --n-games 10
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.match import play_match
from kibitzer.model import Kibitzer
from kibitzer.opponents import KibitzerOpponent, RandomOpponent, StockfishOpponent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--opponent", required=True, choices=["random", "stockfish"])
    p.add_argument("--stockfish-path", default=None)
    p.add_argument("--skill", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--time-ms", type=int, default=None)
    p.add_argument("--n-games", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-swap", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.opponent == "stockfish":
        if (args.depth is None) == (args.time_ms is None):
            p.error("--opponent stockfish requires exactly one of --depth / --time-ms")
    return args


def build_kibitzer_engine(args: argparse.Namespace) -> KibitzerEngine:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Kibitzer()
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        loaded_from = None
        for key in (None, "model", "state_dict"):
            sd = ckpt if key is None else ckpt.get(key)
            if sd is None:
                continue
            try:
                model.load_state_dict(sd)
                loaded_from = key or "<root>"
                break
            except (TypeError, RuntimeError):
                continue
        if loaded_from is None:
            raise RuntimeError(
                f"checkpoint at {args.checkpoint} did not match any known shape"
            )
        print(f"loaded checkpoint from {args.checkpoint} (key: {loaded_from})")
    else:
        if args.checkpoint:
            print(
                f"WARNING: checkpoint not found at {args.checkpoint} — "
                "using random-init weights",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: no checkpoint loaded — using random-init weights",
                file=sys.stderr,
            )

    print(f"device: {device}, dtype: {dtype}")
    return KibitzerEngine(model, device=device, dtype=dtype)


def print_results(opponent_name: str, n_games: int, result: dict) -> None:
    a = result["wins_a"]
    b = result["wins_b"]
    d = result["draws"]
    score_a = a + d / 2
    print(f"\nKibitzer (A) vs {opponent_name} (B) over {n_games} games:")
    print(f"    wins_a:  {a}")
    print(f"    wins_b:  {b}")
    print(f"    draws:   {d}")
    print(f"    score A: {score_a:.1f} / {n_games}")


def main() -> None:
    args = parse_args()
    engine = build_kibitzer_engine(args)
    kib_op = KibitzerOpponent(engine, temperature=args.temp)

    swap = not args.no_swap
    if args.opponent == "random":
        baseline = RandomOpponent(seed=args.seed)
        result = play_match(
            kib_op,
            baseline,
            n_games=args.n_games,
            max_plies=args.max_plies,
            swap_colors=swap,
            verbose=args.verbose,
        )
        print_results("random", args.n_games, result)
    else:
        with StockfishOpponent(
            path=args.stockfish_path,
            depth=args.depth,
            time_ms=args.time_ms,
            skill_level=args.skill,
        ) as sf:
            result = play_match(
                kib_op,
                sf,
                n_games=args.n_games,
                max_plies=args.max_plies,
                swap_colors=swap,
                verbose=args.verbose,
            )
        skill_str = (
            f" skill={args.skill}" if args.skill is not None else ""
        )
        limit_str = (
            f"depth={args.depth}" if args.depth is not None else f"time={args.time_ms}ms"
        )
        print_results(f"stockfish ({limit_str}{skill_str})", args.n_games, result)


if __name__ == "__main__":
    main()
