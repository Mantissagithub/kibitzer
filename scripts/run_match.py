"""Run a cutechess-cli match between Kibitzer and a baseline.

Examples:
    # Kibitzer vs Stockfish at skill 0, fast time control:
    uv run python scripts/run_match.py \\
        --checkpoint runs/best.pt --vs-random-stockfish \\
        --n-games 20 --time-per-move-ms 200

    # Kibitzer vs Stockfish at a specific skill level:
    uv run python scripts/run_match.py \\
        --checkpoint runs/best.pt --vs-stockfish-skill 5 \\
        --n-games 40 --tc 40/60+0.6 --concurrency 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kibitzer.cutechess_runner import run_match


KIBITZER_DEFAULT_CMD = f"{sys.executable} {Path(__file__).parent / 'uci.py'}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--engine-a",
        default=KIBITZER_DEFAULT_CMD,
        help="full command for engine A (default: this repo's UCI adapter)",
    )
    p.add_argument(
        "--engine-b",
        default=None,
        help="full command for engine B (default: 'stockfish' on PATH)",
    )
    p.add_argument("--checkpoint", default=None,
                   help="path to a Kibitzer checkpoint, applied as A's Checkpoint option")
    p.add_argument("--temperature", default=None,
                   help="A's Temperature option (UCI string)")

    presets = p.add_mutually_exclusive_group()
    presets.add_argument("--vs-random-stockfish", action="store_true",
                         help="set B to stockfish at Skill Level 0")
    presets.add_argument("--vs-stockfish-skill", type=int, default=None,
                         metavar="N",
                         help="set B to stockfish at Skill Level N (0-20)")

    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--time-per-move-ms", type=int, default=1000)
    p.add_argument("--tc", default=None,
                   help="full time control (e.g. '40/60+0.6'); overrides --time-per-move-ms")
    p.add_argument("--opening-book", default=None,
                   help="path to a .pgn or .epd opening book")
    p.add_argument("--pgn-output", default="match.pgn")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--cutechess-path", default=None,
                   help="explicit path to cutechess-cli (else resolved via PATH)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    a_options: dict[str, str] = {}
    if args.checkpoint:
        a_options["Checkpoint"] = args.checkpoint
    if args.temperature is not None:
        a_options["Temperature"] = str(args.temperature)

    b_cmd = args.engine_b
    b_options: dict[str, str] = {}
    if args.vs_random_stockfish:
        b_cmd = b_cmd or "stockfish"
        b_options["Skill Level"] = "0"
    elif args.vs_stockfish_skill is not None:
        b_cmd = b_cmd or "stockfish"
        b_options["Skill Level"] = str(args.vs_stockfish_skill)
    elif b_cmd is None:
        b_cmd = "stockfish"

    result = run_match(
        engine_a_cmd=args.engine_a,
        engine_b_cmd=b_cmd,
        engine_a_options=a_options or None,
        engine_b_options=b_options or None,
        n_games=args.n_games,
        time_per_move_ms=args.time_per_move_ms,
        tc=args.tc,
        opening_book=args.opening_book,
        pgn_output=args.pgn_output,
        concurrency=args.concurrency,
        cutechess_path=args.cutechess_path,
    )

    print()
    print(f"wins:    {result['wins']}")
    print(f"losses:  {result['losses']}")
    print(f"draws:   {result['draws']}")
    print(f"elo:     {result['elo_diff']:+.1f} +/- {result['elo_err']:.1f}")
    print(f"pgn:     {result['pgn_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
