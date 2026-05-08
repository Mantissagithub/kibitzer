"""Evaluate a Kibitzer checkpoint via cutechess-cli (rich TUI).

Examples:
    # vs Stockfish at skill 3:
    uv run python scripts/eval_checkpoint.py \\
        --checkpoint runs/best.pt --opponent stockfish-3 \\
        --n-games 20 --output runs/best.eval.json

    # new vs previous checkpoint:
    uv run python scripts/eval_checkpoint.py \\
        --checkpoint runs/v3.pt --opponent self-vs-prev \\
        --prev-checkpoint runs/v2.pt --n-games 40

    # plain output (e.g. for CI logs):
    uv run python scripts/eval_checkpoint.py ... --no-tui
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.live import Live

from kibitzer.eval import evaluate_checkpoint
from kibitzer import tui


OPPONENTS = [
    "stockfish-0",
    "stockfish-3",
    "stockfish-5",
    "stockfish-10",
    "self-vs-prev",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="stockfish-0", choices=OPPONENTS)
    p.add_argument("--prev-checkpoint", default=None,
                   help="required when --opponent self-vs-prev")
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--time-per-move-ms", type=int, default=200)
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--cutechess-path", default="cutechess-cli")
    p.add_argument("--output", default=None,
                   help="optional JSON path to write the full result dict")
    p.add_argument("--no-tui", action="store_true",
                   help="disable rich TUI (auto-disabled when stdout isn't a TTY)")
    return p.parse_args()


def _run_plain(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        opponent=args.opponent,
        prev_checkpoint=args.prev_checkpoint,
        n_games=args.n_games,
        time_per_move_ms=args.time_per_move_ms,
        stockfish_path=args.stockfish_path,
        cutechess_path=args.cutechess_path,
    )


def _run_tui(args) -> dict:
    label_a = Path(args.checkpoint).stem
    label_b = (
        Path(args.prev_checkpoint).stem if args.opponent == "self-vs-prev"
        else args.opponent
    )
    state = tui.MatchState(n_games=args.n_games, label_a=label_a, label_b=label_b)

    tui.console.print(tui.header(
        f"{label_a}  vs  {label_b}",
        f"{args.n_games} games · st={args.time_per_move_ms}ms",
    ))

    def on_progress(update: dict) -> None:
        if "wins" in update:
            state.update(update["wins"], update["losses"], update["draws"])
        if "elo_diff" in update:
            state.elo_diff = update["elo_diff"]
            state.elo_err = update["elo_err"]
        live.update(tui.match_progress(state))

    with Live(
        tui.match_progress(state),
        console=tui.console,
        refresh_per_second=8,
        transient=False,
    ) as live:
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            opponent=args.opponent,
            prev_checkpoint=args.prev_checkpoint,
            n_games=args.n_games,
            time_per_move_ms=args.time_per_move_ms,
            stockfish_path=args.stockfish_path,
            cutechess_path=args.cutechess_path,
            on_progress=on_progress,
        )
        live.update(tui.match_progress(state))

    tui.console.print(tui.result_table(result))
    return result


def main() -> int:
    args = parse_args()
    use_tui = not args.no_tui and tui.is_tty()

    if use_tui:
        result = _run_tui(args)
    else:
        result = _run_plain(args)
        pct = 100 * result["win_rate"]
        print(
            f"{result['checkpoint']} vs {result['opponent']}: "
            f"{result['score']:.1f}/{result['n_games']} ({pct:.1f}%) "
            f"elo={result['elo_diff']:+.1f}+/-{result['elo_err']:.1f} "
            f"pgn={result['pgn_path']}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        if use_tui:
            tui.console.print(f"[muted]wrote {args.output}[/]")
        else:
            print(f"wrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
