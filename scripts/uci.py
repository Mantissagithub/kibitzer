"""Minimal UCI adapter for Kibitzer.

Implements the subset of UCI needed for tournament play and for use with
``chess.engine.SimpleEngine.popen_uci``: ``uci``, ``isready``, ``ucinewgame``,
``position``, ``go``, ``stop``, ``quit``. Search is synchronous (one forward
pass per ``go``), so ``stop`` is a no-op — by the time the host can send it,
we've already emitted ``bestmove``. Time controls are accepted but ignored.

Examples:
    # Plug into cutechess / banksia / lichess-bot:
    uv run python scripts/uci.py --checkpoint runs/best.pt

    # Manual smoke test:
    printf 'uci\\nisready\\nposition startpos\\ngo movetime 100\\nquit\\n' \\
        | uv run python scripts/uci.py
"""

from __future__ import annotations

import argparse
import os
import sys

import chess
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer


ENGINE_NAME = "Kibitzer"
ENGINE_AUTHOR = "Pradheep"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def build_engine(args: argparse.Namespace) -> KibitzerEngine:
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
        print(f"loaded checkpoint from {args.checkpoint}", file=sys.stderr)
    elif args.checkpoint:
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

    return KibitzerEngine(model, device=device, dtype=dtype)


def say(line: str) -> None:
    """Write a UCI line to stdout. UCI hosts read line-by-line; flush always."""
    print(line, flush=True)


def parse_position(tokens: list[str]) -> chess.Board | None:
    """Parse a `position ...` command. Returns the new board, or None on error."""
    if len(tokens) < 2:
        return None

    if tokens[1] == "startpos":
        board = chess.Board()
        moves_idx = 2
    elif tokens[1] == "fen":
        if len(tokens) < 8:
            return None
        fen = " ".join(tokens[2:8])
        try:
            board = chess.Board(fen)
        except ValueError:
            return None
        moves_idx = 8
    else:
        return None

    if moves_idx < len(tokens):
        if tokens[moves_idx] != "moves":
            return None
        for uci in tokens[moves_idx + 1 :]:
            try:
                board.push_uci(uci)
            except ValueError:
                return None
    return board


def handle_go(
    engine: KibitzerEngine, board: chess.Board, temperature: float
) -> None:
    """Run one forward pass on ``board``, emit info + bestmove."""
    saved = list(engine.history)
    try:
        if board.move_stack:
            engine.reset(board.root())
            for m in board.move_stack:
                engine.push_move(m)
        else:
            engine.reset(board)

        eval_out = engine.evaluate()
        if temperature == 0.0:
            move = eval_out["move_probs"][0][0]
        else:
            move = engine.select_move(temperature=temperature)
        value = eval_out["value"]
    finally:
        engine.history = saved

    # `cp` is conventionally centipawns; ours is the tanh value × 100, which is
    # "centi-tanh" — fine for UCI display but not a real centipawn score.
    cp = int(round(value * 100))
    say(f"info depth 1 score cp {cp} pv {move.uci()}")
    say(f"bestmove {move.uci()}")


def main() -> None:
    args = parse_args()
    engine = build_engine(args)
    board = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            say(f"id name {ENGINE_NAME}")
            say(f"id author {ENGINE_AUTHOR}")
            say("uciok")
        elif cmd == "isready":
            say("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            new_board = parse_position(tokens)
            if new_board is None:
                print(f"unparseable position: {line}", file=sys.stderr)
            else:
                board = new_board
        elif cmd == "go":
            handle_go(engine, board, args.temperature)
        elif cmd == "stop":
            # We're synchronous; bestmove was already emitted by the prior `go`.
            pass
        elif cmd == "quit":
            return
        else:
            print(f"unrecognized: {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
