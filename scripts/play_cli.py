"""Interactive CLI for playing against Kibitzer or watching it play.

Examples:
    # Human plays white against a random-init engine (sanity check):
    uv run python scripts/play_cli.py --mode human

    # Human plays black, with a checkpoint:
    uv run python scripts/play_cli.py --mode human --black --checkpoint runs/best.pt

    # Watch Kibitzer play itself:
    uv run python scripts/play_cli.py --mode self --temp 0.5

    # Analyze a position:
    uv run python scripts/play_cli.py --mode analyze \\
        --fen 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["human", "self", "analyze"])
    p.add_argument("--checkpoint", default=None, help="state-dict checkpoint path")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--black", action="store_true", help="human plays black")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--fen", default=None, help="required for --mode analyze")
    args = p.parse_args()

    if args.mode == "analyze" and not args.fen:
        p.error("--mode analyze requires --fen")
    if args.black and args.mode != "human":
        p.error("--black is only meaningful with --mode human")
    return args


def load_engine(args: argparse.Namespace) -> KibitzerEngine:
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
                f"checkpoint at {args.checkpoint} did not match any known shape "
                f"(raw state_dict / dict['model'] / dict['state_dict'])"
            )
        print(f"loaded checkpoint from {args.checkpoint} (key: {loaded_from})")
    else:
        if args.checkpoint:
            print(
                f"WARNING: checkpoint not found at {args.checkpoint} — "
                "using random-init weights. Moves will be near-random.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: no checkpoint loaded — using random-init weights. "
                "Moves will be near-random.",
                file=sys.stderr,
            )

    print(f"device: {device}, dtype: {dtype}")
    return KibitzerEngine(model, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_board(board: chess.Board) -> None:
    print(board.unicode(invert_color=True, borders=True))
    print()


def print_eval(engine: KibitzerEngine, board: chess.Board, top_n: int = 3) -> None:
    out = engine.evaluate()
    print(f"  value: {out['value']:+.3f}")
    parts = []
    for m, p in out["move_probs"][:top_n]:
        parts.append(f"{board.san(m)} ({p * 100:.1f}%)")
    print(f"  top {top_n}: {', '.join(parts)}")


def print_outcome(board: chess.Board, plies: int, max_plies: int) -> None:
    if plies > max_plies and not board.is_game_over():
        print(f"\nDRAW: max plies ({max_plies}) reached.")
        return
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("\n(no outcome — game still in progress?)")
        return
    print(f"\nResult: {outcome.result()}  ({outcome.termination.name.lower()})")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def play_human(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    human_color = chess.BLACK if args.black else chess.WHITE
    print(f"You are {'BLACK' if args.black else 'WHITE'}. Enter moves in SAN (e.g. Nf3).\n")

    while True:
        board = engine.history[-1]
        if board.is_game_over() or len(engine.history) > args.max_plies:
            print_board(board)
            print_outcome(board, len(engine.history), args.max_plies)
            return

        print_board(board)

        if board.turn == human_color:
            try:
                raw = input("Your move: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(exiting)")
                return
            if not raw:
                continue
            try:
                move = board.parse_san(raw)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError) as e:
                print(f"  illegal/invalid: {e}")
                continue
            engine.push_move(move)
        else:
            print("Engine thinking...")
            print_eval(engine, board, top_n=3)
            move = engine.select_move(temperature=args.temp)
            print(f"Engine plays: {board.san(move)}\n")
            engine.push_move(move)


def play_self(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    while True:
        board = engine.history[-1]
        if board.is_game_over() or len(engine.history) > args.max_plies:
            print_board(board)
            print_outcome(board, len(engine.history), args.max_plies)
            return

        print_board(board)
        print_eval(engine, board, top_n=3)
        move = engine.select_move(temperature=args.temp)
        ply = len(engine.history)
        side = "W" if board.turn == chess.WHITE else "B"
        print(f"Move {ply} ({side}): {board.san(move)}\n")
        engine.push_move(move)
        time.sleep(0.4)


def analyze_mode(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    board = chess.Board(args.fen)
    engine.reset(board)

    print(f"FEN: {args.fen}")
    print_board(engine.history[-1])

    out = engine.evaluate()
    print(f"value: {out['value']:+.3f}")
    print(f"legal moves: {len(out['legal_moves'])}")
    print("\ntop 10:")
    for m, p in out["move_probs"][:10]:
        print(f"  {engine.history[-1].san(m):8s}  {p * 100:5.2f}%")

    pv = engine.get_principal_variation(depth=5)
    pv_board = engine.history[-1].copy(stack=False)
    sans: list[str] = []
    for m in pv:
        sans.append(pv_board.san(m))
        pv_board.push(m)
    print(f"\nPV (depth {len(pv)}): {' '.join(sans) if sans else '(none)'}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    engine = load_engine(args)
    if args.mode == "human":
        play_human(engine, args)
    elif args.mode == "self":
        play_self(engine, args)
    elif args.mode == "analyze":
        analyze_mode(engine, args)


if __name__ == "__main__":
    main()
