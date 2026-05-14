"""interactive cli for playing against kibitzer or watching it play (rich tui).

examples:
    # human plays white against a random-init engine (sanity check):
    uv run python scripts/play_cli.py --mode human

    # human plays black, with a checkpoint:
    uv run python scripts/play_cli.py --mode human --black --checkpoint runs/best.pt

    # watch kibitzer play itself:
    uv run python scripts/play_cli.py --mode self --temp 0.5

    # analyze a position:
    uv run python scripts/play_cli.py --mode analyze         --fen 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2b1p3/5n2/pppp1ppp/rnbqk2r w kqkq - 4 4'
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import torch
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer
from kibitzer import tui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["human", "self", "analyze"])
    p.add_argument("--checkpoint", default=None, help="state-dict checkpoint path")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--black", action="store_true", help="human plays black")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--fen", default=None, help="required for --mode analyze")
    p.add_argument("--no-tui", action="store_true",
                   help="disable rich TUI (auto-disabled when stdout isn't a TTY)")
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
                f"checkpoint at {args.checkpoint} did not match any known shape"
            )
        tui.console.print(f"[muted]loaded checkpoint from {args.checkpoint}[/]")
    elif args.checkpoint:
        tui.console.print(
            f"[warning]checkpoint not found at {args.checkpoint} — using random-init weights[/]"
        )
    else:
        tui.console.print("[warning]no checkpoint loaded — using random-init weights[/]")
    tui.console.print(f"[muted]device: {device}, dtype: {dtype}[/]")
    return KibitzerEngine(model, device=device, dtype=dtype)


def _make_layout(
    board: chess.Board,
    eval_out: dict,
    last_move: chess.Move | None,
    perspective: chess.Color,
    status: str | None = None,
    error: str | None = None,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=3),
        Layout(name="bottom", size=14),
    )
    layout["top"].split_row(
        Layout(tui.board_panel(board, last_move, perspective), name="board", ratio=1),
        Layout(
            tui.eval_panel(
                eval_out["value"],
                eval_out["move_probs"][:3],
                board,
            ),
            name="eval",
            ratio=1,
        ),
    )
    history = tui.move_history(board)
    bottom_items = [history]
    if status:
        bottom_items.append(Panel(Text(status, style="accent"), border_style="accent", padding=(0, 1)))
    if error:
        bottom_items.append(Panel(Text(error, style="error"), border_style="error", padding=(0, 1)))
    layout["bottom"].update(Group(*bottom_items))
    return layout


def _print_outcome(board: chess.Board, plies: int, max_plies: int) -> None:
    if plies > max_plies and not board.is_game_over():
        tui.console.print(f"[warning]DRAW: max plies ({max_plies}) reached.[/]")
        return
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        tui.console.print("[muted](no outcome — game still in progress)[/]")
        return
    style = "success" if outcome.winner is not None else "warning"
    tui.console.print(
        f"[{style}]Result: {outcome.result()}[/]  "
        f"[muted]({outcome.termination.name.lower()})[/]"
    )


def play_human_tui(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    human_color = chess.BLACK if args.black else chess.WHITE
    perspective = chess.BLACK if args.black else chess.WHITE
    last_move = None
    error: str | None = None

    tui.console.print(
        f"[accent]You are {'BLACK' if args.black else 'WHITE'}[/]. "
        "[muted]Enter moves in SAN (e.g. Nf3); Ctrl-D or Ctrl-C to quit.[/]"
    )

    while True:
        board = engine.history[-1]
        eval_out = engine.evaluate()

        if board.is_game_over() or len(engine.history) > args.max_plies:
            tui.console.print(_make_layout(board, eval_out, last_move, perspective))
            _print_outcome(board, len(engine.history), args.max_plies)
            return

        status = None
        if board.turn != human_color:
            status = "engine thinking..."
        tui.console.print(_make_layout(board, eval_out, last_move, perspective,
                                       status=status, error=error))
        error = None

        if board.turn == human_color:
            try:
                raw = tui.console.input("[accent]Your move:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                tui.console.print("[muted](exiting)[/]")
                return
            if not raw:
                continue
            try:
                move = board.parse_san(raw)
            except (chess.InvalidMoveError, chess.IllegalMoveError,
                    chess.AmbiguousMoveError, ValueError) as e:
                error = f"illegal/invalid: {e}"
                continue
            engine.push_move(move)
            last_move = move
        else:
            move = engine.select_move(temperature=args.temp)
            engine.push_move(move)
            last_move = move


def play_self_tui(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    last_move: chess.Move | None = None

    def render() -> Layout:
        board = engine.history[-1]
        eval_out = engine.evaluate()
        ply = len(engine.history)
        side = "white" if board.turn == chess.WHITE else "black"
        status = f"ply {ply} · {side} to move"
        return _make_layout(board, eval_out, last_move, chess.WHITE, status=status)

    with Live(render(), console=tui.console, refresh_per_second=4, screen=False) as live:
        while True:
            board = engine.history[-1]
            if board.is_game_over() or len(engine.history) > args.max_plies:
                live.update(render())
                break
            move = engine.select_move(temperature=args.temp)
            engine.push_move(move)
            last_move = move
            live.update(render())
            time.sleep(0.4)

    _print_outcome(engine.history[-1], len(engine.history), args.max_plies)


def analyze_tui(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    board = chess.Board(args.fen)
    engine.reset(board)
    eval_out = engine.evaluate()

    pv = engine.get_principal_variation(depth=5)
    pv_board = board.copy(stack=False)
    sans: list[str] = []
    for m in pv:
        sans.append(pv_board.san(m))
        pv_board.push(m)
    pv_text = " ".join(sans) if sans else "(none)"

    tui.console.print(tui.header("position analysis", args.fen))
    tui.console.print(_make_layout(board, eval_out, None, chess.WHITE,
                                   status=f"PV (depth {len(pv)}): {pv_text}"))

    extra = []
    extra.append(f"value: {eval_out['value']:+.3f}")
    extra.append(f"legal moves: {len(eval_out['legal_moves'])}")
    extra.append("\ntop 10:")
    for m, p in eval_out["move_probs"][:10]:
        extra.append(f"  {board.san(m):8s}  {p * 100:5.2f}%")
    tui.console.print(Panel(Text("\n".join(extra)), border_style="muted"))


def _print_board_plain(board: chess.Board) -> None:
    print(board.unicode(invert_color=True, borders=True))
    print()


def _print_eval_plain(engine: KibitzerEngine, board: chess.Board, top_n: int = 3) -> None:
    out = engine.evaluate()
    print(f"  value: {out['value']:+.3f}")
    parts = [f"{board.san(m)} ({p * 100:.1f}%)" for m, p in out["move_probs"][:top_n]]
    print(f"  top {top_n}: {', '.join(parts)}")


def _print_outcome_plain(board: chess.Board, plies: int, max_plies: int) -> None:
    if plies > max_plies and not board.is_game_over():
        print(f"\nDRAW: max plies ({max_plies}) reached.")
        return
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("\n(no outcome — game still in progress?)")
        return
    print(f"\nResult: {outcome.result()}  ({outcome.termination.name.lower()})")


def play_human_plain(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    human_color = chess.BLACK if args.black else chess.WHITE
    print(f"You are {'BLACK' if args.black else 'WHITE'}. Enter moves in SAN.\n")
    while True:
        board = engine.history[-1]
        if board.is_game_over() or len(engine.history) > args.max_plies:
            _print_board_plain(board)
            _print_outcome_plain(board, len(engine.history), args.max_plies)
            return
        _print_board_plain(board)
        if board.turn == human_color:
            try:
                raw = input("Your move: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not raw:
                continue
            try:
                move = board.parse_san(raw)
            except (chess.InvalidMoveError, chess.IllegalMoveError,
                    chess.AmbiguousMoveError, ValueError) as e:
                print(f"  illegal/invalid: {e}")
                continue
            engine.push_move(move)
        else:
            print("Engine thinking...")
            _print_eval_plain(engine, board, top_n=3)
            move = engine.select_move(temperature=args.temp)
            print(f"Engine plays: {board.san(move)}\n")
            engine.push_move(move)


def play_self_plain(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    engine.reset()
    while True:
        board = engine.history[-1]
        if board.is_game_over() or len(engine.history) > args.max_plies:
            _print_board_plain(board)
            _print_outcome_plain(board, len(engine.history), args.max_plies)
            return
        _print_board_plain(board)
        _print_eval_plain(engine, board, top_n=3)
        move = engine.select_move(temperature=args.temp)
        ply = len(engine.history)
        side = "W" if board.turn == chess.WHITE else "B"
        print(f"Move {ply} ({side}): {board.san(move)}\n")
        engine.push_move(move)
        time.sleep(0.4)


def analyze_plain(engine: KibitzerEngine, args: argparse.Namespace) -> None:
    board = chess.Board(args.fen)
    engine.reset(board)
    print(f"FEN: {args.fen}")
    _print_board_plain(engine.history[-1])
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


def main() -> None:
    args = parse_args()
    engine = load_engine(args)
    use_tui = not args.no_tui and tui.is_tty()
    if args.mode == "human":
        (play_human_tui if use_tui else play_human_plain)(engine, args)
    elif args.mode == "self":
        (play_self_tui if use_tui else play_self_plain)(engine, args)
    elif args.mode == "analyze":
        (analyze_tui if use_tui else analyze_plain)(engine, args)


if __name__ == "__main__":
    main()
