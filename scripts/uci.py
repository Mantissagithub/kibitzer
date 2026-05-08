"""UCI adapter for Kibitzer.

Implements the subset of UCI needed for cutechess-cli tournament play and for
``chess.engine.SimpleEngine.popen_uci``: ``uci``, ``setoption``, ``isready``,
``ucinewgame``, ``position``, ``go``, ``stop``, ``quit``. Search is synchronous
(one forward pass per ``go``), so ``stop`` is a no-op — by the time the host
can send it, we've already emitted ``bestmove``. Time-control fields on ``go``
(wtime/btime/winc/binc/movetime/depth/nodes/infinite) are accepted and ignored.

Options:
    Checkpoint  (string)         path to a .pt file; reloaded on next isready
    Temperature (string -> float) sampling temperature; 0.0 = greedy
    Device      (combo cuda|cpu) reloads the model on next isready

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
import time

import chess
import torch

from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer


ENGINE_NAME = "Kibitzer"
ENGINE_AUTHOR = "Pradheep"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_model(checkpoint: str | None, device: str) -> Kibitzer:
    model = Kibitzer()
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
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
                f"checkpoint at {checkpoint} did not match any known shape"
            )
        print(f"loaded checkpoint from {checkpoint}", file=sys.stderr)
    elif checkpoint:
        print(
            f"WARNING: checkpoint not found at {checkpoint} — "
            "using random-init weights",
            file=sys.stderr,
        )
    else:
        print(
            "WARNING: no checkpoint loaded — using random-init weights",
            file=sys.stderr,
        )
    return model


def build_engine(config: dict) -> KibitzerEngine:
    device = config["device"]
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_model(config["checkpoint"], device)
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


def parse_setoption(tokens: list[str]) -> tuple[str, str] | None:
    """Parse `setoption name <Name> value <Value>`. Value may contain spaces."""
    if len(tokens) < 5 or tokens[1] != "name":
        return None
    try:
        value_idx = tokens.index("value", 2)
    except ValueError:
        return None
    if value_idx <= 2:
        return None
    name = " ".join(tokens[2:value_idx])
    value = " ".join(tokens[value_idx + 1 :])
    return name, value


def handle_setoption(tokens: list[str], config: dict, state: dict) -> None:
    parsed = parse_setoption(tokens)
    if parsed is None:
        print(f"could not parse setoption: {' '.join(tokens)}", file=sys.stderr)
        return
    name, value = parsed
    if name == "Checkpoint":
        config["checkpoint"] = value or None
        state["dirty"] = True
    elif name == "Temperature":
        try:
            config["temperature"] = float(value)
        except ValueError:
            print(f"bad Temperature value: {value!r}", file=sys.stderr)
    elif name == "Device":
        if value not in ("cuda", "cpu"):
            print(f"bad Device value: {value!r}", file=sys.stderr)
            return
        config["device"] = value
        state["dirty"] = True
    else:
        print(f"unknown option: {name}", file=sys.stderr)


def handle_go(
    engine: KibitzerEngine, board: chess.Board, temperature: float
) -> None:
    """Run one forward pass on ``board``, emit info + bestmove."""
    saved = list(engine.history)
    t0 = time.perf_counter()
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
    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))

    # cp is conventionally centipawns; ours is the tanh value × 100 ("centi-tanh"),
    # clamped to ±2000 to match Stockfish-style non-mate output.
    cp = max(-2000, min(2000, int(round(value * 100))))
    say(
        f"info depth 1 score cp {cp} nodes 1 nps 1 "
        f"time {elapsed_ms} pv {move.uci()}"
    )
    say(f"bestmove {move.uci()}")


def main() -> None:
    args = parse_args()
    config = {
        "checkpoint": args.checkpoint,
        "temperature": args.temperature,
        "device": args.device or DEFAULT_DEVICE,
    }
    state = {"dirty": True}
    engine: KibitzerEngine | None = None
    board = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "quit":
            return

        try:
            if cmd == "uci":
                say(f"id name {ENGINE_NAME}")
                say(f"id author {ENGINE_AUTHOR}")
                say(
                    "option name Checkpoint type string default "
                    f"{config['checkpoint'] or ''}"
                )
                say(
                    "option name Temperature type string default "
                    f"{config['temperature']}"
                )
                say(
                    "option name Device type combo default "
                    f"{config['device']} var cuda var cpu"
                )
                say("uciok")
            elif cmd == "setoption":
                handle_setoption(tokens, config, state)
            elif cmd == "isready":
                if engine is None or state["dirty"]:
                    engine = build_engine(config)
                    state["dirty"] = False
                say("readyok")
            elif cmd == "ucinewgame":
                board = chess.Board()
                if engine is not None:
                    engine.reset()
            elif cmd == "position":
                new_board = parse_position(tokens)
                if new_board is None:
                    print(f"unparseable position: {line}", file=sys.stderr)
                else:
                    board = new_board
            elif cmd == "go":
                # Time-control tokens (wtime/btime/winc/binc/movetime/depth/
                # nodes/infinite) are accepted and ignored — we're synchronous.
                if engine is None or state["dirty"]:
                    engine = build_engine(config)
                    state["dirty"] = False
                handle_go(engine, board, config["temperature"])
            elif cmd == "stop":
                # Synchronous; bestmove was already emitted by the prior `go`.
                pass
            else:
                print(f"unrecognized: {line}", file=sys.stderr)
        except Exception as e:
            print(f"error handling '{line}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
