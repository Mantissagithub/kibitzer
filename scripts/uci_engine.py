# minimal UCI engine wrapper around the model + puct search, so cutechess-cli (or
# any chess GUI) can drive it in a real tournament. `go` ignores the clock and always
# runs a fixed number of simulations (default 512) -- that's the whole point: we are
# rating the model *wrapped in 512-sim search*. options let cutechess set Sims etc.

from __future__ import annotations

import argparse
import sys

import chess
import torch

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


def parse_position(line: str) -> chess.Board:
    tokens = line.split()
    board = chess.Board()
    idx = len(tokens)
    if "startpos" in tokens:
        idx = tokens.index("startpos") + 1
    elif "fen" in tokens:
        fi = tokens.index("fen")
        board = chess.Board(" ".join(tokens[fi + 1: fi + 7]))
        idx = fi + 7
    if idx < len(tokens) and tokens[idx] == "moves":
        for mv in tokens[idx + 1:]:
            board.push_uci(mv)
    return board


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/tactical/tactical_repair.pt")
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sims, cpuct, vscale = args.sims, args.c_puct, args.value_scale
    evaluator: ModelEvaluator | None = None
    board = chess.Board()

    def send(s: str) -> None:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def ensure_loaded() -> None:
        nonlocal evaluator
        if evaluator is None:
            evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)

    for raw in sys.stdin:
        line = raw.strip()
        if line == "uci":
            send("id name Kibitzer-s%d" % sims)
            send("id author pradheep")
            send(f"option name Sims type spin default {sims} min 1 max 100000")
            send("option name CPuct type string default 1.5")
            send("option name ValueScale type string default 1.0")
            send("uciok")
        elif line == "isready":
            ensure_loaded()
            send("readyok")
        elif line.startswith("setoption"):
            t = line.split()
            if "name" in t and "value" in t:
                name = t[t.index("name") + 1].lower()
                val = t[t.index("value") + 1]
                if name == "sims":
                    sims = int(val)
                elif name == "cpuct":
                    cpuct = float(val)
                elif name == "valuescale":
                    vscale = float(val)
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = parse_position(line)
        elif line.startswith("go"):
            ensure_loaded()
            # cutechess owns draw claims. returning 0000 for a merely claimable draw
            # is reported as an illegal move and poisons the tournament PGN.
            if board.is_game_over(claim_draw=False):
                send("bestmove 0000")
            else:
                res = puct_search(
                    board,
                    evaluator,
                    simulations=sims,
                    c_puct=cpuct,
                    value_scale=vscale,
                    claim_draw=False,
                )
                send(f"bestmove {res.move.uci()}")
        elif line in ("quit", "stop"):
            if line == "quit":
                break


if __name__ == "__main__":
    main()
