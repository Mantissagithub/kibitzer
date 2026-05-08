"""Round-trip verifier for scripts/uci.py.

Spawns the UCI adapter as a subprocess via python-chess's
``SimpleEngine.popen_uci``, plays 5 moves against it, and confirms every
response is well-formed (handshake completes, bestmove is legal in the
position it was played from).

Run:
    uv run python scripts/uci_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.engine


def main() -> None:
    uci_path = Path(__file__).parent / "uci.py"
    cmd = [sys.executable, str(uci_path)]

    with chess.engine.SimpleEngine.popen_uci(cmd) as engine:
        info = engine.id
        assert info.get("name", "").lower().startswith("kibitzer"), info
        print(f"engine id: {info}")

        board = chess.Board()
        for ply in range(5):
            result = engine.play(board, chess.engine.Limit(time=0.5))
            move = result.move
            assert move is not None, f"ply {ply + 1}: no bestmove returned"
            assert move in board.legal_moves, (
                f"ply {ply + 1}: illegal move {move} at {board.fen()}"
            )
            print(f"ply {ply + 1}: {board.san(move)}")
            board.push(move)

    print("uci_test: all 5 plies returned legal moves.")


if __name__ == "__main__":
    main()
