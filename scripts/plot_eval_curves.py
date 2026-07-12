# eval-trajectory overlays for a few representative losses. re-scores just the named games
# move by move (stockfish, model pov) and plots the running eval vs ply, so the gradual
# "slow bleed" shape and the rarer sudden "cliff" shape are visible side by side.

from __future__ import annotations

import argparse

import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt

from analyze_failures import score_cp  # reuse the mate-ramp mapping

INK = "#e8e6e3"; BG = "#14161a"; PANEL = "#1b1e24"; GRID = "#2c313a"
GRAD = "#4d9de0"; SUDD = "#e0533d"


def curve(engine, game, limit):
    board = game.board()
    white = game.headers.get("White", "").startswith("Kibitzer")
    pov = chess.WHITE if white else chess.BLACK
    xs, ys = [], []
    for node in game.mainline():
        mv = node.move
        if board.turn == pov and not board.is_game_over():
            board.push(mv)
            s = score_cp(engine.analyse(board, limit)["score"], pov)
            xs.append(board.ply()); ys.append(max(-1200, min(1200, s)))
        else:
            board.push(mv)
    opp = game.headers.get("Black" if white else "White", "?")
    return xs, ys, opp


def nth_game(path, idx):
    with open(path) as f:
        for i in range(idx + 1):
            g = chess.pgn.read_game(f)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()
    limit = chess.engine.Limit(depth=args.depth)
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    # (pgn, game index, shape, short label)
    picks = [
        ("reports/sims_sweep/kibitzer_vs2700_s128_g40_seed23.pgn", 7, "gradual", "gradual: slow bleed vs Maia-2700"),
        ("reports/scaling_law/elo_tactical/eval_1900.pgn", 33, "gradual", "gradual: drift vs SF-1900"),
        ("reports/scaling_law/elo_tactical/eval_2300.pgn", 33, "sudden", "sudden: one throw vs SF-2300"),
        ("reports/scaling_law/elo_tactical/eval_1900.pgn", 2, "sudden", "sudden: one throw vs SF-1900"),
    ]

    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ax, (path, idx, shape, label) in zip(axes.flat, picks):
        g = nth_game(path, idx)
        xs, ys, opp = curve(engine, g, limit)
        col = SUDD if shape == "sudden" else GRAD
        ax.axhline(0, color=GRID, lw=1)
        ax.axhline(-150, color="#7a4b3a", lw=0.8, ls="--")
        ax.plot(xs, ys, color=col, lw=2.2, marker="o", ms=3)
        ax.fill_between(xs, ys, 0, where=[y < 0 for y in ys], color=col, alpha=0.12)
        ax.set_title(label, fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("ply"); ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Stockfish eval, model POV (cp)")
    axes[1, 0].set_ylabel("Stockfish eval, model POV (cp)")
    fig.suptitle("How Kibitzer loses: slow bleed (blue) vs rare cliff (red)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    engine.quit()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
