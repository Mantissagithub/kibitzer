# figures for the failure-analysis report. reads failures.jsonl (per-game records from
# analyze_failures.py) and renders four panels: acpl by phase, loss-shape split, blunder
# concentration by phase, and eval curves of a few representative losses. dark, print-ish.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import chess
import chess.engine
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#e8e6e3"
BG = "#14161a"
PANEL = "#1b1e24"
ACC = "#e0533d"      # blunder red
ACC2 = "#4d9de0"     # cool blue
GRID = "#2c313a"


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
    })


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def valid(games):
    return [g for g in games if g["valid"] and g["result"] in ("win", "draw", "loss")]


def fig_acpl_phase(games, ax):
    phases = ["opening", "middlegame", "endgame"]
    v = valid(games)
    means = []
    for ph in phases:
        vals = [g["acpl_by_phase"][ph] for g in v if g["acpl_by_phase"].get(ph) is not None]
        means.append(mean(vals) if vals else 0)
    bars = ax.bar(phases, means, color=[ACC2, ACC, "#c9a227"], width=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.5, f"{m:.0f}", ha="center", fontweight="bold")
    ax.set_ylabel("avg centipawn loss / move")
    ax.set_title("where the model bleeds (contested positions)")
    ax.grid(axis="y", alpha=0.3)


def fig_loss_shape(games, ax):
    losses = [g for g in valid(games) if g["result"] == "loss"]
    c = defaultdict(int)
    for g in losses:
        c[g["loss_shape"]] += 1
    labels = ["gradual", "sudden"]
    vals = [c.get("gradual", 0), c.get("sudden", 0)]
    ax.bar(labels, vals, color=[ACC2, ACC], width=0.55)
    for i, val in enumerate(vals):
        pct = 100 * val / max(1, sum(vals))
        ax.text(i, val + 0.3, f"{val}\n({pct:.0f}%)", ha="center", fontweight="bold")
    ax.set_ylabel("decisive losses")
    ax.set_title(f"how it loses (n={sum(vals)})")
    ax.grid(axis="y", alpha=0.3)


def fig_blunder_phase(games, ax):
    v = valid(games)
    c = defaultdict(int)
    for g in v:
        for b in g["blunders"]:
            c[b["phase"]] += 1
    phases = ["opening", "middlegame", "endgame"]
    vals = [c.get(p, 0) for p in phases]
    ax.bar(phases, vals, color=[ACC2, ACC, "#c9a227"], width=0.6)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.3, str(val), ha="center", fontweight="bold")
    ax.set_ylabel("blunders (cpl>=200)")
    ax.set_title("blunder count by phase")
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    style()
    games = load(args.jsonl)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Kibitzer failure taxonomy  (tactical_repair @ 512/128 sims vs Maia-2700 + SF ladder)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig_acpl_phase(games, axes[0])
    fig_loss_shape(games, axes[1])
    fig_blunder_phase(games, axes[2])
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
