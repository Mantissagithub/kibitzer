"""Generate reports/search_depth/fig1_search_depth_curve.png from results.json.

Reads reports/search_depth/results.json ("primary_runs_n20") and plots score vs.
Stockfish-1320 across MCTS simulation counts for the two value-head checkpoints,
with +/-0.09 (N=20 standard-error) bars on every point.

Usage:
    .venv/bin/python scripts/plot_search_depth_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 150,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "reports" / "search_depth" / "results.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "search_depth" / "fig1_search_depth_curve.png"

SE_N20 = 0.09

# Fixed categorical slot order (dataviz skill palette), not cycled.
COLOR_VALUE_FINAL = "#2a78d6"  # slot 1: blue
COLOR_JOINT_SCRATCH = "#1baf7a"  # slot 2: aqua


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    runs = payload["primary_runs_n20"]

    value_final = sorted(
        (r["sims"], r["score"]) for r in runs if r["checkpoint"] == "value_final"
    )
    joint_scratch = sorted(
        (r["sims"], r["score"]) for r in runs if r["checkpoint"] == "joint_scratch"
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(0.5, color="#8a8a86", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(
        64 * 0.97,
        0.5 + 0.015,
        "parity vs Stockfish-1320",
        color="#5a5a56",
        fontsize=8.5,
        ha="left",
        va="bottom",
    )

    series = [
        ("value_final (cp value head, SFT policy)", value_final, COLOR_VALUE_FINAL, "o"),
        ("joint_scratch (±1 value head)", joint_scratch, COLOR_JOINT_SCRATCH, "s"),
    ]

    # A small multiplicative x-dodge keeps series visually separated on the log
    # axis at sims=64, where both scores coincide exactly at 0.100.
    dodge = {"value_final (cp value head, SFT policy)": 0.94, "joint_scratch (±1 value head)": 1.06}

    for label, points, color, marker in series:
        xs = [p[0] * dodge[label] for p in points]
        ys = [p[1] for p in points]
        ax.errorbar(
            xs,
            ys,
            yerr=SE_N20,
            marker=marker,
            markersize=8,
            linewidth=2,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=4,
            capthick=1.3,
            label=label,
            zorder=3,
        )
        for x, y in zip(xs, ys):
            offset = 12 if label.startswith("value_final") else -18
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256, 512])
    ax.set_xticklabels(["64", "256", "512"])
    ax.set_xlim(48, 700)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("MCTS simulations per move (log scale)")
    ax.set_ylabel("score vs. Stockfish-1320 (W + 0.5×D) / games")

    fig.suptitle(
        "Search depth gives a one-shot lift, then plateaus", fontsize=13, fontweight="bold", y=0.98
    )
    fig.text(
        0.5,
        0.915,
        "64→256 sims lifts score; 256→512 is flat within noise — the two value heads\n"
        "are statistically indistinguishable at every simulation count (±0.09 SE)",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )

    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.annotate(
        "error bars: ±0.09 SE (N=20 games/point)",
        xy=(0.98, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    fig.text(
        0.01,
        0.01,
        "source: reports/search_depth/results.json — N=20 games/point, "
        "Stockfish UCI_LimitStrength Elo-1320 floor",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#444444",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.85))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
