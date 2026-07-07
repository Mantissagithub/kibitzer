# two-panel ceiling figure for the s2 shaw 100m model.
# panel a (left): score vs leela net at 1/8/32 nodes, showing the model
# gets crushed as soon as leela does real search.
# panel b (right): honest strength across three independent opponents
# (maia, stockfish uci_elo, leela net) converging on ~2500-2600.
# data: reports/scaling_law/elo_leela/ceiling_figure_data.json
# run: uv run python scripts/plot_ceiling.py

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kibitzer-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 300,
    }
)


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_leela_ceiling(ax: plt.Axes, data: dict) -> None:
    rows = data["leela_ceiling"]
    xs = list(range(len(rows)))
    scores = [row["score"] for row in rows]
    labels = [f"{row['nodes']} (~{row['approx_elo']})" for row in rows]

    ax.plot(
        xs,
        scores,
        color="#e15759",
        linewidth=1.8,
        marker="o",
        markersize=7,
        zorder=3,
        label="measured score vs Leela net",
    )

    for x, row in zip(xs, rows):
        ax.annotate(
            row["wdl"],
            (x, row["score"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(-0.35, 0.5, "even (0.5)", va="bottom", ha="left", fontsize=8, color="#555555")

    ax.annotate(
        "~even only at 1 node;\ncrushed by any real search",
        xy=(0, rows[0]["score"]),
        xytext=(0.55, 0.62),
        fontsize=8,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.8},
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Leela net search budget: nodes (approx Elo)")
    ax.set_ylabel("model score (win=1, draw=½, loss=0)")
    ax.set_title("vs Leela net (our model, 256 sims)", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_multi_scale(ax: plt.Axes, data: dict) -> None:
    lo, hi = 1800, 2800

    ax.axvspan(2500, 2600, color="#59a14f", alpha=0.15, zorder=0)
    ax.text(
        2550,
        3.55,
        "~2500-2600 Elo\n(honest)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )

    # maia: a right-pointing floor arrow starting at 1900.
    ax.annotate(
        "",
        xy=(hi - 40, 3),
        xytext=(1900, 3),
        arrowprops={"arrowstyle": "-|>", "color": "#4e79a7", "linewidth": 2.2},
    )
    ax.text(1900, 3.18, "Maia (Lichess, human-calibrated)", fontsize=8.5, color="#4e79a7", va="bottom")
    ax.text(1900, 2.78, ">1900 (159.5/160, floor)", fontsize=8, color="#4e79a7", va="top")

    # stockfish uci_elo point.
    sf_elo = data["multi_scale"]["Stockfish UCI_Elo"]
    ax.scatter([sf_elo], [2], s=70, color="#e15759", zorder=3)
    ax.text(sf_elo, 2.18, "Stockfish UCI_Elo", fontsize=8.5, color="#e15759", ha="center", va="bottom")
    ax.text(sf_elo, 1.78, f"{sf_elo}", fontsize=8, color="#e15759", ha="center", va="top")

    # leela net implied elo point.
    leela_elo = data["implied_elo_from_leela1n"]
    ax.scatter([leela_elo], [1], s=70, color="#f28e2b", zorder=3)
    ax.text(leela_elo, 1.18, "Leela net (1 node)", fontsize=8.5, color="#f28e2b", ha="center", va="bottom")
    ax.text(leela_elo, 0.78, f"{leela_elo}", fontsize=8, color="#f28e2b", ha="center", va="top")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.3, 4.0)
    ax.set_yticks([])
    ax.set_xlabel("Elo (three independent rating pools)")
    ax.set_title("Converged estimate across 3 opponents", pad=10)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "reports" / "scaling_law" / "elo_leela" / "ceiling_figure_data.json"
    output_path = repo_root / "reports" / "scaling_law" / "fig_ceiling.png"

    data = load_data(data_path)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plot_leela_ceiling(ax_a, data)
    plot_multi_scale(ax_b, data)

    fig.suptitle(
        "Kibitzer 100M: honestly ~2500-2600 Elo (strong engine, ceiling <2700)",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.005,
        "measured vs human-calibrated Maia (Lichess), Stockfish UCI_Elo, and a Leela net at "
        "1/8/32 nodes; different rating pools, consistent ~2500-2600 picture; well short of 3000.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
