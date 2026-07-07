# two-panel elo comparison figure: base (100m) vs tactical mid-training (100m + 20m puzzle).
# panel a (left): measured score vs stockfish level for both models, showing near-overlap.
# panel b (right): the two measured elo values with overlapping cis, annotated as neutral.
# data: reports/scaling_law/elo_tactical/tactical_figure_data.json
# run: uv run python scripts/plot_elo_tactical.py

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

BASE_NAME = "base (100M)"
TACTICAL_NAME = "tactical (100M + 20M puzzle)"
BASE_COLOR = "#999999"
TACTICAL_COLOR = "#e15759"


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_score_vs_level(ax: plt.Axes, data: dict) -> None:
    levels = data["levels"]
    base = data["models"][BASE_NAME]
    tactical = data["models"][TACTICAL_NAME]

    base_scores = [base["per_level"][str(level)] for level in levels]
    tactical_scores = [tactical["per_level"][str(level)] for level in levels]

    # two lines+markers so the near-overlap between models is visible.
    ax.plot(
        levels,
        base_scores,
        color=BASE_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=7,
        zorder=3,
        label=f"{BASE_NAME}, {base['elo']} Elo",
    )
    ax.plot(
        levels,
        tactical_scores,
        color=TACTICAL_COLOR,
        linewidth=1.8,
        marker="s",
        markersize=7,
        zorder=3,
        label=f"{TACTICAL_NAME}, {tactical['elo']} Elo",
    )

    # even-score reference line.
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(levels[0] - 40, 0.5, "even (0.5)", va="bottom", ha="left", fontsize=8, color="#555555")

    ax.set_xlim(levels[0] - 100, levels[-1] + 100)
    ax.set_xticks(levels)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("opponent Stockfish level (Elo)")
    ax.set_ylabel("model score (win=1, draw=½, loss=0)")
    ax.set_title(f"PUCT (256 sims) vs Stockfish, {data['games_per_level']} games/level", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_elo_summary(ax: plt.Axes, data: dict) -> None:
    levels = data["levels"]
    base = data["models"][BASE_NAME]
    tactical = data["models"][TACTICAL_NAME]

    # stockfish level ladder as light reference gridlines.
    for level in levels:
        ax.axhline(level, color="#bbbbbb", linewidth=0.8, linestyle="-", zorder=1)
        ax.text(0.04, level, f"SF-{level}", va="bottom", ha="left", fontsize=8, color="#888888")

    # measured elo points with overlapping error bars.
    ax.errorbar(
        [0.35],
        [base["elo"]],
        yerr=[[base["ci"]], [base["ci"]]],
        fmt="o",
        color=BASE_COLOR,
        ecolor=BASE_COLOR,
        elinewidth=1.6,
        capsize=6,
        markersize=9,
        zorder=3,
        label=f"{BASE_NAME}: {base['elo']} ± {base['ci']}",
    )
    ax.errorbar(
        [0.65],
        [tactical["elo"]],
        yerr=[[tactical["ci"]], [tactical["ci"]]],
        fmt="o",
        color=TACTICAL_COLOR,
        ecolor=TACTICAL_COLOR,
        elinewidth=1.6,
        capsize=6,
        markersize=9,
        zorder=3,
        label=f"{TACTICAL_NAME}: {tactical['elo']} ± {tactical['ci']}",
    )

    # delta annotation between the two points.
    mid_elo = (base["elo"] + tactical["elo"]) / 2
    ax.annotate(
        "",
        xy=(0.65, tactical["elo"]),
        xytext=(0.35, base["elo"]),
        arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.8, "linestyle": ":"},
        zorder=2,
    )
    ax.text(
        0.5,
        mid_elo + 55,
        f"Δ = {data['delta_elo']} (n.s.)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.5,
        mid_elo + 30,
        "neutral — within CI",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([0.35, 0.65])
    ax.set_xticklabels(["base", "tactical"])
    ax.set_ylim(min(levels) - 100, max(levels) + 250)
    ax.set_ylabel("Elo")
    ax.set_title("Tactical mid-training: no change in Elo", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "reports" / "scaling_law" / "elo_tactical" / "tactical_figure_data.json"
    output_path = repo_root / "reports" / "scaling_law" / "fig_elo_tactical.png"

    data = load_data(data_path)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plot_score_vs_level(ax_a, data)
    plot_elo_summary(ax_b, data)

    fig.suptitle(
        "Tactical mid-training is neutral: 2464 vs 2483 Elo",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.005,
        "20M mixed puzzle+game mid-training on the 100M shaw model (value_weight=0); "
        "Elo via iterative fit over 160 games; the CIs overlap — no measurable strength change.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
