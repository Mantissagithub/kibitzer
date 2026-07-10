# D62 value_scale sweep figure: down-weighting the search's trust in the value
# head regresses play monotonically vs Leela-2700.
# panel a (left): score_rate vs value_scale, control line at value_scale=1.0.
# panel b (right): stacked W/D/L bars per value_scale setting, out of 40 games.
# data: hard-coded below (40 games each @128 sims, seed 23, vs Leela-2700).
# run: uv run python scripts/plot_value_scale_sweep.py

from __future__ import annotations

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
        "savefig.dpi": 180,
    }
)

# value_scale sweep, 40 games each @128 sims, seed 23, vs Leela-2700.
SWEEP = [
    {"value_scale": 1.0, "w": 6, "d": 11, "l": 23, "score_rate": 0.287, "elo": 2542},
    {"value_scale": 0.75, "w": 3, "d": 7, "l": 30, "score_rate": 0.163, "elo": 2415},
    {"value_scale": 0.5, "w": 2, "d": 1, "l": 37, "score_rate": 0.062, "elo": 2230},
]
GAMES_PER_RUN = 40

WIN_COLOR = "#59a14f"
DRAW_COLOR = "#bab0ac"
LOSS_COLOR = "#e15759"
LINE_COLOR = "#4e79a7"


def plot_score_vs_scale(ax: plt.Axes) -> None:
    scales = [row["value_scale"] for row in SWEEP]
    scores = [row["score_rate"] for row in SWEEP]
    control_score = SWEEP[0]["score_rate"]

    ax.axhline(control_score, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(
        0.45,
        control_score,
        f"control ({control_score:.3f})",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#555555",
    )

    ax.plot(
        scales,
        scores,
        color=LINE_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=8,
        zorder=3,
    )
    for row in SWEEP:
        label = f"{row['score_rate']:.3f}\n({row['elo']} Elo)"
        ax.annotate(
            label,
            xy=(row["value_scale"], row["score_rate"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )

    # mark the control point distinctly.
    ax.scatter(
        [SWEEP[0]["value_scale"]],
        [SWEEP[0]["score_rate"]],
        s=110,
        facecolors="none",
        edgecolors=LINE_COLOR,
        linewidths=1.6,
        zorder=4,
    )

    ax.set_xlim(0.42, 1.08)
    ax.set_xticks([row["value_scale"] for row in SWEEP])
    ax.set_ylim(0.0, 0.4)
    ax.set_xlabel("value_scale (search trust in the value head)")
    ax.set_ylabel("score rate vs Leela-2700 (40 games)")
    ax.set_title("less value → monotonically worse", pad=10)
    ax.text(
        0.5,
        0.02,
        "value head is load-bearing",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#333333",
    )


def plot_wdl_bars(ax: plt.Axes) -> None:
    labels = [f"{row['value_scale']:.2f}" for row in SWEEP]
    wins = [row["w"] for row in SWEEP]
    draws = [row["d"] for row in SWEEP]
    losses = [row["l"] for row in SWEEP]
    x = range(len(SWEEP))

    ax.bar(x, wins, color=WIN_COLOR, label="wins", zorder=3)
    ax.bar(x, draws, bottom=wins, color=DRAW_COLOR, label="draws", zorder=3)
    bottoms_wd = [w + d for w, d in zip(wins, draws)]
    ax.bar(x, losses, bottom=bottoms_wd, color=LOSS_COLOR, label="losses", zorder=3)

    for i, row in enumerate(SWEEP):
        ax.text(i, row["w"] / 2, str(row["w"]), ha="center", va="center", fontsize=8.5, color="white")
        ax.text(
            i,
            row["w"] + row["d"] / 2,
            str(row["d"]),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
        ax.text(
            i,
            row["w"] + row["d"] + row["l"] / 2,
            str(row["l"]),
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"value_scale={lab}" for lab in labels])
    ax.set_ylim(0, GAMES_PER_RUN)
    ax.set_ylabel(f"games (of {GAMES_PER_RUN})")
    ax.set_title("W/D/L collapse as value_scale drops", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "reports" / "value_scale_sweep" / "fig_value_scale_sweep.png"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plot_score_vs_scale(ax_a)
    plot_wdl_bars(ax_b)

    fig.suptitle(
        "D62: down-weighting the noisy value head in search regresses play\n"
        "(value head is load-bearing)",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.005,
        "PUCT value_scale sweep, 40 games each @128 sims, seed 23, vs Leela-2700; "
        "score rate and W/D/L both drop monotonically as value_scale falls from 1.0 to 0.5.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
