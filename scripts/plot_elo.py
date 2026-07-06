# two-panel elo result figure for the s2 shaw 100m model.
# panel a (left): measured score vs stockfish level with the fitted logistic
# expected-score curve overlaid, validating the elo fit.
# panel b (right): the single measured elo value (2483 +/- 32) against the
# stockfish level ladder as reference.
# data: reports/scaling_law/elo_local/elo_figure_data.json
# run: uv run python scripts/plot_elo.py

from __future__ import annotations

import json
import math
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


def expected_score(opp_elo: float, model_elo: float) -> float:
    # standard elo logistic expected score for the model vs an opponent.
    return 1.0 / (1.0 + 10 ** ((opp_elo - model_elo) / 400.0))


def plot_score_curve(ax: plt.Axes, data: dict) -> None:
    per_level = data["per_level"]
    model_elo = data["final_elo"]

    opp_elos = [row["opp_elo"] for row in per_level]
    scores = [row["score"] for row in per_level]

    # fitted logistic curve, smooth across the opponent elo range.
    xs = [1800 + i * (2600 - 1800) / 200 for i in range(201)]
    ys = [expected_score(x, model_elo) for x in xs]
    ax.plot(
        xs,
        ys,
        color="#1f77b4",
        linewidth=1.8,
        zorder=2,
        label=f"fitted logistic (model = {model_elo} Elo)",
    )

    # measured per-level scores.
    ax.scatter(
        opp_elos,
        scores,
        s=45,
        color="#e15759",
        zorder=3,
        label="measured score (40 games/level)",
    )
    for row in per_level:
        ax.annotate(
            f"{row['w']}W/{row['d']}D/{row['l']}L",
            (row["opp_elo"], row["score"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )

    # even-score reference line.
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(1810, 0.5, "even (0.5)", va="bottom", ha="left", fontsize=8, color="#555555")

    # annotate the crossing point near sf-2500.
    ax.annotate(
        "crosses even ~SF-2500",
        xy=(2500, 0.525),
        xytext=(2320, 0.30),
        fontsize=8,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": 0.8},
    )

    ax.set_xlim(1800, 2600)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("opponent Stockfish level (Elo)")
    ax.set_ylabel("model score (win=1, draw=½, loss=0)")
    ax.set_title("PUCT (256 sims) vs Stockfish, 40 games/level", pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_elo_summary(ax: plt.Axes, data: dict) -> None:
    final_elo = data["final_elo"]
    ci = data["elo_ci"]
    levels = [row["opp_elo"] for row in data["per_level"]]

    # stockfish level ladder as light reference gridlines.
    for level in levels:
        ax.axhline(level, color="#bbbbbb", linewidth=0.8, linestyle="-", zorder=1)
        ax.text(0.04, level, f"SF-{level}", va="bottom", ha="left", fontsize=8, color="#888888")

    # measured elo point with error bar.
    ax.errorbar(
        [0.5],
        [final_elo],
        yerr=[[ci], [ci]],
        fmt="o",
        color="#e15759",
        ecolor="#e15759",
        elinewidth=1.6,
        capsize=6,
        markersize=9,
        zorder=3,
    )
    ax.text(
        0.5,
        final_elo + ci + 25,
        f"{final_elo} ± {ci} Elo",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#333333",
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(min(levels) - 100, max(levels) + 250)
    ax.set_ylabel("Elo")
    ax.set_title("Measured performance Elo (iterative fit, 160 games)", pad=10)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "reports" / "scaling_law" / "elo_local" / "elo_figure_data.json"
    output_path = repo_root / "reports" / "scaling_law" / "fig_elo.png"

    data = load_data(data_path)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plot_score_curve(ax_a, data)
    plot_elo_summary(ax_b, data)

    fig.suptitle(
        "Kibitzer S2 shaw (100M): measured ~2483 Elo vs Stockfish",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.005,
        "15.2M-param attention-first transformer, shaw encoding, 100M Lichess-Elite positions; "
        "Elo via iterative R+=K(S-E) over 160 games.",
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
