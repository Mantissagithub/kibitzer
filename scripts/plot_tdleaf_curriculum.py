# panel a of the tdleaf(lambda) figure: the curriculum climb across stockfish
# rungs. reads reports/tdleaf/tdleaf_log.jsonl (200 self-play games) and renders
# the rolling score vs the stockfish opponent across the auto-advancing curriculum.
# run: uv run python scripts/plot_tdleaf_curriculum.py

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

# rung boundaries by game index (inclusive), matching the curriculum's actual
# auto-advance points.
RUNGS = [
    ("SF-1320", 1, 20, "#a6cee3"),
    ("SF-1500", 21, 40, "#b2df8a"),
    ("SF-1700", 41, 60, "#fdbf6f"),
    ("SF-1900", 61, 200, "#fb9a99"),
]


def load_log(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_curriculum(rows: list[dict], output_path: Path) -> Path:
    games = [row["game"] for row in rows]
    rolling = [row["rolling_score"] for row in rows]
    z_as_score = [(row["z"] + 1.0) / 2.0 for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # rung background bands + boundary lines + centered top labels.
    for name, lo, hi, color in RUNGS:
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.25, zorder=0, lw=0)
        ax.text(
            (lo + hi) / 2,
            1.08,
            name,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    for boundary in (20.5, 40.5, 60.5):
        ax.axvline(boundary, color="#888888", linewidth=0.7, linestyle="-", zorder=1)

    # even-score reference line.
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=2)
    ax.text(202, 0.5, "even", va="center", ha="left", fontsize=8, color="#555555")

    # faint per-game outcome texture underneath the hero line.
    ax.scatter(games, z_as_score, s=8, color="#4c4c4c", alpha=0.12, zorder=2, linewidths=0)

    # hero series: rolling score.
    ax.plot(
        games,
        rolling,
        color="#1f77b4",
        linewidth=1.8,
        zorder=3,
        label="rolling score (resets each curriculum advance)",
    )

    ax.set_xlim(1, 200)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("self-play game")
    ax.set_ylabel("score vs Stockfish (win=1, draw=½, loss=0)")
    ax.set_title("TDLeaf(λ) curriculum: value learns from its own search leaves", pad=22)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.text(
        0.5,
        0.005,
        "S2 shaw base (44.5% top-1), 200 self-play games, 64 sims; "
        "rolling score resets at each curriculum advance.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "reports" / "tdleaf" / "tdleaf_log.jsonl"
    output_path = repo_root / "reports" / "tdleaf" / "fig_panelA_curriculum.png"

    rows = load_log(log_path)
    path = plot_curriculum(rows, output_path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
