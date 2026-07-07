# search-axis summary figure for D50.
# run: uv run python scripts/plot_search_lab.py

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
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 320,
    }
)


ORDER = [
    ("baseline_puct", 128, "baseline\nPUCT"),
    ("puct_fpu", 128, "PUCT\nFPU"),
    ("puct_prune", 128, "PUCT\nprune"),
    ("puct_stacked", 128, "stacked\n128"),
    ("puct_stacked", 256, "stacked\n256"),
    ("alphabeta", 128, "alpha-beta"),
    ("alphabeta_quiescence", 128, "alpha-beta\n+ qsearch"),
]


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        rows.append(row)
    by_key = {(row["variant"], row["budget"]): row for row in rows}
    ordered = []
    for variant, budget, label in ORDER:
        ordered.append(by_key[(variant, budget)] | {"label": label})
    return ordered


def plot_scores(ax: plt.Axes, rows: list[dict]) -> None:
    ys = list(range(len(rows)))
    scores = [row["score"] for row in rows]
    colors = [
        "#4e79a7" if row["variant"].startswith("puct") or row["variant"] == "baseline_puct" else "#e15759"
        for row in rows
    ]

    ax.barh(ys, scores, color=colors, alpha=0.90, edgecolor="#333333", linewidth=0.5)
    ax.axvline(0.5, color="#555555", linewidth=1.1, linestyle="--", label="even")
    ax.axvline(0.6, color="#59a14f", linewidth=1.1, linestyle=":", label="baseline self-match")
    ax.axvspan(0.5, 0.65, color="#59a14f", alpha=0.07, zorder=0)

    for y, row in zip(ys, rows):
        ax.text(
            row["score"] + 0.015,
            y,
            f"{row['score']:.3f}   {row['w']}W/{row['d']}D/{row['l']}L",
            ha="left",
            va="center",
            fontsize=10,
            color="#333333",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"].replace("\n", " ") for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.78)
    ax.set_xlabel("score vs baseline PUCT")
    ax.set_title("Match result by search method")
    ax.legend(loc="lower right", framealpha=0.92)


def plot_evals(ax: plt.Axes, rows: list[dict]) -> None:
    ys = list(range(len(rows)))
    variant = [row["avg_evals_variant"] for row in rows]
    baseline = [row["avg_evals_baseline"] for row in rows]
    offset = 0.18

    ax.barh([y - offset for y in ys], variant, height=0.32, color="#4e79a7", alpha=0.90, label="variant")
    ax.barh([y + offset for y in ys], baseline, height=0.32, color="#f28e2b", alpha=0.85, label="baseline PUCT")
    for y, row in zip(ys, rows):
        ax.text(
            row["avg_evals_variant"] + 4,
            y - offset,
            f"{row['avg_evals_variant']:.0f}",
            ha="left",
            va="center",
            fontsize=9,
            color="#333333",
        )
        ax.text(
            row["avg_evals_baseline"] + 4,
            y + offset,
            f"{row['avg_evals_baseline']:.0f}",
            ha="left",
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"].replace("\n", " ") for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 285)
    ax.set_xlabel("average model evaluations / move")
    ax.set_title("Compute spent per move")
    ax.legend(loc="upper left", framealpha=0.92)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = load_rows(repo_root / "search_lab" / "results")
    output_path = repo_root / "search_lab" / "fig_search_lab.png"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7), width_ratios=(1.18, 1.0))
    plot_scores(ax_a, rows)
    plot_evals(ax_b, rows)
    fig.suptitle("Search-axis lab: PUCT tweaks are noise; alpha-beta collapses", fontsize=15)
    fig.text(
        0.5,
        0.005,
        "score = (wins + 0.5 × draws) / games from the variant side. "
        "The 0.60 baseline-vs-baseline row is the observed sample/color/opening noise floor.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
