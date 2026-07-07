# search-axis summary figure for D50.
# run: uv run python scripts/plot_search_lab.py

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
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 300,
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


def wilson_half_width(score: float, games: int) -> float:
    # Binomial SE is good enough for a visual error bar. Draws are folded into
    # the score, so this is a rough read, not a formal confidence interval.
    return 1.96 * math.sqrt(max(0.0, score * (1.0 - score)) / max(1, games))


def plot_scores(ax: plt.Axes, rows: list[dict]) -> None:
    xs = list(range(len(rows)))
    scores = [row["score"] for row in rows]
    yerr = [wilson_half_width(row["score"], row["games"]) for row in rows]
    colors = [
        "#4e79a7" if row["variant"].startswith("puct") or row["variant"] == "baseline_puct" else "#e15759"
        for row in rows
    ]

    ax.bar(xs, scores, yerr=yerr, capsize=3, color=colors, alpha=0.88, edgecolor="#333333", linewidth=0.4)
    ax.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--")
    ax.axhline(0.6, color="#59a14f", linewidth=1.0, linestyle=":", label="baseline self-match noise floor")
    ax.axhspan(0.5, 0.65, color="#59a14f", alpha=0.07, zorder=0)

    for x, row in zip(xs, rows):
        ax.text(
            x,
            row["score"] + 0.035,
            f"{row['w']}/{row['d']}/{row['l']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.set_ylim(0.0, 0.78)
    ax.set_ylabel("score vs baseline PUCT")
    ax.set_title("Search variants at equal net-eval budget")
    ax.legend(loc="upper right", framealpha=0.92)


def plot_evals(ax: plt.Axes, rows: list[dict]) -> None:
    xs = list(range(len(rows)))
    variant = [row["avg_evals_variant"] for row in rows]
    baseline = [row["avg_evals_baseline"] for row in rows]

    ax.plot(xs, variant, marker="o", linewidth=1.8, color="#4e79a7", label="variant")
    ax.plot(xs, baseline, marker="o", linewidth=1.2, color="#f28e2b", label="baseline PUCT")
    for x, row in zip(xs, rows):
        ax.text(
            x,
            row["avg_evals_variant"] + 7,
            f"{row['avg_evals_variant']:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.set_ylabel("average model evaluations / move")
    ax.set_title("Compute actually spent")
    ax.legend(loc="upper left", framealpha=0.92)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = load_rows(repo_root / "search_lab" / "results")
    output_path = repo_root / "reports" / "scaling_law" / "fig_search_lab.png"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    plot_scores(ax_a, rows)
    plot_evals(ax_b, rows)
    fig.suptitle("Search-axis lab: PUCT tweaks are noise; alpha-beta collapses", fontsize=12.5)
    fig.text(
        0.5,
        0.005,
        "score is from the variant's side; labels above bars are W/D/L. "
        "The 0.60 baseline-vs-baseline result is treated as the observed noise floor.",
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
