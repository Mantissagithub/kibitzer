# combined two-panel tdleaf(lambda) result figure.
# panel a (left): the curriculum climb across stockfish rungs, reusing the exact
# rendering logic from scripts/plot_tdleaf_curriculum.py (reads
# reports/tdleaf/tdleaf_log.jsonl).
# panel b (right): the controlled gate comparing the tdleaf-trained value head
# against the untrained base at 256 sims, n=40 games, at stockfish 1320 and 1900
# (numbers from reports/tdleaf/gate_{base,tdleaf}_{1320,1900}.json).
# run: uv run python scripts/plot_tdleaf_result.py

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
# auto-advance points. identical to plot_tdleaf_curriculum.py.
RUNGS = [
    ("SF-1320", 1, 20, "#a6cee3"),
    ("SF-1500", 21, 40, "#b2df8a"),
    ("SF-1700", 41, 60, "#fdbf6f"),
    ("SF-1900", 61, 200, "#fb9a99"),
]

# controlled-gate numbers (256 sims, n=40 games), from
# reports/tdleaf/gate_{base,tdleaf}_{1320,1900}.json.
GATE_RESULTS = {
    ("base", "1320"): {"score": 0.975, "ci": (0.871, 0.996), "wdl": "39W/0D/1L"},
    ("tdleaf", "1320"): {"score": 0.975, "ci": (0.871, 0.996), "wdl": "39W/0D/1L"},
    ("base", "1900"): {"score": 0.738, "ci": (0.585, 0.849), "wdl": "27W/5D/8L"},
    ("tdleaf", "1900"): {"score": 0.762, "ci": (0.611, 0.868), "wdl": "26W/9D/5L"},
}
D35_CEILING = 0.2


def load_log(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_curriculum(ax: plt.Axes, rows: list[dict]) -> None:
    games = [row["game"] for row in rows]
    rolling = [row["rolling_score"] for row in rows]
    z_as_score = [(row["z"] + 1.0) / 2.0 for row in rows]

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
    ax.set_title("Curriculum climb: value learns from its own search leaves", pad=22)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def plot_gate(ax: plt.Axes) -> None:
    groups = ["1320", "1900"]
    group_labels = ["vs SF-1320", "vs SF-1900"]
    conditions = ["base", "tdleaf"]
    colors = {"base": "#999999", "tdleaf": "#e15759"}
    cond_labels = {"base": "base (untrained)", "tdleaf": "TDLeaf"}

    width = 0.35
    x = range(len(groups))

    for offset, cond in ((-1, "base"), (1, "tdleaf")):
        scores = [GATE_RESULTS[(cond, g)]["score"] for g in groups]
        lows = [GATE_RESULTS[(cond, g)]["ci"][0] for g in groups]
        highs = [GATE_RESULTS[(cond, g)]["ci"][1] for g in groups]
        yerr_lower = [s - lo for s, lo in zip(scores, lows, strict=True)]
        yerr_upper = [hi - s for s, hi in zip(scores, highs, strict=True)]
        positions = [pos + offset * width / 2 for pos in x]
        bars = ax.bar(
            positions,
            scores,
            width=width,
            color=colors[cond],
            label=cond_labels[cond],
            zorder=3,
        )
        ax.errorbar(
            positions,
            scores,
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.1,
            capsize=4,
            zorder=4,
        )
        for pos, score in zip(positions, scores, strict=True):
            ax.text(
                pos,
                score + 0.035,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # even-score reference line.
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=2)
    ax.text(len(groups) - 0.42, 0.5, "even", va="bottom", ha="left", fontsize=8, color="#555555")

    # d35 ceiling reference line.
    ax.axhline(D35_CEILING, color="#8c564b", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(
        -0.42,
        D35_CEILING + 0.02,
        "D35 ceiling (old base, vs SF-1320)",
        va="bottom",
        ha="left",
        fontsize=7.5,
        color="#8c564b",
    )

    # annotate saturation over the 1320 group.
    ax.text(
        0,
        1.18,
        "saturated (0.975 = 0.975)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    # bracket + annotation over the 1900 group showing the null result.
    bracket_y = 0.98
    ax.plot(
        [1 - width / 2, 1 - width / 2, 1 + width / 2, 1 + width / 2],
        [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
        color="#333333",
        linewidth=1.0,
        zorder=5,
    )
    ax.text(
        1,
        bracket_y + 0.015,
        "Δ = +0.025 (n.s.)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("score vs Stockfish")
    ax.set_title("Controlled gate: TDLeaf vs untrained base (256 sims, N=40)", pad=22)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "reports" / "tdleaf" / "tdleaf_log.jsonl"
    output_path = repo_root / "reports" / "tdleaf" / "fig_tdleaf_result.png"

    rows = load_log(log_path)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_curriculum(ax_a, rows)
    plot_gate(ax_b)

    fig.suptitle(
        "TDLeaf(λ): the ceiling moved with the base + search, not the value training",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.005,
        "S2 shaw base (44.5% top-1); left: 200 self-play games, 64 sims; "
        "right: 40-game gate matches, 256 sims, Wilson 95% CIs.",
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
