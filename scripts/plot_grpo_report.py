# D55 GRPO+DPPO report figures.
# 4 figures telling the "neutral" story: ladder climb confirms ~2500 base level,
# fixed-opponent probe is flat, external gate vs Leela-2700 shows no gain
# (grpo_v5 0.275 vs base 0.294), and a ledger of every non-scale lever tried.
# data: runs/grpo/metrics.jsonl, values given directly in task spec for the
# external gate and the non-scale ledger.
# run: uv run python scripts/plot_grpo_report.py

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
        "savefig.dpi": 180,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "runs" / "grpo" / "metrics.jsonl"
OUTPUT_DIR = REPO_ROOT / "reports" / "grpo"

MODEL_COLOR = "#4c78a8"
LADDER_COLOR = "#e15759"
PROBE_COLOR = "#54a24b"
GREY = "#8c8c8c"
NEUTRAL_COLOR = "#9a9a9a"
NEGATIVE_COLOR = "#e15759"
SCALE_COLOR = "#2ca02c"


def load_metrics() -> list[dict]:
    rows = []
    with METRICS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig1_ladder_climb(rows: list[dict], output_dir: Path) -> Path:
    iters = [row["iter"] for row in rows]
    scores = [row["score"] for row in rows]
    elos = [row["elo"] for row in rows]
    probe_iters = [row["iter"] for row in rows if row.get("probe") is not None]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.bar(iters, scores, color=MODEL_COLOR, alpha=0.85, width=0.55, zorder=3, label="model score vs ladder")
    ax1.axhline(0.5, color="#333333", linewidth=1, linestyle="--", zorder=2)
    ax1.text(1.0, 0.515, "even (0.5)", fontsize=8, color="#555555", va="bottom")
    ax1.set_xlabel("GRPO iteration")
    ax1.set_ylabel("model score vs adaptive ladder", color=MODEL_COLOR)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(iters)
    ax1.tick_params(axis="y", labelcolor=MODEL_COLOR)

    for it in probe_iters:
        ax1.axvline(it, color=PROBE_COLOR, linewidth=1.2, linestyle=":", zorder=1)
    ax1.text(
        probe_iters[0] - 0.15, 0.97, "probe", color=PROBE_COLOR, fontsize=8,
        ha="right", va="top", rotation=90,
    )
    ax1.text(
        probe_iters[1] - 0.15, 0.97, "probe", color=PROBE_COLOR, fontsize=8,
        ha="right", va="top", rotation=90,
    )

    ax2 = ax1.twinx()
    ax2.plot(iters, elos, color=LADDER_COLOR, marker="o", markersize=6, linewidth=1.8, zorder=4, label="opponent Stockfish Elo")
    ax2.set_ylabel("opponent Stockfish Elo (adaptive ladder)", color=LADDER_COLOR)
    ax2.tick_params(axis="y", labelcolor=LADDER_COLOR)
    ax2.set_ylim(1800, 2700)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8, framealpha=0.9)

    ax1.set_title("Adaptive ladder climbs to ~2500, then score settles near 50%", pad=10)
    fig.text(
        0.5,
        0.005,
        "The ladder steps ±100 Elo toward the model's level; settling at ~2500 confirms the base's\n"
        "existing strength — it is what a static ~2500 model produces here, not evidence of a gain.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path = output_dir / "fig1_ladder_climb.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig2_probe_flat(rows: list[dict], output_dir: Path) -> Path:
    probe_rows = [row for row in rows if row.get("probe") is not None]
    iters = [row["iter"] for row in probe_rows]
    probes = [row["probe"] for row in probe_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, probes, color=PROBE_COLOR, marker="o", markersize=10, linewidth=2, zorder=3, label="probe@2000 score")
    mean_probe = sum(probes) / len(probes)
    ax.axhline(mean_probe, color="#333333", linewidth=1, linestyle="--", zorder=2, label=f"flat guide ({mean_probe:.3f})")

    for it, val in zip(iters, probes, strict=True):
        ax.annotate(f"{val:.3f}", (it, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlim(iters[0] - 2, iters[-1] + 2)
    ax.set_xticks(iters)
    ax.set_ylim(0.80, 1.00)
    ax.set_xlabel("GRPO iteration")
    ax.set_ylabel("probe score vs fixed SF-2000 opponent")
    ax.set_title("probe@2000 (fixed held-out opponent, greedy 128-sim): flat — no real gain", pad=10)
    ax.legend(loc="lower center", fontsize=8, framealpha=0.9)

    fig.text(
        0.5,
        0.005,
        "y-axis zoomed to 0.80-1.00 to keep the two points legible; note the full possible range is 0-1,\n"
        "so this ~1pt drift is well within noise for 20-game probes — the fixed-opponent signal never moved.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    path = output_dir / "fig2_probe_flat.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig3_external_gate(output_dir: Path) -> Path:
    labels = ["grpo_v5", "tactical_repair\n(base)"]
    scores = [0.275, 0.294]
    wins = [12, 12]
    draws = [20, 23]
    losses = [48, 45]
    threshold = 0.324
    colors = [LADDER_COLOR, GREY]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.2))

    bars = ax_a.bar(labels, scores, color=colors, width=0.5, zorder=3)
    for bar, val in zip(bars, scores, strict=True):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax_a.axhline(threshold, color="#333333", linewidth=1.4, linestyle="--", zorder=2)
    ax_a.text(-0.4, threshold + 0.01, "promotion bar (base +0.03) = 0.324", fontsize=8, color="#333333", va="bottom")
    ax_a.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    ax_a.text(-0.4, 0.505, "even (0.5)", fontsize=7.5, color="#888888", va="bottom")
    ax_a.set_ylim(0, 0.6)
    ax_a.set_ylabel("match score rate vs Leela/Maia-2700 (80 games)")
    ax_a.set_title("grpo_v5 sits below base — no promotion", pad=10)

    x = range(len(labels))
    b_wins = ax_b.bar(x, wins, color="#2ca02c", label="wins")
    b_draws = ax_b.bar(x, draws, bottom=wins, color="#ffbf00", label="draws")
    bottoms = [w + d for w, d in zip(wins, draws, strict=True)]
    b_losses = ax_b.bar(x, losses, bottom=bottoms, color="#d62728", label="losses")
    ax_b.set_xticks(list(x), labels)
    ax_b.set_ylabel("games (of 80)")
    ax_b.set_title("wins identical (12=12); 3 base draws become losses", pad=10)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)
    for i, (w, d, l) in enumerate(zip(wins, draws, losses, strict=True)):
        ax_b.text(i, w / 2, str(w), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax_b.text(i, w + d / 2, str(d), ha="center", va="center", fontsize=8, color="black", fontweight="bold")
        ax_b.text(i, w + d + l / 2, str(l), ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    fig.suptitle("External gate vs Leela-2700: grpo_v5 = 0.275, base = 0.294 (neutral, no gain)", fontsize=12)
    fig.text(
        0.5,
        0.005,
        "80-game paired gate, 128-sim PUCT, seed 23. Both far below the 0.324 promotion threshold; the\n"
        "0.019 gap is within 80-game noise — win counts match exactly, only draw/loss split differs.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    path = output_dir / "fig3_external_gate.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig4_nonscale_ledger(output_dir: Path) -> Path:
    entries = [
        ("D35 value-target repair", "neutral"),
        ("D45 tactical mid-training", "neutral"),
        ("D48 self-play smoke (AZ-lite)", "negative"),
        ("D49 proper AlphaZero 1-iter", "negative"),
        ("D50 search-axis / PUCT tuning", "neutral"),
        ("D51 competition-data continuation", "neutral"),
        ("D52 enlarged value head", "negative"),
        ("D53/54 on-policy distill from lc0", "negative"),
        ("D55 GRPO+DPPO (this run)", "neutral"),
        ("D43 SCALE (100M data)", "positive"),
    ]
    # keep chronological top-to-bottom on the chart (barh plots bottom-up).
    entries = list(reversed(entries))
    labels = [label for label, _ in entries]
    outcomes = [outcome for _, outcome in entries]
    color_map = {"neutral": NEUTRAL_COLOR, "negative": NEGATIVE_COLOR, "positive": SCALE_COLOR}
    colors = [color_map[outcome] for outcome in outcomes]
    values = [1.0 if outcome != "positive" else 1.6 for outcome in outcomes]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, color=colors, zorder=3)
    for bar, outcome in zip(bars, outcomes, strict=True):
        ax.text(
            bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, outcome,
            va="center", fontsize=9, fontweight="bold",
            color=color_map[outcome],
        )
    ax.set_xlim(0, 2.1)
    ax.set_xticks([])
    ax.set_xlabel("outcome vs external yardstick (bar length is illustrative, not a magnitude)")
    ax.set_title("Nine non-scale levers: flat or down. Scale is the only positive slope.", pad=10)
    ax.grid(False)

    fig.text(
        0.5,
        0.01,
        "Every cheap fine-tuning/search/data lever tried against the external Leela/Maia-2700 yardstick\n"
        "either held flat or regressed strength; only scaling training data (D43, 100M) moved the needle.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    path = output_dir / "fig4_nonscale_ledger.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_readme(output_dir: Path, figures: list[Path]) -> Path:
    lines = [
        "# D55 GRPO+DPPO report",
        "",
        "Figures for the GRPO + exact-divergence DPPO RL fine-tuning run on the 15.2M tactical_repair "
        "checkpoint (~2500 Elo), trained against an adaptive Stockfish Elo ladder with searched (128-sim "
        "PUCT) rollouts, a DPPO total-variation trust region, and a KL anchor to the base.",
        "",
        "## Verdict",
        "",
        "- External gate vs Leela/Maia-2700 (80 games, 128 sims, seed 23): `grpo_v5` scores 0.275 "
        "(12W/20D/48L) vs base `tactical_repair` 0.294 (12W/23D/45L). Both far below the 0.324 promotion "
        "bar (base + 0.03); win counts are identical, only 3 draws flip to losses — the gap is noise, "
        "not signal. **Not promoted.**",
        "- The fixed-opponent probe@2000 is flat across training (0.9125 at iter 5, 0.900 at iter 10) — "
        "the one honest signal in the run shows no real gain.",
        "- The adaptive ladder climbing to ~2500 Elo and the score settling near 50% there is consistent "
        "with the base's pre-existing strength, not with the run producing a stronger model.",
        "- The ceiling estimate (~2500-2600 Elo) stands: this is the 9th non-scale lever to land flat or "
        "negative; scale (D43) remains the only positive lever found so far.",
        "",
        "## Figures",
        "",
    ]
    descriptions = {
        "fig1_ladder_climb.png": "Per-iteration model score vs the adaptive ladder Elo; the ladder climbs "
        "to ~2500 while score settles near 50%, confirming the base's existing level rather than a gain.",
        "fig2_probe_flat.png": "The fixed-opponent probe@2000 at iterations 5 and 10 (0.9125, 0.900) — "
        "flat, the honest read on whether training produced real improvement.",
        "fig3_external_gate.png": "The decisive comparison vs Leela/Maia-2700: score bars against the "
        "promotion threshold, plus a W/D/L breakdown showing identical win counts for grpo_v5 and base.",
        "fig4_nonscale_ledger.png": "Ledger of every non-scale lever tried (D35 through D55) vs the "
        "external yardstick; nine land flat or negative, only scale (D43) is positive.",
    }
    for figure in figures:
        lines.append(f"- **{figure.name}** — {descriptions[figure.name]}")
        lines.append(f"  ![{figure.stem}]({figure.name})")
    lines.append("")
    lines.extend(
        [
            "## Rebuild",
            "",
            "```bash",
            "uv run python scripts/plot_grpo_report.py",
            "```",
            "",
        ]
    )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    figures = [
        fig1_ladder_climb(rows, OUTPUT_DIR),
        fig2_probe_flat(rows, OUTPUT_DIR),
        fig3_external_gate(OUTPUT_DIR),
        fig4_nonscale_ledger(OUTPUT_DIR),
    ]
    readme = write_readme(OUTPUT_DIR, figures)
    print(f"wrote {readme}")
    for figure in figures:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
