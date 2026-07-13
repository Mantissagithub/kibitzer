from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "evidence.json").read_text(encoding="utf-8"))

INK = "#17202a"
MUTED = "#667085"
GRID = "#d0d5dd"
BASE = "#157f6f"
CANDIDATE = "#d1495b"
NEUTRAL = "#e9b44c"
SEARCH = "#3f7cac"
PAPER = "#f8f9fb"


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "legend.frameon": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def short_method(row: dict) -> str:
    name = row["method"]
    if name == "DPO/AWAC anchor retry":
        return "DPO/AWAC\nanchor"
    if name == "512-sim expert iteration":
        return "Expert iteration\n" + ("128 gate" if "128" in row["protocol"] else "512 gate")
    return name.replace(" + ", " +\n")


def figure_external_scores() -> None:
    rows = DATA["external_comparisons"]
    labels = [short_method(row) for row in rows]
    baseline = np.array([row["baseline_score"] for row in rows])
    candidate = np.array([row["candidate_score"] for row in rows])
    x = np.arange(len(rows))
    width = 0.34

    fig, (ax, delta_ax) = plt.subplots(
        2,
        1,
        figsize=(13.2, 8.4),
        gridspec_kw={"height_ratios": [3.0, 1.35], "hspace": 0.18},
    )
    ax.bar(x - width / 2, baseline, width, color=BASE, label="run-specific baseline")
    ax.bar(x + width / 2, candidate, width, color=CANDIDATE, label="RL candidate")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score rate")
    ax.set_title("Every RL-style update was neutral or worse on its decision gate", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    ax.set_xticks([])
    for index, (base, cand) in enumerate(zip(baseline, candidate)):
        ax.text(index - width / 2, base + 0.025, f"{base:.3f}", ha="center", fontsize=9, color=BASE)
        ax.text(index + width / 2, cand + 0.025, f"{cand:.3f}", ha="center", fontsize=9, color=CANDIDATE)

    deltas = candidate - baseline
    colors = [NEUTRAL if row["decision"] == "neutral" else CANDIDATE for row in rows]
    delta_ax.axhline(0, color=INK, linewidth=1.1)
    delta_ax.bar(x, deltas, color=colors, width=0.58)
    delta_ax.set_ylim(-0.18, 0.06)
    delta_ax.set_ylabel("candidate - baseline")
    delta_ax.set_xticks(x, labels)
    delta_ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    delta_ax.set_axisbelow(True)
    for index, delta in enumerate(deltas):
        offset = 0.008 if delta >= 0 else -0.013
        va = "bottom" if delta >= 0 else "top"
        delta_ax.text(index, delta + offset, f"{delta:+.3f}", ha="center", va=va, fontsize=9)

    fig.text(
        0.01,
        0.01,
        "Protocols differ by row. Deltas are always against that run's own documented baseline; they are not a cross-method ranking.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "fig1_external_gate_summary.png")


def figure_transfer_gap() -> None:
    transfer = DATA["selfplay_transfer"]
    historical = DATA["historical_selfplay"]
    labels = ["AZ-lite\nvs parent", "Proper AZ\nvs parent", "AZ iter 1\nvs parent", "AZ iter 1\nvs Leela-2700"]
    values = [
        historical[0]["candidate_score_vs_parent"],
        historical[1]["candidate_score_vs_parent"],
        transfer["sibling_score"],
        transfer["external_score"],
    ]
    colors = [CANDIDATE, CANDIDATE, BASE, CANDIDATE]

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=colors)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.2, label="head-to-head parity")
    ax.axhline(
        transfer["external_reference"],
        color=SEARCH,
        linestyle=":",
        linewidth=1.8,
        label="external base reference (0.225)",
    )
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("candidate score rate")
    ax.set_xticks(x, labels)
    ax.set_title("Self-play could beat its parent and still lose external strength", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", weight="bold")
    ax.annotate(
        "+0.125 vs parent",
        xy=(2, values[2]),
        xytext=(1.45, 0.71),
        arrowprops={"arrowstyle": "->", "color": BASE, "lw": 1.4},
        color=BASE,
        weight="bold",
    )
    ax.annotate(
        "-0.125 vs external reference",
        xy=(3, values[3]),
        xytext=(2.25, 0.31),
        arrowprops={"arrowstyle": "->", "color": CANDIDATE, "lw": 1.4},
        color=CANDIDATE,
        weight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "The first three bars are sibling matches. The last bar is an external gate, so the contrast is diagnostic rather than a shared rating scale.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "fig2_selfplay_transfer_gap.png")


def figure_grpo_diagnostics() -> None:
    rows = DATA["grpo_iterations"]
    iterations = np.array([row[0] for row in rows])
    elos = np.array([row[1] for row in rows])
    scores = np.array([row[2] for row in rows])
    probe_x = np.array([row[0] for row in rows if row[3] is not None])
    probes = np.array([row[3] for row in rows if row[3] is not None])

    fig, (ax, gate_ax) = plt.subplots(1, 2, figsize=(13.4, 5.9), gridspec_kw={"width_ratios": [1.45, 1]})
    ax.plot(iterations, scores, color=SEARCH, marker="o", linewidth=2.2, label="score vs adaptive opponent")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("GRPO iteration")
    ax.set_ylabel("rollout score")
    ax.set_title("The ladder found the base's level", loc="left")
    ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
    elo_ax = ax.twinx()
    elo_ax.step(iterations, elos, where="mid", color=NEUTRAL, linewidth=2, label="opponent Elo")
    elo_ax.set_ylim(1800, 2600)
    elo_ax.set_ylabel("adaptive Stockfish Elo", color=NEUTRAL)
    elo_ax.tick_params(axis="y", colors=NEUTRAL)
    lines = ax.get_lines()[:1] + elo_ax.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")

    gate_ax.plot(probe_x, probes, color=BASE, marker="o", linewidth=2.5, markersize=8)
    gate_ax.set_xlim(4, 11)
    gate_ax.set_ylim(0.86, 0.94)
    gate_ax.set_xticks(probe_x)
    gate_ax.set_xlabel("GRPO iteration")
    gate_ax.set_ylabel("fixed SF-2000 probe score")
    gate_ax.set_title("The fixed probe stayed flat", loc="left")
    gate_ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
    for x_value, value in zip(probe_x, probes):
        gate_ax.text(x_value, value + 0.006, f"{value:.4f}", ha="center", color=BASE, weight="bold")
    gate_ax.text(
        0.05,
        0.08,
        "External Leela gate:\nGRPO 0.275\nbase 0.294",
        transform=gate_ax.transAxes,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID},
        fontsize=11,
    )
    fig.suptitle("External rewards fixed self-play exploitation, but not credit assignment", x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig3_grpo_diagnostics.png")


def figure_failure_map() -> None:
    failure = DATA["failure_modes"]
    values = np.array(failure["strength"])
    cmap = ListedColormap(["#eef1f5", "#f3cf72", "#d1495b"])

    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    dimension_labels = [
        "self-referential\nteacher",
        "sparse\ncredit",
        "value bottleneck\nuntouched",
        "offline/external\nmismatch",
        "narrow state\ndistribution",
    ]
    ax.set_xticks(np.arange(len(dimension_labels)), dimension_labels)
    ax.set_yticks(np.arange(len(failure["methods"])), failure["methods"])
    ax.set_title("Observed failure mechanisms recur across different RL objectives", loc="left", pad=18)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            label = ["not central", "contributing", "direct evidence"][values[row, col]]
            ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.01,
        0.01,
        "This is an evidence map, not a causal estimate. Ratings summarize the experiment ledger and external failure analysis.",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.20)
    save(fig, "fig4_failure_mechanism_map.png")


def rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10.5, weight="bold")


def figure_causal_summary() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("The common path from a successful optimizer step to a failed engine", loc="left", pad=14)

    boxes = [
        (0.02, 0.58, 0.18, 0.20, "Narrow or self-generated\nrollout distribution", SEARCH),
        (0.27, 0.58, 0.18, 0.20, "Correlated targets or\nsparse game reward", NEUTRAL),
        (0.52, 0.58, 0.18, 0.20, "Training objective\nimproves", BASE),
        (0.77, 0.58, 0.20, 0.20, "External play is flat\nor weaker", CANDIDATE),
        (0.27, 0.15, 0.18, 0.22, "Weak load-bearing\nvalue estimate", CANDIDATE),
        (0.50, 0.15, 0.22, 0.22, "Policy-only updates\ncannot repair the\nrepresentation", MUTED),
    ]
    for box in boxes:
        rounded_box(ax, *box)

    arrow = {"arrowstyle": "->", "lw": 1.8, "color": MUTED}
    ax.annotate("", xy=(0.27, 0.68), xytext=(0.20, 0.68), arrowprops=arrow)
    ax.annotate("", xy=(0.52, 0.68), xytext=(0.45, 0.68), arrowprops=arrow)
    ax.annotate("", xy=(0.77, 0.68), xytext=(0.70, 0.68), arrowprops=arrow)
    ax.annotate("", xy=(0.36, 0.58), xytext=(0.36, 0.36), arrowprops=arrow)
    ax.annotate("", xy=(0.50, 0.26), xytext=(0.45, 0.26), arrowprops=arrow)
    ax.annotate("", xy=(0.87, 0.58), xytext=(0.72, 0.26), arrowprops=arrow)
    ax.text(0.5, 0.03, "Search can average the noisy value at inference; distilling that behavior back into the same weights did not transfer.", ha="center", color=MUTED)
    save(fig, "fig5_causal_summary.png")


def main() -> None:
    setup()
    figure_external_scores()
    figure_transfer_gap()
    figure_grpo_diagnostics()
    figure_failure_map()
    figure_causal_summary()
    print(f"wrote 5 figures -> {ROOT}")


if __name__ == "__main__":
    main()
