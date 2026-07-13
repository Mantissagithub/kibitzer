from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LABELED = REPO / "runs/oracle_process_rl/oracle_labeled.jsonl"
METRICS = ROOT / "training_metrics_signed_selector.jsonl"
BASE_CHECKPOINT = REPO / "runs/tactical/tactical_repair.pt"
FINAL_CHECKPOINT = REPO / "runs/oracle_process_rl/oracle_process_rl.pt"

INK = "#17202a"
MUTED = "#667085"
GRID = "#d0d5dd"
BASE = "#157f6f"
CANDIDATE = "#d1495b"
NEUTRAL = "#e9b44c"
SEARCH = "#3f7cac"
PAPER = "#f8f9fb"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_interval(scores: list[float], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(scores, dtype=float)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def checkpoint_difference() -> tuple[float, int]:
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    final = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    max_diff = max(float((base[key] - final[key]).abs().max().item()) for key in base)
    changed = sum(not torch.equal(base[key], final[key]) for key in base)
    return max_diff, changed


def figure_signal_funnel(records: list[dict]) -> None:
    outcomes: dict[int, float] = {}
    for record in records:
        outcomes[int(record["game_id"])] = float(record["reward"])
    wins = sum(value == 1.0 for value in outcomes.values())
    draws = sum(value == 0.5 for value in outcomes.values())
    losses = sum(value == 0.0 for value in outcomes.values())
    regret_hits = sum(float(record["regret"]) >= 0.05 for record in records)
    advantage_hits = sum(abs(float(record["advantage"])) >= 0.1 for record in records)
    kept = [
        record
        for record in records
        if float(record["regret"]) >= 0.05 and abs(float(record["advantage"])) >= 0.1
    ]
    positive = sum(float(record["advantage"]) > 0.0 for record in kept)
    negative = sum(float(record["advantage"]) < 0.0 for record in kept)

    fig, (game_ax, funnel_ax) = plt.subplots(1, 2, figsize=(13.4, 5.8), gridspec_kw={"width_ratios": [0.8, 1.35]})
    bars = game_ax.bar(["wins", "draws", "losses"], [wins, draws, losses], color=[BASE, NEUTRAL, CANDIDATE], width=0.62)
    game_ax.set_ylim(0, 26)
    game_ax.set_ylabel("games")
    game_ax.set_title("Fresh 512-sim rollout", loc="left")
    game_ax.grid(axis="y", color=GRID, alpha=0.7)
    game_ax.set_axisbelow(True)
    for bar, value in zip(bars, [wins, draws, losses]):
        game_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, str(value), ha="center", weight="bold")
    game_ax.text(0.02, 0.92, "score 0.812 vs SF-2300", transform=game_ax.transAxes, color=MUTED)

    labels = ["all labeled", "regret >= 0.05", "|advantage| >= 0.1", "both filters"]
    values = [len(records), regret_hits, advantage_hits, len(kept)]
    colors = [SEARCH, NEUTRAL, NEUTRAL, CANDIDATE]
    y = np.arange(len(labels))
    bars = funnel_ax.barh(y, values, color=colors, height=0.58)
    funnel_ax.set_yticks(y, labels)
    funnel_ax.invert_yaxis()
    funnel_ax.set_xlabel("model positions")
    funnel_ax.set_title("Only 13.2% carried the training signal", loc="left")
    funnel_ax.grid(axis="x", color=GRID, alpha=0.7)
    funnel_ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        funnel_ax.text(value + 18, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", weight="bold")
    funnel_ax.text(
        0.98,
        0.08,
        f"kept advantages\n{positive} positive  |  {negative} negative",
        transform=funnel_ax.transAxes,
        ha="right",
        color=MUTED,
    )
    fig.suptitle("Oracle process RL: rollout and signal funnel", x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig1_signal_funnel.png")


def figure_reward_distributions(records: list[dict]) -> None:
    regret_cp = np.asarray([float(record["regret"]) * 1000.0 for record in records])
    process = np.asarray([float(record["process_reward"]) for record in records])
    returns = np.asarray([float(record["return"]) for record in records])
    advantages = np.asarray([float(record["advantage"]) for record in records])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    plots = [
        (axes[0, 0], regret_cp, "Stockfish regret", "approximate centipawns", SEARCH),
        (axes[0, 1], process, "Clipped process reward", "reward", CANDIDATE),
        (axes[1, 0], returns, "Return-to-go", "shaped return", BASE),
        (axes[1, 1], advantages, "Group-relative advantage", "advantage", NEUTRAL),
    ]
    for ax, values, title, xlabel, color in plots:
        ax.hist(values, bins=35, color=color, alpha=0.88, edgecolor=PAPER)
        ax.axvline(0.0, color=INK, linewidth=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("positions")
        ax.grid(axis="y", color=GRID, alpha=0.65)
        ax.set_axisbelow(True)
    axes[0, 0].axvline(50.0, color=CANDIDATE, linestyle="--", linewidth=1.4, label="training threshold")
    axes[0, 0].legend()
    fig.suptitle("Dense labels created signed credit, but most searched moves had little regret", x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_reward_distributions.png")


def figure_training_selection(metrics: list[dict]) -> None:
    epochs = [0]
    eval_rows = [metrics[0]]
    for row in metrics[1:]:
        epochs.append(int(row["epoch"]))
        eval_rows.append(row["eval"])
    expected = np.asarray([float(row["expected_regret"]) for row in eval_rows])
    objective = np.asarray([float(row["signed_logprob"]) for row in eval_rows])
    tv = np.asarray([float(row["tv_base"]) for row in eval_rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.1))
    expected_delta = (expected - expected[0]) * 1_000_000.0
    objective_delta = (objective - objective[0]) * 1_000_000.0
    axes[0].plot(epochs, expected_delta, marker="o", color=SEARCH, linewidth=2.3)
    axes[0].axhline(0, color=INK, linewidth=1)
    axes[0].set_title("Teacher regret barely moved", loc="left")
    axes[0].set_ylabel("change from base, micro-value")
    axes[0].set_xticks(epochs)
    axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, objective_delta, marker="o", color=CANDIDATE, linewidth=2.3)
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[1].set_title("Held-out RL objective worsened", loc="left")
    axes[1].set_ylabel("change in held-out A log pi, x1e-6")
    axes[1].set_xticks(epochs)
    axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, tv, marker="o", color=BASE, linewidth=2.3)
    axes[2].set_title("KL anchor kept drift tiny", loc="left")
    axes[2].set_ylabel("mean TV to tactical base")
    axes[2].set_xticks(epochs)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylim(-0.00002, 0.00036)
    axes[2].text(0.05, 0.88, "reject ceiling = 0.08\n(off scale)", transform=axes[2].transAxes, color=MUTED)

    for ax in axes:
        ax.grid(color=GRID, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Neither epoch earned promotion, so epoch 0 was restored", x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig3_training_selection.png")


def figure_external_identity() -> None:
    seed23 = read_jsonl(REPO / "reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.jsonl")
    seed31 = read_jsonl(ROOT / "oracle_process_rl_vs2700_s128_g80_seed31.jsonl")
    scores = [[float(row["score"]) for row in seed23], [float(row["score"]) for row in seed31]]
    means = [float(np.mean(values)) for values in scores]
    intervals = [bootstrap_interval(values, 23 + index) for index, values in enumerate(scores)]
    errors = np.asarray([[mean - low, high - mean] for mean, (low, high) in zip(means, intervals)]).T
    max_diff, changed = checkpoint_difference()

    fig, (identity_ax, gate_ax) = plt.subplots(1, 2, figsize=(13.0, 5.8), gridspec_kw={"width_ratios": [0.9, 1.2]})
    identity_ax.axis("off")
    identity_ax.text(0.02, 0.90, "Checkpoint identity", fontsize=15, weight="bold")
    identity_ax.text(0.02, 0.60, f"{max_diff:.1f}", fontsize=54, weight="bold", color=BASE)
    identity_ax.text(0.02, 0.50, "maximum absolute tensor difference", color=MUTED)
    identity_ax.text(0.02, 0.30, f"{changed} changed tensors", fontsize=18, weight="bold")
    identity_ax.text(0.02, 0.16, "best epoch = 0\nfinal output = tactical base", color=MUTED, linespacing=1.5)

    labels = ["tactical base\nseed 23", "oracle output\nseed 31"]
    bars = gate_ax.bar(labels, means, yerr=errors, capsize=6, color=[BASE, SEARCH], width=0.58)
    gate_ax.set_ylim(0, 0.45)
    gate_ax.set_ylabel("score vs Leela-2700, 128 sims")
    gate_ax.set_title("Different gate samples, identical weights", loc="left")
    gate_ax.grid(axis="y", color=GRID, alpha=0.7)
    gate_ax.set_axisbelow(True)
    for bar, value in zip(bars, means):
        gate_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", weight="bold")
    gate_ax.text(
        0.50,
        0.96,
        "The seed-31 score is not an RL regression.\nThe selected checkpoint contains no trained update.",
        transform=gate_ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": PAPER, "edgecolor": GRID, "alpha": 0.95},
    )
    fig.suptitle("External play cannot show a training effect when the selected model is unchanged", x=0.01, ha="left", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig4_external_identity.png")


def main() -> None:
    setup()
    records = read_jsonl(LABELED)
    metrics = read_jsonl(METRICS)
    figure_signal_funnel(records)
    figure_reward_distributions(records)
    figure_training_selection(metrics)
    figure_external_identity()
    print(f"wrote 4 figures to {ROOT}")


if __name__ == "__main__":
    main()
