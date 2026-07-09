from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        "savefig.dpi": 180,
    }
)


@dataclass(frozen=True)
class EvalRun:
    key: str
    label: str
    path: Path
    sims: int
    note: str


@dataclass(frozen=True)
class EvalSummary:
    run: EvalRun
    games: int
    wins: int
    draws: int
    losses: int
    score: float

    @property
    def rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.score / self.games

    @property
    def elo_delta(self) -> float:
        return elo_delta_from_score(self.rate)

    @property
    def elo(self) -> float:
        return 2700.0 + self.elo_delta


@dataclass(frozen=True)
class BufferSummary:
    label: str
    path: Path
    rows: int
    mean_regret: float
    p50_regret: float
    p90_regret: float
    max_regret: float
    mean_outcome_gap: float


EVAL_RUNS = [
    EvalRun(
        "base",
        "comp base (20g)",
        Path("reports/regret/comp_base_vs2700_s128.jsonl"),
        128,
        "direct base rerun",
    ),
    EvalRun(
        "base_g80",
        "comp base (80g)",
        Path("reports/regret/comp_base_vs2700_s128_g80_seed17.jsonl"),
        128,
        "paired 80-game gate",
    ),
    EvalRun(
        "az1",
        "AZ iter 1",
        Path("reports/az/az_iter1_vs2700.jsonl"),
        64,
        "old AZ self-play check",
    ),
    EvalRun(
        "regret",
        "outcome-regret repair",
        Path("reports/regret/regret_repair_vs2700_s128.jsonl"),
        128,
        "value/outcome-heavy repair",
    ),
    EvalRun(
        "policy",
        "policy-regret (20g)",
        Path("reports/regret/policy_regret_repair_vs2700_s128.jsonl"),
        128,
        "first policy-regret probe",
    ),
    EvalRun(
        "policy_g80",
        "policy-regret (80g)",
        Path("reports/regret/policy_regret_repair_vs2700_s128_g80_seed17.jsonl"),
        128,
        "paired 80-game gate",
    ),
    EvalRun(
        "policy_g80_s23",
        "policy-regret (80g s23)",
        Path("reports/regret/policy_regret_repair_vs2700_s128_g80_seed23.jsonl"),
        128,
        "paired 80-game gate, seed 23",
    ),
    EvalRun(
        "policy_big",
        "bigger policy-regret",
        Path("reports/regret/policy_regret_repair_bigger_vs2700_s128.jsonl"),
        128,
        "broader buffer, longer train",
    ),
    EvalRun(
        "regret_start",
        "regret-start self-play",
        Path("reports/regret_start/regret_start_az_vs2700_s128.jsonl"),
        128,
        "targeted self-play",
    ),
    EvalRun(
        "tactical",
        "tactical repair (20g)",
        Path("reports/tactical_repair/tactical_repair_vs2700_s128.jsonl"),
        128,
        "first tactical probe",
    ),
    EvalRun(
        "tactical_g80",
        "tactical repair (80g)",
        Path("reports/tactical_repair/tactical_repair_vs2700_s128_g80_seed17.jsonl"),
        128,
        "paired 80-game gate",
    ),
    EvalRun(
        "tactical_r1_g80_s23",
        "tactical R1 (80g s23)",
        Path("reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.jsonl"),
        128,
        "paired 80-game gate, seed 23",
    ),
    EvalRun(
        "tactical_r2_g80_s23",
        "tactical R2 (80g s23)",
        Path("reports/tactical_repair/tactical_repair_r2_vs2700_s128_g80_seed23.jsonl"),
        128,
        "paired 80-game gate, seed 23",
    ),
]

BUFFER_RUNS = [
    ("outcome-heavy regret buffer", Path("runs/regret/az1_sf12.jsonl")),
    ("policy-regret buffer", Path("runs/regret/az12_policy_regret_sf12.jsonl")),
    ("bigger policy-regret buffer", Path("runs/regret/az12_policy_regret_sf12_bigger.jsonl")),
]

CHECKPOINT_RUNS = [
    ("outcome-regret repair", Path("runs/regret/regret_repair.pt")),
    ("policy-regret repair", Path("runs/regret/policy_regret_repair.pt")),
    ("bigger policy-regret", Path("runs/regret/policy_regret_repair_bigger.pt")),
    ("regret-start self-play", Path("runs/regret_start/regret_start_az.pt")),
    ("tactical repair", Path("runs/tactical/tactical_repair.pt")),
    ("tactical repair R2", Path("runs/tactical/tactical_repair_r2.pt")),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def elo_delta_from_score(score_rate: float) -> float:
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def _format_elo(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.0f}"


def summarize_eval(run: EvalRun) -> EvalSummary:
    rows = _read_jsonl(run.path)
    wins = sum(1 for row in rows if row.get("score") == 1.0)
    draws = sum(1 for row in rows if row.get("score") == 0.5)
    losses = sum(1 for row in rows if row.get("score") == 0.0)
    score = sum(float(row.get("score", 0.0)) for row in rows)
    return EvalSummary(run, wins + draws + losses, wins, draws, losses, score)


def summarize_buffer(label: str, path: Path) -> BufferSummary:
    rows = _read_jsonl(path)
    regrets = [float(row.get("policy_regret", 0.0)) for row in rows]
    gaps = [float(row.get("outcome_gap", 0.0)) for row in rows]
    return BufferSummary(
        label=label,
        path=path,
        rows=len(rows),
        mean_regret=statistics.fmean(regrets) if regrets else 0.0,
        p50_regret=_percentile(regrets, 0.50),
        p90_regret=_percentile(regrets, 0.90),
        max_regret=max(regrets) if regrets else 0.0,
        mean_outcome_gap=statistics.fmean(gaps) if gaps else 0.0,
    )


def _load_checkpoint_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    # local checkpoints are generated by this repo, so full pickle loading is
    # acceptable here. torch 2.6 defaults to weights_only=True, which hides the
    # metric dicts we need for the report.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("eval_metrics", {}) if isinstance(checkpoint, dict) else {}
    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def _annotate_bars(ax: plt.Axes, bars: Any, values: list[float], fmt: str = "{:.3f}") -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_external_scores(summaries: list[EvalSummary], output_dir: Path) -> Path:
    labels = [summary.run.label for summary in summaries]
    rates = [summary.rate for summary in summaries]
    colors = ["#2ca02c" if summary.run.key == "tactical_r1_g80_s23" else "#8c8c8c" for summary in summaries]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    bars = ax.bar(labels, rates, color=colors)
    _annotate_bars(ax, bars, rates)
    ax.axhline(0.5, color="#333333", linewidth=1, linestyle="--", label="break-even")
    ax.set_ylim(0, max(0.55, max(rates, default=0) + 0.08))
    ax.set_ylabel("match score rate")
    ax.set_title("External gate vs Leela/Maia-2700 proxy")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Source: reports/regret/*.jsonl, reports/regret_start/*.jsonl, reports/tactical_repair/*.jsonl, reports/az/az_iter1_vs2700.jsonl",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "fig1_external_gate_scores.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_wdl(summaries: list[EvalSummary], output_dir: Path) -> Path:
    labels = [summary.run.label for summary in summaries]
    wins = [summary.wins for summary in summaries]
    draws = [summary.draws for summary in summaries]
    losses = [summary.losses for summary in summaries]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.bar(labels, wins, label="wins", color="#2ca02c")
    ax.bar(labels, draws, bottom=wins, label="draws", color="#ffbf00")
    bottoms = [w + d for w, d in zip(wins, draws, strict=True)]
    ax.bar(labels, losses, bottom=bottoms, label="losses", color="#d62728")
    ax.set_ylabel("games")
    ax.set_title("Win/draw/loss shape for each external gate")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "fig2_wdl_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_elo_estimates(summaries: list[EvalSummary], output_dir: Path) -> Path:
    labels = [summary.run.label for summary in summaries]
    elos = [summary.elo for summary in summaries]
    colors = ["#2ca02c" if summary.run.key == "tactical_r1_g80_s23" else "#8c8c8c" for summary in summaries]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    bars = ax.bar(labels, elos, color=colors)
    _annotate_bars(ax, bars, elos, fmt="{:.0f}")
    ax.axhline(2700, color="#333333", linewidth=1, linestyle="--", label="opponent label")
    ax.set_ylabel("implied Elo")
    ax.set_title("Implied Elo from score vs 2700: 400*log10(score/(1-score))")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "fig5_implied_elo.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_buffer_quality(buffers: list[BufferSummary], output_dir: Path) -> Path:
    labels = [buffer.label for buffer in buffers]
    rows = [buffer.rows for buffer in buffers]
    means = [buffer.mean_regret for buffer in buffers]
    p90s = [buffer.p90_regret for buffer in buffers]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax_count, ax_regret = axes
    count_bars = ax_count.bar(labels, rows, color="#4c78a8")
    _annotate_bars(ax_count, count_bars, [float(v) for v in rows], fmt="{:.0f}")
    ax_count.set_title("Labeled repair positions")
    ax_count.set_ylabel("rows")
    ax_count.tick_params(axis="x", rotation=18)

    x = range(len(labels))
    ax_regret.bar([i - 0.18 for i in x], means, width=0.36, label="mean", color="#72b7b2")
    ax_regret.bar([i + 0.18 for i in x], p90s, width=0.36, label="p90", color="#f58518")
    ax_regret.set_xticks(list(x), labels, rotation=18)
    ax_regret.set_ylabel("policy regret")
    ax_regret.set_title("Buffer sharpness")
    ax_regret.legend()
    fig.tight_layout()
    path = output_dir / "fig3_buffer_quality.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_checkpoint_metrics(output_dir: Path) -> Path:
    rows = [(label, path, _load_checkpoint_metrics(path)) for label, path in CHECKPOINT_RUNS]
    labels = [row[0] for row in rows]
    policy_losses = [row[2].get("policy_loss", 0.0) for row in rows]
    top1s = [row[2].get("policy_top1", 0.0) for row in rows]
    anchor_kls = [row[2].get("anchor_kl", 0.0) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        (axes[0], policy_losses, "policy CE", "#4c78a8"),
        (axes[1], top1s, "held-out top-1", "#54a24b"),
        (axes[2], anchor_kls, "anchor KL", "#e45756"),
    ]
    for ax, values, title, color in panels:
        bars = ax.bar(labels, values, color=color)
        _annotate_bars(ax, bars, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Offline checkpoint metrics are diagnostic only")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / "fig4_checkpoint_offline_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_readme(
    output_dir: Path,
    summaries: list[EvalSummary],
    buffers: list[BufferSummary],
    figures: list[Path],
) -> Path:
    best = max(summaries, key=lambda row: row.rate)
    paired = [summary for summary in summaries if summary.games == 80]
    paired_best = max(paired, key=lambda row: row.rate) if paired else best
    lines = [
        "# Repair evaluation report",
        "",
        "This folder is the decision report for the recent repair branches. It uses the cheap external gates we already ran, plus the local repair buffers/checkpoints.",
        "",
        "## Verdict",
        "",
        f"- Current best: `{best.run.label}` at score rate `{best.rate:.3f}` ({best.wins}W/{best.draws}D/{best.losses}L).",
        f"- Best paired 80-game gate: `{paired_best.run.label}` at score rate `{paired_best.rate:.3f}`, implied Elo `{_format_elo(paired_best.elo)}`.",
        "- Tactical R1 remains the current best; tactical R2 passed held-out top1 but regressed externally, so do not promote R2.",
        "- The broader policy-regret buffer did not beat the first policy-regret checkpoint.",
        "- Regret-start self-play fell back to the comp-base score band, so it stays closed unless continuation states get external teacher labels.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- ![{figure.stem}]({figure.name})")
    lines.extend(
        [
            "",
            "## External gates",
            "",
            "| run | sims | games | W/D/L | score rate | Elo delta | implied Elo | source | note |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.run.label} | {summary.run.sims} | {summary.games} | "
            f"{summary.wins}/{summary.draws}/{summary.losses} | {summary.rate:.3f} | "
            f"{_format_elo(summary.elo_delta)} | {_format_elo(summary.elo)} | "
            f"`{summary.run.path}` | {summary.run.note} |"
        )
    lines.extend(
        [
            "",
            "## Repair buffers",
            "",
            "| buffer | rows | mean regret | p50 regret | p90 regret | max regret | mean outcome gap | source |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for buffer in buffers:
        lines.append(
            f"| {buffer.label} | {buffer.rows} | {buffer.mean_regret:.3f} | "
            f"{buffer.p50_regret:.3f} | {buffer.p90_regret:.3f} | "
            f"{buffer.max_regret:.3f} | {buffer.mean_outcome_gap:.3f} | `{buffer.path}` |"
        )
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "uv run python scripts/plot_regret_policy.py",
            "```",
            "",
            "## Cheap gate monitor",
            "",
            "Use this while a small match is writing JSONL:",
            "",
            "```bash",
            "uv run python scripts/monitor_match_jsonl.py --path reports/regret/next_eval.jsonl --expected-games 20 --poll-seconds 30",
            "```",
            "",
            "One-shot check for an already finished run:",
            "",
            "```bash",
            "uv run python scripts/monitor_match_jsonl.py --path reports/regret/policy_regret_repair_vs2700_s128.jsonl --once",
            "```",
            "",
        ]
    )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/repair_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_eval(run) for run in EVAL_RUNS]
    buffers = [summarize_buffer(label, path) for label, path in BUFFER_RUNS]
    figures = [
        plot_external_scores(summaries, args.output_dir),
        plot_wdl(summaries, args.output_dir),
        plot_buffer_quality(buffers, args.output_dir),
        plot_checkpoint_metrics(args.output_dir),
        plot_elo_estimates(summaries, args.output_dir),
    ]
    readme = write_readme(args.output_dir, summaries, buffers, figures)
    print(f"wrote {readme}")
    for figure in figures:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
