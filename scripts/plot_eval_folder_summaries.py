from __future__ import annotations

import json
import math
import os
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
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
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
class MatchSummary:
    label: str
    path: Path
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


REPORT_FOLDERS = [
    Path("reports/az"),
    Path("reports/regret"),
    Path("reports/regret_start"),
    Path("reports/tactical_repair"),
]


def elo_delta_from_score(score_rate: float) -> float:
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def format_elo(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.0f}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: Path) -> MatchSummary:
    rows = read_jsonl(path)
    wins = sum(1 for row in rows if row.get("score") == 1.0)
    draws = sum(1 for row in rows if row.get("score") == 0.5)
    losses = sum(1 for row in rows if row.get("score") == 0.0)
    score = sum(float(row.get("score", 0.0)) for row in rows)
    label = path.stem.replace("_vs2700", "").replace("_", " ")
    return MatchSummary(label, path, wins + draws + losses, wins, draws, losses, score)


def collect(folder: Path) -> list[MatchSummary]:
    summaries = [summarize(path) for path in sorted(folder.glob("*.jsonl"))]
    return [summary for summary in summaries if summary.games > 0]


def annotate_bars(ax: plt.Axes, bars: Any, values: list[float], fmt: str) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_scores(folder: Path, summaries: list[MatchSummary]) -> Path:
    labels = [summary.label for summary in summaries]
    rates = [summary.rate for summary in summaries]
    colors = ["#2ca02c" if summary.rate == max(rates) else "#8c8c8c" for summary in summaries]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    bars = ax.bar(labels, rates, color=colors)
    annotate_bars(ax, bars, rates, "{:.3f}")
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1, label="break-even")
    ax.set_ylim(0, max(0.55, max(rates) + 0.08))
    ax.set_ylabel("score rate")
    ax.set_title(f"{folder.name}: external gate score")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = folder / "fig_external_scores.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_wdl(folder: Path, summaries: list[MatchSummary]) -> Path:
    labels = [summary.label for summary in summaries]
    wins = [summary.wins for summary in summaries]
    draws = [summary.draws for summary in summaries]
    losses = [summary.losses for summary in summaries]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(labels, wins, label="wins", color="#2ca02c")
    ax.bar(labels, draws, bottom=wins, label="draws", color="#ffbf00")
    bottoms = [w + d for w, d in zip(wins, draws, strict=True)]
    ax.bar(labels, losses, bottom=bottoms, label="losses", color="#d62728")
    ax.set_ylabel("games")
    ax.set_title(f"{folder.name}: W/D/L")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = folder / "fig_wdl_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_elo(folder: Path, summaries: list[MatchSummary]) -> Path:
    labels = [summary.label for summary in summaries]
    elos = [summary.elo for summary in summaries]
    colors = ["#2ca02c" if summary.elo == max(elos) else "#8c8c8c" for summary in summaries]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    bars = ax.bar(labels, elos, color=colors)
    annotate_bars(ax, bars, elos, "{:.0f}")
    ax.axhline(2700, color="#333333", linestyle="--", linewidth=1, label="opponent label")
    ax.set_ylabel("implied Elo")
    ax.set_title(f"{folder.name}: implied Elo from score")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = folder / "fig_implied_elo.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_readme(folder: Path, summaries: list[MatchSummary], figures: list[Path]) -> Path:
    best = max(summaries, key=lambda summary: summary.rate)
    lines = [
        f"# {folder.name} eval report",
        "",
        f"Best score here: `{best.label}` at `{best.rate:.3f}` ({best.wins}W/{best.draws}D/{best.losses}L), implied Elo `{format_elo(best.elo)}`.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- ![{figure.stem}]({figure.name})")
    lines.extend(
        [
            "",
            "## Matches",
            "",
            "| run | games | W/D/L | score rate | Elo delta | implied Elo | source |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.label} | {summary.games} | {summary.wins}/{summary.draws}/{summary.losses} | "
            f"{summary.rate:.3f} | {format_elo(summary.elo_delta)} | {format_elo(summary.elo)} | "
            f"`{summary.path.name}` |"
        )
    lines.append("")
    path = folder / "README.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    for folder in REPORT_FOLDERS:
        summaries = collect(folder)
        if not summaries:
            continue
        figures = [
            plot_scores(folder, summaries),
            plot_wdl(folder, summaries),
            plot_elo(folder, summaries),
        ]
        readme = write_readme(folder, summaries, figures)
        print(f"wrote {readme}")
        for figure in figures:
            print(f"wrote {figure}")


if __name__ == "__main__":
    main()
