# D63: PUCT simulation-count sweep vs a fixed Leela/Maia-2700 nodes=1 opponent.
# reads reports/sims_sweep/*.jsonl, redraws the two figures, and rewrites the
# folder report. keep this data-driven because the sweep is expensive to rerun.
# run: uv run python scripts/plot_sims_sweep.py

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kibitzer-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

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

CONTROL_SIMS = 128
WIN_COLOR = "#59a14f"
DRAW_COLOR = "#bab0ac"
LOSS_COLOR = "#e15759"
LINE_COLOR = "#4e79a7"
EMPH_COLOR = "#e15759"


@dataclass(frozen=True)
class SweepRow:
    sims: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elo_delta: float
    implied_elo: float
    source: Path


def elo_delta_from_score(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score / (1.0 - score))


def format_elo(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{round(value):d}"


def read_rows(out_dir: Path) -> list[SweepRow]:
    rows: list[SweepRow] = []
    pattern = re.compile(r"_s(\d+)_g\d+_seed\d+\.jsonl$")
    for path in sorted(out_dir.glob("kibitzer_vs2700_s*_g*_seed*.jsonl")):
        match = pattern.search(path.name)
        if match is None:
            continue
        sims = int(match.group(1))
        games = wins = draws = losses = 0
        score_sum = 0.0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                score = float(row["score"])
                games += 1
                score_sum += score
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1
        if games == 0:
            continue
        score = score_sum / games
        elo_delta = elo_delta_from_score(score)
        rows.append(
            SweepRow(
                sims=sims,
                games=games,
                wins=wins,
                draws=draws,
                losses=losses,
                score=score,
                elo_delta=elo_delta,
                implied_elo=2700.0 + elo_delta,
                source=path,
            )
        )
    return sorted(rows, key=lambda row: row.sims)


def plot_score_vs_sims(ax: plt.Axes, rows: list[SweepRow]) -> None:
    sims = [row.sims for row in rows]
    scores = [row.score for row in rows]
    best = max(rows, key=lambda row: row.score)
    control = next(row for row in rows if row.sims == CONTROL_SIMS)

    ax.plot(sims, scores, color=LINE_COLOR, linewidth=1.8, marker="o", markersize=7, zorder=2)
    ax.plot(best.sims, best.score, marker="o", markersize=12, color=EMPH_COLOR, zorder=4)
    ax.annotate(
        f"{best.score:.3f}, {best.score - control.score:+.3f} over control",
        xy=(best.sims, best.score),
        xytext=(max(CONTROL_SIMS * 2, best.sims * 0.58), min(0.95, best.score + 0.08)),
        fontsize=8.5,
        fontweight="bold",
        color=EMPH_COLOR,
        arrowprops={"arrowstyle": "-", "color": EMPH_COLOR, "linewidth": 0.8},
    )

    ax.axhline(control.score, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(
        min(sims),
        control.score + 0.02,
        f"control ({CONTROL_SIMS} sims) = {control.score:.3f}",
        va="bottom",
        ha="left",
        fontsize=7.5,
        color="#555555",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sims)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(min(sims) * 0.88, max(sims) * 1.13)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("PUCT simulations per move")
    ax.set_ylabel("score rate vs Leela-2700 nodes=1")
    ax.set_title("(a) search depth vs score rate", pad=10)


def plot_wdl_bars(ax: plt.Axes, rows: list[SweepRow]) -> None:
    x = range(len(rows))
    wins = [row.wins for row in rows]
    draws = [row.draws for row in rows]
    losses = [row.losses for row in rows]

    ax.bar(x, wins, color=WIN_COLOR, label="win", zorder=2)
    ax.bar(x, draws, bottom=wins, color=DRAW_COLOR, label="draw", zorder=2)
    bottoms = [w + d for w, d in zip(wins, draws)]
    ax.bar(x, losses, bottom=bottoms, color=LOSS_COLOR, label="loss", zorder=2)

    for i, row in enumerate(rows):
        if row.wins:
            ax.text(i, row.wins / 2, str(row.wins), ha="center", va="center", fontsize=8, color="white")
        if row.draws:
            ax.text(i, row.wins + row.draws / 2, str(row.draws), ha="center", va="center", fontsize=8, color="#333333")
        if row.losses:
            ax.text(
                i,
                row.wins + row.draws + row.losses / 2,
                str(row.losses),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(row.sims) for row in rows])
    ax.set_ylim(0, max(row.games for row in rows))
    ax.set_xlabel("PUCT simulations per move")
    ax.set_ylabel("games")
    ax.set_title("(b) win/draw/loss breakdown", pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def plot_elo_vs_sims(ax: plt.Axes, rows: list[SweepRow]) -> None:
    sims = [row.sims for row in rows]
    elos = [row.implied_elo for row in rows]
    best = max(rows, key=lambda row: row.score)

    ax.plot(sims, elos, color=LINE_COLOR, linewidth=1.8, marker="o", markersize=7, zorder=3)
    ax.plot(best.sims, best.implied_elo, marker="o", markersize=12, color=EMPH_COLOR, zorder=4)
    for row in rows:
        ax.annotate(
            format_elo(row.implied_elo),
            xy=(row.sims, row.implied_elo),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sims)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(min(sims) * 0.88, max(sims) * 1.13)
    ax.set_ylim(min(elos) - 130, max(elos) + 130)
    ax.set_xlabel("PUCT simulations per move")
    ax.set_ylabel("implied Elo vs Leela-2700 nodes=1")
    ax.set_title("D63: implied Elo vs PUCT simulation count", pad=10)


def write_report(out_dir: Path, rows: list[SweepRow]) -> Path:
    control = next(row for row in rows if row.sims == CONTROL_SIMS)
    best = max(rows, key=lambda row: row.score)
    table = [
        "| sims | games | W/D/L | score | Elo delta | implied proxy Elo | source |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row.sims} | {row.games} | {row.wins}/{row.draws}/{row.losses} | "
            f"{row.score:.3f} | {format_elo(row.elo_delta)} | {format_elo(row.implied_elo)} | "
            f"`{row.source.name}` |"
        )

    lines = [
        "# sims_sweep report",
        "",
        "D63 tests whether the current best checkpoint is actually compute-starved at inference time.",
        "The model is fixed: `runs/tactical/tactical_repair.pt`. Only the PUCT simulation count changes.",
        "The opponent is the same Leela/Maia-2700 proxy at nodes=1, so these numbers are a search-budget yardstick, not an intrinsic model Elo.",
        "",
        "![score and wdl](fig_sims_sweep.png)",
        "",
        "![implied elo](fig_sims_elo.png)",
        "",
        "## results",
        "",
        *table,
        "",
        "## read",
        "",
        f"- 128 sims is the control: {control.wins}W/{control.draws}D/{control.losses}L, score {control.score:.3f}.",
        f"- {best.sims} sims is the winner: {best.wins}W/{best.draws}D/{best.losses}L, score {best.score:.3f}.",
        f"- The best point is {best.score - control.score:+.3f} score rate over the 128-sim control.",
        "- The jump is inference-time search, not learning. It says the checkpoint had latent strength that shallow search was failing to extract.",
        "- Because the opponent is one-node Leela/Maia, the implied Elo should be read as a proxy scale only.",
        "",
        "## next",
        "",
        "Run a budgeted 1024/2048 confirmation on a rented GPU, ideally against 2700 plus stronger Leela checkpoints.",
        "Keep the same paired openings and colors before claiming a new default search budget.",
    ]
    report_path = out_dir / "README.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "reports" / "sims_sweep"
    rows = read_rows(out_dir)
    if not rows:
        raise SystemExit(f"no sims sweep jsonl files found under {out_dir}")
    if not any(row.sims == CONTROL_SIMS for row in rows):
        raise SystemExit(f"missing {CONTROL_SIMS}-sim control in {out_dir}")

    footer = (
        "40 games/point, seed 23, same checkpoint and opponent; "
        "Leela/Maia-2700 runs at nodes=1, so this is asymmetric search-budget evidence."
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_score_vs_sims(ax_a, rows)
    plot_wdl_bars(ax_b, rows)
    fig.suptitle("D63: PUCT simulation count vs Leela-2700 nodes=1", fontsize=12.5)
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    main_path = out_dir / "fig_sims_sweep.png"
    fig.savefig(main_path)
    plt.close(fig)
    print(f"wrote {main_path}")

    fig2, ax = plt.subplots(figsize=(6.6, 5.6))
    plot_elo_vs_sims(ax, rows)
    fig2.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555555")
    fig2.tight_layout(rect=(0, 0.11, 1, 1))
    elo_path = out_dir / "fig_sims_elo.png"
    fig2.savefig(elo_path)
    plt.close(fig2)
    print(f"wrote {elo_path}")

    print(f"wrote {write_report(out_dir, rows)}")


if __name__ == "__main__":
    main()
