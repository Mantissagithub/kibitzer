from __future__ import annotations

import argparse
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


PUCT_COLOR = "#4e79a7"
GUMBEL_COLOR = "#f28e2b"
WIN_COLOR = "#2ca02c"
DRAW_COLOR = "#ffbf00"
LOSS_COLOR = "#d62728"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 240,
    }
)


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    out = []
    for index, value in enumerate(values, start=1):
        total += value
        out.append(total / index)
    return out


def _summary(rows: list[dict]) -> dict[str, float | int]:
    scores = [float(row["score"]) for row in rows]
    return {
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "rate": sum(scores) / len(scores),
    }


def _elo(score: float, opponent: int) -> float:
    return opponent + 400.0 * math.log10(score / (1.0 - score))


def _validate_pairing(puct: list[dict], gumbel: list[dict]) -> None:
    if not puct or len(puct) != len(gumbel):
        raise ValueError("both gates must contain the same non-zero number of games")
    for left, right in zip(puct, gumbel, strict=True):
        if left.get("opening_fen") != right.get("opening_fen"):
            raise ValueError("gate openings do not match")
        if left.get("network_white") != right.get("network_white"):
            raise ValueError("gate colors do not match")


def _plot_curve(puct: list[dict], gumbel: list[dict], output_dir: Path) -> Path:
    puct_scores = [float(row["score"]) for row in puct]
    gumbel_scores = [float(row["score"]) for row in gumbel]
    games = list(range(1, len(puct) + 1))
    puct_curve = _cumulative(puct_scores)
    gumbel_curve = _cumulative(gumbel_scores)
    delta_curve = _cumulative([
        candidate - control
        for control, candidate in zip(puct_scores, gumbel_scores, strict=True)
    ])

    fig, (ax_score, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (1.5, 1.0)},
    )
    ax_score.plot(games, puct_curve, color=PUCT_COLOR, linewidth=2.0, label="PUCT")
    ax_score.plot(games, gumbel_curve, color=GUMBEL_COLOR, linewidth=2.0, label="Gumbel")
    ax_score.scatter([games[-1]], [puct_curve[-1]], color=PUCT_COLOR, s=42, zorder=4)
    ax_score.scatter([games[-1]], [gumbel_curve[-1]], color=GUMBEL_COLOR, s=42, zorder=4)
    ax_score.annotate(f"{puct_curve[-1]:.3f}", (games[-1], puct_curve[-1]), xytext=(-8, 8), textcoords="offset points", ha="right", color=PUCT_COLOR)
    ax_score.annotate(f"{gumbel_curve[-1]:.3f}", (games[-1], gumbel_curve[-1]), xytext=(-8, -15), textcoords="offset points", ha="right", color=GUMBEL_COLOR)
    ax_score.set_ylim(0.0, 0.65)
    ax_score.set_ylabel("cumulative score rate")
    ax_score.set_title("External search gate: Gumbel finishes below normal PUCT")
    ax_score.legend(loc="upper right", framealpha=0.9)

    ax_delta.plot(games, delta_curve, color=GUMBEL_COLOR, linewidth=2.0)
    ax_delta.axhline(0.0, color="#555555", linewidth=1.0)
    ax_delta.axhline(0.03, color=WIN_COLOR, linewidth=1.2, linestyle="--", label="continue threshold (+0.03)")
    ax_delta.axhline(-0.03, color=LOSS_COLOR, linewidth=1.2, linestyle="--", label="stop threshold (-0.03)")
    ax_delta.fill_between(games, -0.03, 0.03, color="#999999", alpha=0.08)
    ax_delta.scatter([games[-1]], [delta_curve[-1]], color=LOSS_COLOR, s=42, zorder=4)
    ax_delta.annotate(f"{delta_curve[-1]:+.3f}", (games[-1], delta_curve[-1]), xytext=(-8, -15), textcoords="offset points", ha="right", color=LOSS_COLOR)
    ax_delta.set_ylim(-0.30, 0.30)
    ax_delta.set_xlabel("paired game number")
    ax_delta.set_ylabel("Gumbel minus PUCT")
    ax_delta.legend(loc="lower right", framealpha=0.9)

    fig.text(
        0.5,
        0.01,
        "40 paired openings, 128 network evaluations per move, seed 23, vs Leela-2700. "
        "Final paired delta -0.0375; bootstrap 95% CI [-0.200, +0.125].",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "fig_gumbel_score_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_summary(
    puct: list[dict],
    gumbel: list[dict],
    opponent_elo: int,
    output_dir: Path,
) -> Path:
    summaries = [_summary(puct), _summary(gumbel)]
    labels = ["normal PUCT", "Gumbel"]
    rates = [float(row["rate"]) for row in summaries]
    colors = [PUCT_COLOR, GUMBEL_COLOR]

    fig, (ax_rate, ax_wdl) = plt.subplots(1, 2, figsize=(11, 5.3))
    bars = ax_rate.bar(labels, rates, color=colors, width=0.52, zorder=3)
    ax_rate.axhline(
        rates[0] + 0.03,
        color=WIN_COLOR,
        linewidth=1.2,
        linestyle="--",
        label=f"continue bar = {rates[0] + 0.03:.3f}",
    )
    for bar, rate in zip(bars, rates, strict=True):
        elo = _elo(rate, opponent_elo)
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.012,
            f"{rate:.3f}\n~{elo:.0f} Elo",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_rate.set_ylim(0.0, 0.45)
    ax_rate.set_ylabel("score rate vs Leela-2700")
    ax_rate.set_title("Gumbel misses the continue threshold")
    ax_rate.legend(loc="upper right", framealpha=0.9)

    wins = [int(row["wins"]) for row in summaries]
    draws = [int(row["draws"]) for row in summaries]
    losses = [int(row["losses"]) for row in summaries]
    x = range(len(labels))
    ax_wdl.bar(x, wins, color=WIN_COLOR, label="wins")
    ax_wdl.bar(x, draws, bottom=wins, color=DRAW_COLOR, label="draws")
    bottoms = [win + draw for win, draw in zip(wins, draws, strict=True)]
    ax_wdl.bar(x, losses, bottom=bottoms, color=LOSS_COLOR, label="losses")
    ax_wdl.set_xticks(list(x), labels)
    ax_wdl.set_ylim(0, len(puct) + 3)
    ax_wdl.set_ylabel(f"games (of {len(puct)})")
    ax_wdl.set_title("Same losses, but Gumbel gives up three wins")
    ax_wdl.legend(loc="upper right", framealpha=0.9)
    for index, (win, draw, loss) in enumerate(zip(wins, draws, losses, strict=True)):
        ax_wdl.text(index, win / 2, str(win), ha="center", va="center", color="white", fontweight="bold")
        ax_wdl.text(index, win + draw / 2, str(draw), ha="center", va="center", color="#222222", fontweight="bold")
        ax_wdl.text(index, win + draw + loss / 2, str(loss), ha="center", va="center", color="white", fontweight="bold")

    fig.suptitle("D61 Gumbel search gate: rejected after the 40-game paired probe", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "PUCT 6W/11D/23L (0.2875, ~2542) vs Gumbel 3W/14D/23L (0.2500, ~2509). "
        "No 80-game confirmation and no Gumbel self-play training.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = output_dir / "fig_gumbel_wdl_elo.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puct", type=Path, required=True)
    parser.add_argument("--gumbel", type=Path, required=True)
    parser.add_argument("--opponent-elo", type=int, default=2700)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    puct = _read(args.puct)
    gumbel = _read(args.gumbel)
    _validate_pairing(puct, gumbel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_curve(puct, gumbel, args.output_dir),
        _plot_summary(puct, gumbel, args.opponent_elo, args.output_dir),
    ]
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
