# three-panel scaling-law result figure for the s2 shaw 100m run.
# panel a: offline top-1 accuracy + value mse vs training positions, shaw vs
# the absolute-encoding reference curve.
# panel b: in-loop play score vs stockfish-1900 (64 sims) vs training positions.
# panel c: estimated elo per checkpoint at 256 sims with 95% ci error bars.
# reads reports/scaling_law/figure_data_100M.json
# run: uv run python scripts/plot_scaling_100M.py

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


def load_data(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_data_scaling(ax: plt.Axes, data: dict) -> None:
    shaw = data["scaling_curve_shaw"]
    ref = data["scaling_curve_absolute_reference"]

    shaw_x = [row["positions"] for row in shaw]
    shaw_top1 = [row["top1"] * 100 for row in shaw]
    shaw_mse = [row["value_mse"] for row in shaw]
    ref_x = [row["positions"] for row in ref]
    ref_top1 = [row["top1"] * 100 for row in ref]

    # hero series: shaw top-1 accuracy, left axis.
    (line_top1,) = ax.plot(
        shaw_x,
        shaw_top1,
        color="#1f77b4",
        marker="o",
        linewidth=1.8,
        zorder=3,
        label="top-1 accuracy (shaw)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("training positions")
    ax.set_ylabel("top-1 accuracy (%)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    # faint reference curve showing the pre-shaw absolute-encoding baseline.
    ax.plot(
        ref_x,
        ref_top1,
        color="#1f77b4",
        marker="o",
        markerfacecolor="none",
        linewidth=1.2,
        linestyle=":",
        alpha=0.45,
        zorder=2,
        label="absolute-enc (reference)",
    )

    # value mse on a second, muted, dashed axis.
    ax2 = ax.twinx()
    ax2.grid(False)
    (line_mse,) = ax2.plot(
        shaw_x,
        shaw_mse,
        color="#7f7f7f",
        marker="s",
        linewidth=1.4,
        linestyle="--",
        zorder=2,
        label="value MSE (shaw)",
    )
    ax2.set_ylabel("value MSE", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")
    ax2.spines["top"].set_visible(False)

    handles = [line_top1, line_mse, ax.lines[1]]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="lower right", fontsize=7.5, framealpha=0.9)

    ax.set_title("Data scaling: shaw, S2 (15.2M params)", pad=12)


def plot_play_scaling(ax: plt.Axes, data: dict) -> None:
    rows = data["inloop_play_sf1900_64sims"]
    x = [row["positions"] for row in rows]
    y = [row["score"] for row in rows]

    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(x[0], 0.5, "even (0.5)", va="bottom", ha="left", fontsize=8, color="#555555")

    ax.plot(
        x,
        y,
        color="#d62728",
        marker="o",
        linewidth=2.2,
        markersize=6,
        zorder=3,
        label="score vs SF-1900",
    )

    ax.set_xscale("log")
    ax.set_xlabel("training positions")
    ax.set_ylabel("score vs Stockfish-1900")
    ax.set_ylim(0, 1.0)
    ax.set_title("In-loop play vs SF-1900 (64 sims, N=20/pt)", pad=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def plot_elo(ax: plt.Axes, data: dict) -> None:
    rows = data["elo_per_checkpoint_256sim_sf1900"]
    labels = [row["checkpoint"] for row in rows]
    elos = [row["elo"] for row in rows]
    lows = [row["elo_ci"][0] for row in rows]
    highs = [row["elo_ci"][1] for row in rows]
    yerr_lower = [e - lo for e, lo in zip(elos, lows, strict=True)]
    yerr_upper = [hi - e for e, hi in zip(elos, highs, strict=True)]

    x = range(len(labels))
    ax.errorbar(
        x,
        elos,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color="#2ca02c",
        ecolor="#2ca02c",
        elinewidth=1.3,
        capsize=5,
        markersize=7,
        zorder=3,
    )

    # annotate the 100m checkpoint.
    ax.annotate(
        f"{elos[-1]} Elo",
        xy=(len(labels) - 1, elos[-1]),
        xytext=(len(labels) - 1.35, elos[-1] + 160),
        fontsize=8.5,
        color="#2ca02c",
        ha="center",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("estimated Elo vs Stockfish")
    ax.set_title("Estimated Elo vs Stockfish (256 sims)", pad=12)
    ax.text(
        0.02,
        0.02,
        "single-anchor N=20 estimates, wide CI",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#777777",
        ha="left",
        va="bottom",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "reports" / "scaling_law" / "figure_data_100M.json"
    output_path = repo_root / "reports" / "scaling_law" / "fig_scaling_100M.png"

    data = load_data(data_path)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_data_scaling(ax_a, data)
    plot_play_scaling(ax_b, data)
    plot_elo(ax_c, data)

    fig.suptitle(
        "Kibitzer S2 shaw: scaling data lifts offline accuracy AND play strength",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.005,
        "15.2M-param attention-first transformer, shaw relative position encoding; "
        "100M Lichess-Elite positions; play via PUCT search.",
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
