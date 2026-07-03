"""Generate reports/scaling_law/fig2_data_scaling_curve.png from results.json
and data_arm_s2_20M.json.

Reads the S2 rung (14.89M params, attention-first, LR fixed 1.5e-4) at two
training-data sizes — 5M positions (reports/scaling_law/results.json) and 20M
positions (reports/scaling_law/data_arm_s2_20M.json) — and plots held-out
policy cross-entropy and top-1 move-match vs. training positions (log-x).

The headline comparison: the data-scaling slope (~-0.1425 CE per
data-doubling) is ~10x steeper than the params-scaling slope from fig1
(~-0.0147 CE per params-doubling).

Usage:
    .venv/bin/python scripts/plot_data_scaling_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "savefig.dpi": 150,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "reports" / "scaling_law" / "results.json"
DATA_ARM_PATH = REPO_ROOT / "reports" / "scaling_law" / "data_arm_s2_20M.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "scaling_law" / "fig2_data_scaling_curve.png"

# Fixed categorical slot order (dataviz skill palette), not cycled — matches fig1.
COLOR_CE = "#2a78d6"  # slot 1: blue
COLOR_TOP1 = "#1baf7a"  # slot 2: aqua
COLOR_FIT = "#8a8a86"  # recessive gray for the params-arm reference line
COLOR_VALUE_BAND = "#c9553a"  # slot 3: warm, used sparingly for the value-MSE annotation

# Params-arm slope from fig1 (log-linear fit, CE per params-doubling).
PARAMS_ARM_SLOPE_PER_DOUBLING = -0.0147


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    data_arm = json.loads(DATA_ARM_PATH.read_text(encoding="utf-8"))

    run_5m = next(r for r in results["runs"] if r["tag"] == "S2")
    run_20m = data_arm["runs"][0]

    positions = np.array(
        [run_5m["train_positions"], run_20m["train_positions"]], dtype=float
    )
    ce = np.array([run_5m["eval_policy_ce"], run_20m["eval_policy_ce"]], dtype=float)
    top1 = np.array(
        [run_5m["eval_top1"] * 100.0, run_20m["eval_top1"] * 100.0], dtype=float
    )
    value_mse = np.array(
        [run_5m["eval_value_mse"], run_20m["eval_value_mse"]], dtype=float
    )

    doublings = np.log2(positions[-1] / positions[0])
    ce_drop = ce[0] - ce[-1]
    data_slope_per_doubling = -ce_drop / doublings
    top1_gain = top1[-1] - top1[0]
    steepness_ratio = data_slope_per_doubling / PARAMS_ARM_SLOPE_PER_DOUBLING

    fig, (ax_ce, ax_top1) = plt.subplots(
        2, 1, figsize=(8, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]}
    )

    # --- Top panel: policy cross-entropy vs training positions ---
    # Faint reference line: what the params-arm's shallow slope would look like
    # anchored at the same starting point, for visual contrast only.
    ref_x = np.geomspace(positions.min(), positions.max(), 50)
    ref_y = ce[0] + PARAMS_ARM_SLOPE_PER_DOUBLING * np.log2(ref_x / positions[0])
    ax_ce.plot(
        ref_x,
        ref_y,
        color=COLOR_FIT,
        linewidth=1.5,
        linestyle="--",
        zorder=1,
        label="params-arm slope (reference, same start)",
    )

    ax_ce.plot(
        positions,
        ce,
        marker="o",
        markersize=8,
        linewidth=2,
        color=COLOR_CE,
        zorder=3,
        label="held-out policy CE",
    )
    for x, y in zip(positions, ce):
        ax_ce.annotate(
            f"CE={y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.5,
            color=COLOR_CE,
            fontweight="bold",
        )

    ax_ce.text(
        0.02,
        0.06,
        f"data slope: {data_slope_per_doubling:.4f} CE per data-doubling\n"
        f"vs. params slope: {PARAMS_ARM_SLOPE_PER_DOUBLING:.4f} CE per params-doubling\n"
        f"→ data scaling is ~{steepness_ratio:.1f}x steeper than params scaling",
        transform=ax_ce.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
        fontweight="bold",
    )

    ax_ce.text(
        0.98,
        0.68,
        f"value MSE also improved with data: {value_mse[0]:.4f} → {value_mse[-1]:.4f}\n"
        "(flat across the params arm — data helps value where size did not)",
        transform=ax_ce.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLOR_VALUE_BAND,
        style="italic",
    )

    ax_ce.set_ylabel("held-out policy cross-entropy (nats)")
    ax_ce.legend(loc="lower left", frameon=True, fontsize=9, bbox_to_anchor=(0.02, 0.30))

    # --- Bottom panel: top-1 move-match vs training positions ---
    ax_top1.plot(
        positions,
        top1,
        marker="s",
        markersize=8,
        linewidth=2,
        color=COLOR_TOP1,
        zorder=3,
        label="held-out top-1 move-match",
    )
    for x, y in zip(positions, top1):
        ax_top1.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.5,
            color=COLOR_TOP1,
            fontweight="bold",
        )

    ax_top1.set_xscale("log")
    ax_top1.set_xlabel("training positions (log scale)")
    ax_top1.set_ylabel("top-1 move-match (%)")
    ax_top1.set_ylim(top1.min() - 1.5, top1.max() + 2.5)
    ax_top1.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(
        "Data is the lever: ~10x steeper than params.",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"same 15M model, 4x data → CE {-ce_drop:+.3f} / top-1 {top1_gain:+.1f}pp",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )

    fig.text(
        0.01,
        0.01,
        "source: reports/scaling_law/results.json + data_arm_s2_20M.json — "
        "S2 (14.89M params), LR 1.5e-4, held-out eval on lichess-elite 2025-11",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#444444",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")
    print(f"data_slope_per_doubling={data_slope_per_doubling:.4f}")
    print(f"params_slope_per_doubling={PARAMS_ARM_SLOPE_PER_DOUBLING:.4f}")
    print(f"steepness_ratio={steepness_ratio:.2f}")


if __name__ == "__main__":
    main()
