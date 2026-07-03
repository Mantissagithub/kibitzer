"""Generate reports/scaling_law/fig1_scaling_curve.png from results.json.

Reads reports/scaling_law/results.json ("runs") and plots held-out policy
cross-entropy and top-1 move-match vs. parameter count across the four-rung
attention-first ladder (attention_every=1, 5M positions/rung, context_window=1).

Fits a log-linear trend (CE vs ln(params)) via numpy.polyfit, since scipy is
not installed in the project venv (a power-law fit CE = E_inf + A*params^-alpha
would otherwise be preferred).

Usage:
    .venv/bin/python scripts/plot_scaling_law_curve.py
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
OUTPUT_PATH = REPO_ROOT / "reports" / "scaling_law" / "fig1_scaling_curve.png"

# Fixed categorical slot order (dataviz skill palette), not cycled.
COLOR_CE = "#2a78d6"  # slot 1: blue
COLOR_TOP1 = "#1baf7a"  # slot 2: aqua
COLOR_FIT = "#8a8a86"  # recessive gray for fit line
COLOR_VALUE_BAND = "#c9553a"  # slot 3: warm, used sparingly for the flat-value annotation


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    runs = sorted(payload["runs"], key=lambda r: r["params"])

    tags = [r["tag"] for r in runs]
    params = np.array([r["params"] for r in runs], dtype=float)
    ce = np.array([r["eval_policy_ce"] for r in runs], dtype=float)
    top1 = np.array([r["eval_top1"] * 100.0 for r in runs], dtype=float)
    value_mse = np.array([r["eval_value_mse"] for r in runs], dtype=float)

    # Try a power law CE = e_inf + A * params^-alpha via scipy; fall back to a
    # log-linear trend (CE vs ln(params)) via numpy.polyfit if scipy is absent.
    fit_kind = "log-linear"
    try:
        from scipy.optimize import curve_fit  # type: ignore

        def power_law(p, e_inf, a, alpha):
            return e_inf + a * np.power(p, -alpha)

        popt, _ = curve_fit(
            power_law, params, ce, p0=(2.2, 1.0, 0.1), maxfev=20000
        )
        e_inf, a_coef, alpha = popt
        fit_kind = "power-law"
        fit_params = np.geomspace(params.min() * 0.85, params.max() * 1.15, 200)
        fit_ce = power_law(fit_params, *popt)
    except ImportError:
        b_coef, a_coef_lin = np.polyfit(np.log(params), ce, 1)
        fit_params = np.geomspace(params.min() * 0.85, params.max() * 1.15, 200)
        fit_ce = a_coef_lin + b_coef * np.log(fit_params)
        # CE reduction per params-doubling
        per_doubling = -b_coef * np.log(2)

    ratio = params[-1] / params[0]
    ce_drop = ce[0] - ce[-1]
    top1_gain = top1[-1] - top1[0]

    fig, (ax_ce, ax_top1) = plt.subplots(
        2, 1, figsize=(8, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]}
    )

    # --- Top panel: policy cross-entropy vs params ---
    ax_ce.plot(
        fit_params,
        fit_ce,
        color=COLOR_FIT,
        linewidth=1.5,
        linestyle="--",
        zorder=1,
        label=f"{fit_kind} fit",
    )
    ax_ce.plot(
        params,
        ce,
        marker="o",
        markersize=8,
        linewidth=2,
        color=COLOR_CE,
        zorder=3,
        label="held-out policy CE",
    )
    for tag, x, y in zip(tags, params, ce):
        ax_ce.annotate(
            f"{tag}\nCE={y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.5,
            color=COLOR_CE,
            fontweight="bold",
        )

    if fit_kind == "power-law":
        fit_note = (
            f"power-law fit: CE = {e_inf:.3f} + {a_coef:.2e}·params$^{{-{alpha:.2f}}}$\n"
            f"estimated irreducible floor E_inf ≈ {e_inf:.3f}"
        )
    else:
        fit_note = (
            f"log-linear fit: CE ≈ {a_coef_lin:.3f} + ({b_coef:.4f})·ln(params)\n"
            f"~{per_doubling:.4f} CE reduction per params-doubling"
        )
    ax_ce.text(
        0.02,
        0.06,
        fit_note,
        transform=ax_ce.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    # Flat value-MSE annotation (not a third panel).
    ax_ce.text(
        0.98,
        0.94,
        f"value MSE flat across ladder: {value_mse.min():.3f}–{value_mse.max():.3f}\n"
        "(no improvement with size)",
        transform=ax_ce.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLOR_VALUE_BAND,
        style="italic",
    )

    ax_ce.set_ylabel("held-out policy cross-entropy (nats)")
    ax_ce.legend(loc="lower left", frameon=True, fontsize=9, bbox_to_anchor=(0.02, 0.18))

    # --- Bottom panel: top-1 move-match vs params ---
    ax_top1.plot(
        params,
        top1,
        marker="s",
        markersize=8,
        linewidth=2,
        color=COLOR_TOP1,
        zorder=3,
        label="held-out top-1 move-match",
    )
    for tag, x, y in zip(tags, params, top1):
        ax_top1.annotate(
            f"{tag}\n{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8.5,
            color=COLOR_TOP1,
            fontweight="bold",
        )

    ax_top1.set_xscale("log")
    ax_top1.set_xlabel("model parameters (log scale)")
    ax_top1.set_ylabel("top-1 move-match (%)")
    ax_top1.set_ylim(top1.min() - 1.0, top1.max() + 1.5)
    ax_top1.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(
        "Policy scaling is monotonic but shallow; value does not scale at all",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"{ratio:.1f}x params (S0→S3) buys only −{ce_drop:.3f} CE / +{top1_gain:.1f}pp top-1 "
        "at fixed 5M-position data;\nvalue MSE stays pinned near 0.71–0.72 regardless of model size",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )

    fig.text(
        0.01,
        0.01,
        "source: reports/scaling_law/results.json — attention-first ladder, 5M positions/rung, "
        "context_window=1, held-out eval on lichess-elite 2025-11",
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
    print(f"fit_kind={fit_kind}")


if __name__ == "__main__":
    main()
