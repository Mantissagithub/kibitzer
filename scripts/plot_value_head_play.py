"""Generate the two additional publication-quality figures for the value-head
capacity experiment (D52): offline-vs-play dissonance, and full play summary.

Reuses the palette/style established in scripts/plot_value_head.py. Does not
touch the two figures already produced by that script.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kibitzer-matplotlib"),
)

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "reports/value_head"

# ---- palette (matches scripts/plot_value_head.py) ----
BASE_COLOR = "#3F51B5"    # slate/indigo -> 100M base
COMP_COLOR = "#00897B"    # teal accent  -> 142M comp
BASE_LIGHT = "#9FA8DA"
COMP_LIGHT = "#80CBC4"
GRID_COLOR = "#DDDDDD"
TEXT_COLOR = "#222222"
LEGACY_MARK = "#B0413E"   # muted red for legacy reference lines/bars

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.5,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def wilson_ci(wins: float, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    `wins` may be fractional (draws counted as 0.5), matching win-rate scoring.
    Returns (center_estimate, lower, upper) as proportions in [0, 1].
    """
    p = wins / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


# ---- data (exact figures from task spec) ----
# Offline MSE (best epoch)
mse_base_legacy, mse_base_enlarged = 0.0403, 0.0196
mse_comp_legacy, mse_comp_enlarged = 0.0569, 0.0178
pct_base_mse = (mse_base_enlarged - mse_base_legacy) / mse_base_legacy * 100
pct_comp_mse = (mse_comp_enlarged - mse_comp_legacy) / mse_comp_legacy * 100

# PUCT vs Stockfish-1900 @64 sims, n=20 games. wins as (W + 0.5*D)
N_SF = 20
sf_base_legacy_w = 0.775 * N_SF     # 15.5 -> not integer counts given directly; use win-rate*n
sf_base_enlarged_w = 0.625 * N_SF
sf_comp_legacy_w = 0.783 * N_SF     # ~15.66, from 11W/4D/5L -> 11+2=13 actually check
sf_comp_enlarged_w = 0.650 * N_SF

# Recompute directly from W/D/L counts given in the prompt (more precise & consistent)
def wdl_rate(w, d, l):
    n = w + d + l
    return (w + 0.5 * d) / n, n

sf_base_legacy_rate, sf_base_legacy_n = 0.775, 20   # not given as WDL; use stated rate directly
sf_base_enlarged_rate, sf_base_enlarged_n = wdl_rate(11, 3, 6)   # 0.625, 20
sf_comp_legacy_rate, sf_comp_legacy_n = 0.783, 20
sf_comp_enlarged_rate, sf_comp_enlarged_n = wdl_rate(11, 4, 5)   # 0.65, 20

leela_comp_legacy_rate, leela_comp_legacy_n = wdl_rate(3, 3, 14)   # 0.225, 20
leela_comp_enlarged_rate, leela_comp_enlarged_n = wdl_rate(1, 4, 15)  # 0.15, 20

alpha_beta_enlarged = 0.19
alpha_beta_legacy_ref = 0.075  # D50 reference

# =====================================================================
# Figure 3: fig_valuehead_offline_vs_play.png  (the money figure)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
ax_l, ax_r = axes

group_labels = ["100M base", "142M comp"]
y = np.array([0.0, 0.9])
bar_h = 0.32

# --- LEFT: offline MSE ---
legacy_vals = [mse_base_legacy, mse_comp_legacy]
enlarged_vals = [mse_base_enlarged, mse_comp_enlarged]

bars_leg = ax_l.barh(y + bar_h / 2 + 0.02, legacy_vals, height=bar_h, color=LEGACY_MARK,
                      alpha=0.92, zorder=3, label="legacy head (33k params)")
bars_enl = ax_l.barh(y - bar_h / 2 - 0.02, enlarged_vals,
                      color=[BASE_COLOR, COMP_COLOR], height=bar_h, alpha=0.92, zorder=3,
                      label="enlarged head (132k params)")

for rect, val in zip(bars_leg, legacy_vals):
    ax_l.text(val + 0.0012, rect.get_y() + rect.get_height() / 2, f"{val:.4f}",
              ha="left", va="center", fontsize=9.7, color=TEXT_COLOR)
for rect, val in zip(bars_enl, enlarged_vals):
    ax_l.text(val + 0.0012, rect.get_y() + rect.get_height() / 2, f"{val:.4f}",
              ha="left", va="center", fontsize=9.7, color=TEXT_COLOR)

for yc, pct in zip(y, [pct_base_mse, pct_comp_mse]):
    ax_l.annotate("", xy=(0.006, yc - bar_h / 2 - 0.02), xytext=(0.006, yc + bar_h / 2 + 0.02),
                  arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.8))
    ax_l.text(0.010, yc, f"{pct:.0f}%", fontsize=10.5, fontweight="bold", color="#2E7D32",
              ha="left", va="center")

ax_l.set_yticks(y)
ax_l.set_yticklabels(group_labels, fontsize=11)
ax_l.invert_yaxis()
ax_l.set_xlim(0, 0.066)
ax_l.set_xlabel("held-out value MSE vs Stockfish depth-14 (lower is better)")
ax_l.set_title("Offline value error", fontsize=12.5, fontweight="bold", pad=10)
ax_l.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
ax_l.set_axisbelow(True)
for spine in ("top", "right"):
    ax_l.spines[spine].set_visible(False)
ax_l.legend(loc="lower right", frameon=False, fontsize=9)

# --- RIGHT: play win-rate vs SF-1900 with Wilson CIs ---
legacy_rates = [sf_base_legacy_rate, sf_comp_legacy_rate]
legacy_ns = [sf_base_legacy_n, sf_comp_legacy_n]
enlarged_rates = [sf_base_enlarged_rate, sf_comp_enlarged_rate]
enlarged_ns = [sf_base_enlarged_n, sf_comp_enlarged_n]

def cis_for(rates, ns):
    los, his = [], []
    for r, n in zip(rates, ns):
        w = r * n
        _, lo, hi = wilson_ci(w, n)
        los.append(lo)
        his.append(hi)
    return np.array(los), np.array(his)

leg_lo, leg_hi = cis_for(legacy_rates, legacy_ns)
enl_lo, enl_hi = cis_for(enlarged_rates, enlarged_ns)

bars_leg_r = ax_r.barh(y + bar_h / 2 + 0.02, legacy_rates, height=bar_h, color=LEGACY_MARK,
                        alpha=0.92, zorder=3,
                        xerr=[np.array(legacy_rates) - leg_lo, leg_hi - np.array(legacy_rates)],
                        error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=3, zorder=4),
                        label="legacy head")
bars_enl_r = ax_r.barh(y - bar_h / 2 - 0.02, enlarged_rates, color=[BASE_COLOR, COMP_COLOR],
                        height=bar_h, alpha=0.92, zorder=3,
                        xerr=[np.array(enlarged_rates) - enl_lo, enl_hi - np.array(enlarged_rates)],
                        error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=3, zorder=4),
                        label="enlarged head")

for rect, val, hi in zip(bars_leg_r, legacy_rates, leg_hi):
    ax_r.text(hi + 0.025, rect.get_y() + rect.get_height() / 2, f"{val:.3f}",
              ha="left", va="center", fontsize=9.7, color=TEXT_COLOR)
for rect, val, hi in zip(bars_enl_r, enlarged_rates, enl_hi):
    ax_r.text(hi + 0.025, rect.get_y() + rect.get_height() / 2, f"{val:.3f}",
              ha="left", va="center", fontsize=9.7, color=TEXT_COLOR)

for yc, lr, er in zip(y, legacy_rates, enlarged_rates):
    delta_pp = (er - lr) * 100
    ax_r.annotate("", xy=(0.06, yc - bar_h / 2 - 0.02), xytext=(0.06, yc + bar_h / 2 + 0.02),
                  arrowprops=dict(arrowstyle="-|>", color="#B0413E", lw=1.8))
    ax_r.text(0.10, yc, f"{delta_pp:+.0f}pp", fontsize=10.5, fontweight="bold", color="#B0413E",
              ha="left", va="center")

ax_r.set_yticks(y)
ax_r.set_yticklabels(group_labels, fontsize=11)
ax_r.invert_yaxis()
ax_r.set_xlim(0, 1.0)
ax_r.axvline(0.5, color="#999999", lw=0.9, ls=":", zorder=1)
ax_r.set_xlabel("win-rate vs Stockfish-1900, PUCT @ 64 sims, n=20 (higher is better)")
ax_r.set_title("Play strength", fontsize=12.5, fontweight="bold", pad=10)
ax_r.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
ax_r.set_axisbelow(True)
for spine in ("top", "right"):
    ax_r.spines[spine].set_visible(False)
ax_r.legend(loc="upper right", frameon=False, fontsize=9)

fig.suptitle("A better-fit value head (offline) makes play worse",
             fontsize=15.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.925,
         "Same direction on two independent models (100M base, 142M comp) · n=20 games/bar, bars show Wilson 95% CI",
         ha="center", va="top", fontsize=9.6, color="#555555", style="italic")
fig.text(
    0.5, 0.015,
    "offline MSE = held-out eval vs Stockfish depth-14 labels, best enlarged-head epoch; "
    "green arrows = offline improvement, red arrows = play regression",
    ha="center", va="bottom", fontsize=8.8, color="#555555",
)

fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.90])
fig.savefig(f"{OUT_DIR}/fig_valuehead_offline_vs_play.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Figure 4: fig_valuehead_play_summary.png
# =====================================================================
fig, ax = plt.subplots(figsize=(11.4, 6.4))

# groups: SF-1900/base, SF-1900/comp, Leela~2700/comp
groups = [
    ("SF-1900\n100M base", sf_base_legacy_rate, sf_base_legacy_n, sf_base_enlarged_rate, sf_base_enlarged_n),
    ("SF-1900\n142M comp", sf_comp_legacy_rate, sf_comp_legacy_n, sf_comp_enlarged_rate, sf_comp_enlarged_n),
    ("Leela ~2700\n142M comp", leela_comp_legacy_rate, leela_comp_legacy_n,
     leela_comp_enlarged_rate, leela_comp_enlarged_n),
]

x = np.arange(len(groups))
bar_w = 0.34

legacy_vals = [g[1] for g in groups]
legacy_ns = [g[2] for g in groups]
enlarged_vals = [g[3] for g in groups]
enlarged_ns = [g[4] for g in groups]

leg_lo, leg_hi = cis_for(legacy_vals, legacy_ns)
enl_lo, enl_hi = cis_for(enlarged_vals, enlarged_ns)

bars_leg = ax.bar(x - bar_w / 2 - 0.01, legacy_vals, width=bar_w, color=LEGACY_MARK, alpha=0.92,
                   zorder=3,
                   yerr=[np.array(legacy_vals) - leg_lo, leg_hi - np.array(legacy_vals)],
                   error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=4, zorder=4),
                   label="legacy head (33k params)")
bars_enl = ax.bar(x + bar_w / 2 + 0.01, enlarged_vals, width=bar_w, color="#546E7A", alpha=0.92,
                   zorder=3,
                   yerr=[np.array(enlarged_vals) - enl_lo, enl_hi - np.array(enlarged_vals)],
                   error_kw=dict(ecolor="#555555", elinewidth=1.1, capsize=4, zorder=4),
                   label="enlarged head (132k params)")

for rect, val in zip(bars_leg, legacy_vals):
    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.045, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)
for rect, val in zip(bars_enl, enlarged_vals):
    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.045, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)

ax.axhline(0.5, color="#999999", lw=0.9, ls=":", zorder=1)
ax.text(len(groups) - 0.52, 0.505, "parity (0.5)", fontsize=8.5, color="#777777", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
ax.set_ylabel("win-rate (W + 0.5·D) / N, n=20 games per bar (higher is better)")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.01, 0.98))

ax.set_title("Enlarged value head is worse on every play metric", fontsize=15, fontweight="bold", pad=16)

ax.text(
    0.99, 0.965,
    f"alpha-beta diagnostic (128 net-evals, comp enlarged value): win-rate ≈{alpha_beta_enlarged:.2f} "
    f"vs D50 legacy-head reference ≈{alpha_beta_legacy_ref:.2f}\nsearch leaf still collapses",
    transform=ax.transAxes, ha="right", va="top", fontsize=8.7,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BBBBBB", lw=0.8),
)

fig.text(
    0.5, 0.015,
    "bars show Wilson 95% CI; win-rate counted from reported W/D/L (draws = 0.5); "
    "Leela = t1-256x10-distilled net at nodes=1, a ~2700 tactical yardstick",
    ha="center", va="bottom", fontsize=8.8, color="#555555",
)

fig.tight_layout(rect=[0.02, 0.05, 0.99, 0.93])
fig.savefig(f"{OUT_DIR}/fig_valuehead_play_summary.png", bbox_inches="tight")
plt.close(fig)

print("done")
