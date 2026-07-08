"""Generate three publication-quality figures for the competition-data arm.

Run with: uv run python reports/scaling_law_2/make_figures.py
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# palette (dataviz skill reference palette, light surface)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e2dc"
BASE_COLOR = "#8a8880"     # slate/gray - online-elite base (100M)
COMP_COLOR = "#2a78d6"     # blue - competition-continued (142M)
LOSS_COLOR = "#e34948"     # red - total loss
POLICY_COLOR = "#2a78d6"   # blue - policy loss
VALUE_COLOR = "#1baf7a"    # aqua/green - value loss
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def wilson_ci(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion `score` (0..1) over `n` trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = score
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


def style_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. parse training log
# ---------------------------------------------------------------------------
LOG_PATH = HERE / "comp_train.log"
TOTAL_STEPS = 343750
BATCH = 128
BASE_POSITIONS = 100_000_000

pat = re.compile(
    r"train S2:.*?\|\s*(\d+)/343750.*?loss=([\d.]+), policy=([\d.]+), value=([\d.]+)"
)

steps, losses, policies, values = [], [], [], []
with open(LOG_PATH, "r", errors="ignore") as f:
    text = f.read()

# tqdm writes each update separated by \r (and sometimes \n); split on both
for chunk in re.split(r"[\r\n]", text):
    m = pat.search(chunk)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        policies.append(float(m.group(3)))
        values.append(float(m.group(4)))

steps = np.array(steps)
losses = np.array(losses)
policies = np.array(policies)
values = np.array(values)

# dedupe on step (tqdm re-emits same step with updated postfix); keep last
order = np.argsort(steps, kind="stable")
steps, losses, policies, values = steps[order], losses[order], policies[order], values[order]
_, keep_idx = np.unique(steps, return_index=True)
steps, losses, policies, values = steps[keep_idx], losses[keep_idx], policies[keep_idx], values[keep_idx]

positions_m = (BASE_POSITIONS + steps.astype(np.int64) * BATCH) / 1e6

print(f"parsed {len(steps)} log points, step range {steps.min()}-{steps.max()}")


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


WINDOW = max(50, len(steps) // 300)

# a small fraction of steps (~0.8%) show transient optimizer-instability spikes
# (loss jumping to ~1e6-3e7 for a few consecutive steps before recovering).
# These are real logged values, not a parsing artifact, but they are display
# outliers: we clip the plotted y-range and exclude them from the smoothing
# average (a single spike would otherwise dominate a mean over ~500 points).
OUTLIER_THRESHOLD = 5.0
outlier_frac = float((losses > OUTLIER_THRESHOLD).mean())
print(f"loss spikes >{OUTLIER_THRESHOLD}: {outlier_frac:.3%} of steps (excluded from smoothing, clipped from view)")


def smoothed_xy(x, y, window=WINDOW, clip=OUTLIER_THRESHOLD):
    mask = y <= clip
    x, y = x[mask], y[mask]
    ys = rolling_mean(y, window)
    # x for the 'valid' convolution result: centered
    xs = x[window - 1:]
    xs = xs[: len(ys)]
    return xs, ys


# ---------------------------------------------------------------------------
# 2. load eval / head-to-head JSON data
# ---------------------------------------------------------------------------
comp_arm = json.loads((HERE / "comp_arm.json").read_text())["runs"][0]
h2h_10 = json.loads((HERE / "h2h_vs_base.json").read_text())
h2h_20 = json.loads((HERE / "h2h_vs_base_20g.json").read_text())
sf_base = json.loads((HERE / "sf1900_base_samesetup.json").read_text())
sf_comp_20 = json.loads((HERE / "sf1900_comp_20g.json").read_text())
sf_comp_10 = json.loads((HERE / "sf1900_final.json").read_text())

# combined h2h: 30 games, 11W/14D/5L
h2h_w = h2h_10["w"] + h2h_20["w"]
h2h_d = h2h_10["d"] + h2h_20["d"]
h2h_l = h2h_10["l"] + h2h_20["l"]
h2h_n = h2h_w + h2h_d + h2h_l
h2h_score = (h2h_w + 0.5 * h2h_d) / h2h_n

# combined SF-1900, comp arm: 30 games, 21W/5D/4L
comp_sf_w = sf_comp_20["wins"] + sf_comp_10["wins"]
comp_sf_d = sf_comp_20["draws"] + sf_comp_10["draws"]
comp_sf_l = sf_comp_20["losses"] + sf_comp_10["losses"]
comp_sf_n = comp_sf_w + comp_sf_d + comp_sf_l
comp_sf_score = (comp_sf_w + 0.5 * comp_sf_d) / comp_sf_n

base_sf_n = sf_base["games"]
base_sf_score = sf_base["score"]

print(f"h2h combined: {h2h_w}W/{h2h_d}D/{h2h_l}L = {h2h_score:.3f} (n={h2h_n})")
print(f"comp vs SF-1900 combined: {comp_sf_w}W/{comp_sf_d}D/{comp_sf_l}L = {comp_sf_score:.3f} (n={comp_sf_n})")
print(f"base vs SF-1900: {base_sf_score:.3f} (n={base_sf_n})")

# ===========================================================================
# FIGURE 1: training loss curves during competition-data continuation
# ===========================================================================
fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.6), sharex=True, dpi=300)

panels = [
    (axes[0], losses, "Total loss", LOSS_COLOR, (0.9, 3.0)),
    (axes[1], policies, "Policy loss (CE)", POLICY_COLOR, (0.8, 2.8)),
    (axes[2], values, "Value loss (MSE)", VALUE_COLOR, (0.2, 0.9)),
]

eval_points_m = [120.0, 140.0]

for ax, y, label, color, ylim in panels:
    ax.scatter(positions_m, y, s=2.5, color=color, alpha=0.10, linewidths=0, zorder=1)
    xs, ys = smoothed_xy(positions_m, y)
    ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3, label=f"{label} (rolling mean, w={WINDOW})")
    for xv in eval_points_m:
        ax.axvline(xv, color=TEXT_SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.6, zorder=2)
    ax.set_ylabel(label)
    ax.set_ylim(*ylim)
    style_ax(ax)
    ax.legend(loc="upper right", fontsize=9)

for xv, lbl in zip(eval_points_m, ["SF-1900 eval @120M", "SF-1900 eval @140M"]):
    axes[0].annotate(
        lbl, xy=(xv, panels[0][4][1]), xytext=(4, -4), textcoords="offset points",
        fontsize=8, color=TEXT_SECONDARY, ha="left", va="top", annotation_clip=False,
    )

axes[-1].set_xlabel("Cumulative training positions (millions)")
axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(5))

fig.suptitle(
    "Competition-data continuation: training loss vs. cumulative positions",
    fontsize=13, fontweight="bold", y=0.995,
)
subtitle = (
    "S2-shaw (15.2M params) continued from the 100M online-elite checkpoint on TWIC elite-competition "
    "games (100M → 142M positions). Light dots = raw per-step values; bold line = rolling mean, excluding "
    f"the {outlier_frac:.1%} of steps with transient optimizer-instability spikes (y-axis clipped for readability). "
    "Dashed verticals mark the two in-loop SF-1900 evaluation points."
)
fig.text(
    0.5, 0.965,
    "\n".join(textwrap.wrap(subtitle, width=100)),
    fontsize=8.8, color=TEXT_SECONDARY, ha="center", va="top",
)

fig.tight_layout(rect=(0, 0, 1, 0.90))
out1 = HERE / "fig_comp_loss.png"
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 2: strength comparison — SF-1900 (same harness) + head-to-head
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.0), dpi=300, gridspec_kw={"width_ratios": [1.1, 0.9]})

# --- panel A: vs SF-1900 @ 64 sims, same harness ---
labels_a = ["Online base\n(100M)", "Competition\n(142M)"]
scores_a = [base_sf_score, comp_sf_score]
ns_a = [base_sf_n, comp_sf_n]
colors_a = [BASE_COLOR, COMP_COLOR]
x_a = np.arange(2)

bars = ax1.bar(x_a, scores_a, width=0.5, color=colors_a, zorder=3, edgecolor="none")
for xi, s, n, c in zip(x_a, scores_a, ns_a, colors_a):
    lo, hi = wilson_ci(s, n)
    ax1.errorbar(xi, s, yerr=[[s - lo], [hi - s]], fmt="none", ecolor=TEXT_PRIMARY,
                 elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
    ax1.annotate(f"{s:.3f}\n(n={n})", xy=(xi, hi), xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold")

ax1.set_xticks(x_a)
ax1.set_xticklabels(labels_a)
ax1.set_ylabel("Score vs. Stockfish-1900 @ 64 sims\n(win=1, draw=0.5, loss=0)")
ax1.set_ylim(0, 1.0)
ax1.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
ax1.set_title("vs. SF-1900, identical harness", fontsize=11, loc="left")
style_ax(ax1)

# --- panel B: head-to-head, competition vs base ---
x_b = [0]
lo_h, hi_h = wilson_ci(h2h_score, h2h_n)
ax2.bar(x_b, [h2h_score], width=0.5, color=COMP_COLOR, zorder=3, edgecolor="none")
ax2.errorbar(x_b, [h2h_score], yerr=[[h2h_score - lo_h], [hi_h - h2h_score]], fmt="none",
             ecolor=TEXT_PRIMARY, elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
ax2.annotate(
    f"{h2h_score:.3f}\n({h2h_w}W-{h2h_d}D-{h2h_l}L, n={h2h_n})",
    xy=(0, hi_h), xytext=(0, 6), textcoords="offset points",
    ha="center", fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold",
)
ax2.axhline(0.5, color=TEXT_SECONDARY, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
ax2.text(0.5, 0.5, "parity", transform=ax2.transAxes, fontsize=8.5, color=TEXT_SECONDARY,
         ha="left", va="bottom", style="italic")
ax2.set_xticks(x_b)
ax2.set_xticklabels(["Competition (142M)\nvs. Online base (100M)"])
ax2.set_xlim(-0.75, 0.75)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Head-to-head score (PUCT @ 128 sims)")
ax2.set_title("Direct head-to-head", fontsize=11, loc="left")
style_ax(ax2)

fig.suptitle(
    "Competition data HELD: neither gain nor loss vs. the online-elite base",
    fontsize=13.5, fontweight="bold", y=1.01,
)
fig.text(
    0.5, 0.955,
    "Wilson 95% CIs shown. SF-1900 scores overlap (0.775 vs. 0.783); the head-to-head leans above parity (0.600)\n"
    "but its interval still crosses 0.5 at n=30 — not a statistically distinguishable improvement.",
    fontsize=9, color=TEXT_SECONDARY, ha="center", va="top",
)

fig.tight_layout(rect=(0, 0, 1, 0.90))
out2 = HERE / "fig_comp_strength.png"
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# FIGURE 3: in-loop dip-and-recover during continuation
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)

inloop_x = [p["positions"] / 1e6 for p in comp_arm["play_eval_history"]]
inloop_y = [p["score"] for p in comp_arm["play_eval_history"]]
# convert relative continuation positions to cumulative (100M + delta)
inloop_x_cum = [BASE_POSITIONS / 1e6 + x for x in inloop_x]

base_band_lo, base_band_hi = 0.775, 0.825
ax.axhspan(base_band_lo, base_band_hi, color=BASE_COLOR, alpha=0.15, zorder=1,
           label=f"Online-base reference level ({base_band_lo:.3f}–{base_band_hi:.3f}, illustrative)")
ax.axhline(base_sf_score, color=BASE_COLOR, linewidth=1.3, linestyle=(0, (5, 3)), zorder=2)

ax.plot(inloop_x_cum, inloop_y, color=COMP_COLOR, linewidth=2.2, marker="o", markersize=8,
        zorder=4, label="In-loop SF-1900 score during continuation")
for xv, yv in zip(inloop_x_cum, inloop_y):
    ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold")

# head-to-head markers: mid-training (illustrative, ~130M, small-sample) vs finished (140-142M)
h2h_mid_x, h2h_mid_y = 130.0, 0.300
h2h_final_x, h2h_final_y = 142.0, h2h_score
ax.scatter([h2h_mid_x], [h2h_mid_y], marker="D", s=70, color=CRITICAL, zorder=5,
           label="Head-to-head vs. base, mid-training (small n, illustrative)")
ax.scatter([h2h_final_x], [h2h_final_y], marker="D", s=70, color=GOOD, zorder=5,
           label=f"Head-to-head vs. base, finished (n={h2h_n})")
ax.annotate(f"{h2h_mid_y:.3f}", xy=(h2h_mid_x, h2h_mid_y), xytext=(6, -14), textcoords="offset points",
            fontsize=9, color=CRITICAL)
ax.annotate(f"{h2h_final_y:.3f}", xy=(h2h_final_x, h2h_final_y), xytext=(-40, 10), textcoords="offset points",
            fontsize=9, color=GOOD)

ax.set_xlabel("Cumulative training positions (millions)")
ax.set_ylabel("Score (win=1, draw=0.5, loss=0)")
ax.set_ylim(0.15, 1.0)
ax.set_xlim(105, 148)
style_ax(ax)
ax.legend(loc="upper left", fontsize=8.3)

fig.suptitle("Transient dip and recovery under distribution shift", fontsize=13.5, fontweight="bold", y=0.99)
fig.text(
    0.5, 0.925,
    "In-loop SF-1900 eval dips at 120M then recovers by 140M; head-to-head vs. the base shows the same pattern.\n"
    "Small-sample / illustrative: mid-training head-to-head point and the base reference band are not independently re-measured at each x.",
    fontsize=9, color=TEXT_SECONDARY, ha="center", va="top",
)

fig.tight_layout(rect=(0, 0, 1, 0.88))
out3 = HERE / "fig_comp_dip_recover.png"
fig.savefig(out3, dpi=300, bbox_inches="tight")
plt.close(fig)

print("wrote:", out1, out2, out3)
