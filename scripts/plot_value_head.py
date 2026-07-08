"""Generate publication-quality figures for the value-head capacity experiment.

Reads hardcoded eval metrics (transcribed from reports/value_head/value_big_*_train.log)
and writes two PNGs to reports/value_head/.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "kibitzer-matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "reports/value_head"

# ---- palette ----
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

# ---- data ----
legacy_base_mse = 0.0403
legacy_comp_mse = 0.0569

base_epochs = np.arange(1, 9)
base_eval_mse = np.array([0.0206, 0.0196, 0.0202, 0.0202, 0.0209, 0.0213, 0.0214, 0.0223])
base_train_mse = np.array([0.0231, 0.0197, 0.0185, 0.0175, 0.0165, 0.0154, 0.0143, 0.0132])

comp_eval_mse = np.array([0.0187, 0.0178, 0.0181, 0.0183, 0.0188, 0.0192, 0.0193, 0.0197])
comp_train_mse = np.array([0.0210, 0.0180, 0.0169, 0.0160, 0.0150, 0.0140, 0.0129, 0.0119])

best_base_mse = base_eval_mse.min()
best_comp_mse = comp_eval_mse.min()
best_base_epoch = base_epochs[np.argmin(base_eval_mse)]
best_comp_epoch = comp_epochs = base_epochs[np.argmin(comp_eval_mse)]

pct_base = (best_base_mse - legacy_base_mse) / legacy_base_mse * 100
pct_comp = (best_comp_mse - legacy_comp_mse) / legacy_comp_mse * 100

# =====================================================================
# Figure 1: before / after horizontal bars
# =====================================================================
fig, ax = plt.subplots(figsize=(10.8, 5.9))

labels = [
    "100M base · legacy",
    "100M base · enlarged",
    "142M comp · legacy",
    "142M comp · enlarged",
]
values = [legacy_base_mse, best_base_mse, legacy_comp_mse, best_comp_mse]
colors = [LEGACY_MARK, BASE_COLOR, LEGACY_MARK, COMP_COLOR]
y = np.array([0.0, 0.42, 1.25, 1.67])

bars = ax.barh(y, values, height=0.30, color=colors, alpha=0.92, zorder=3)
for rect, val in zip(bars, values):
    ax.text(val + 0.0011, rect.get_y() + rect.get_height() / 2, f"{val:.4f}",
            ha="left", va="center", fontsize=10.5, color=TEXT_COLOR)

ax.text(0.045, 0.21, f"{pct_base:.0f}% MSE", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color="#333333")
ax.text(0.045, 1.46, f"{pct_comp:.0f}% MSE", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color="#333333")

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("held-out value MSE vs Stockfish depth-14 labels (lower is better)")
ax.set_xlim(0.0, 0.064)
ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=LEGACY_MARK, alpha=0.92),
    plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR, alpha=0.92),
    plt.Rectangle((0, 0), 1, 1, color=COMP_COLOR, alpha=0.92),
]
ax.legend(
    legend_handles,
    ["legacy head", "enlarged head on 100M base", "enlarged head on 142M comp"],
    loc="lower right",
    frameon=False,
    fontsize=9.5,
)
ax.set_title("Value-head capacity: offline value error drops, play impact untested",
             fontsize=14, fontweight="bold", pad=14)
fig.text(
    0.5,
    0.02,
    "trunk/policy frozen; 250k Stockfish-label cache; 25,010-position game-disjoint eval; best enlarged-head epoch shown",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
)

fig.tight_layout(rect=[0.03, 0.07, 0.99, 0.94])
fig.savefig(f"{OUT_DIR}/fig_valuehead_beforeafter.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Figure 2: eval MSE vs epoch, both models, legacy reference lines
# =====================================================================
fig, ax = plt.subplots(figsize=(10.6, 5.8))

ax.plot(base_epochs, base_eval_mse, marker="o", color=BASE_COLOR, lw=2.0, ms=5.5,
        label="100M base — eval MSE (enlarged head)", zorder=4)
ax.plot(base_epochs, comp_eval_mse, marker="o", color=COMP_COLOR, lw=2.0, ms=5.5,
        label="142M comp — eval MSE (enlarged head)", zorder=4)

ax.plot(base_epochs, base_train_mse, marker="", color=BASE_LIGHT, lw=1.4, ls=(0, (1, 1)),
        label="100M base — train MSE", zorder=2)
ax.plot(base_epochs, comp_train_mse, marker="", color=COMP_LIGHT, lw=1.4, ls=(0, (1, 1)),
        label="142M comp — train MSE", zorder=2)

ax.axhline(legacy_base_mse, color=BASE_COLOR, lw=1.2, ls="--", alpha=0.6, zorder=1)
ax.axhline(legacy_comp_mse, color=COMP_COLOR, lw=1.2, ls="--", alpha=0.6, zorder=1)
ax.text(8.05, legacy_base_mse, "legacy head\n(100M base)", fontsize=8, color=BASE_COLOR, va="center")
ax.text(8.05, legacy_comp_mse, "legacy head\n(142M comp)", fontsize=8, color=COMP_COLOR, va="center")

# mark best epoch for each
ax.scatter([best_base_epoch], [best_base_mse], s=90, facecolors="none", edgecolors=BASE_COLOR, lw=1.8, zorder=5)
ax.scatter([best_comp_epoch], [best_comp_mse], s=90, facecolors="none", edgecolors=COMP_COLOR, lw=1.8, zorder=5)
ax.annotate("best eval epoch = 2", xy=(best_base_epoch, best_base_mse), xytext=(3.2, 0.0144),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))

ax.set_xlabel("Epoch")
ax.set_ylabel("Value MSE vs. Stockfish depth-14 labels")
ax.set_xlim(0.7, 9.6)
ax.set_xticks(base_epochs)
ax.set_ylim(0.010, 0.062)
ax.grid(color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

ax.set_title("Enlarged value head overfits after ~2 epochs",
              fontsize=14, fontweight="bold", pad=14)
fig.text(0.5, 0.93,
          "",
          ha="center", va="top", fontsize=8.6, color="#555555", style="italic")
fig.text(
    0.5,
    0.02,
    "eval MSE bottoms at epoch 2 while train MSE keeps falling; dashed lines are legacy-head eval MSE. offline metric only.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
)

fig.tight_layout(rect=[0.03, 0.07, 0.99, 0.94])
fig.savefig(f"{OUT_DIR}/fig_valuehead_epochs.png", bbox_inches="tight")
plt.close(fig)

print("done")
