"""Draw the actual Kibitzer architecture (S2, 15.2m params, attention-only).

Output: docs/kibitzer-architecture.png
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kibitzer-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/kibitzer-architecture.png")


def box(ax, xy, wh, text, face, edge="#333333", lw=1.4, fs=10.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight)
    return patch


def arrow(ax, start, end, color="#333333", lw=1.4, text=None, text_xy=None):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, shrinkA=4, shrinkB=4,
    ))
    if text:
        x, y = text_xy if text_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    blue = "#4E79A7"
    pale_blue = "#DCE9F6"
    purple = "#6B5FB5"
    pale_purple = "#E8E8F7"
    orange = "#F28E2B"
    pale_orange = "#FCE8D0"
    red = "#B0413E"
    pale_red = "#F3DADA"
    gray = "#666666"

    ax.text(7, 7.15, "Kibitzer S2 architecture (15.2m params, attention-only)",
            ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(7, 6.82, "d_model=256, 10 trunk layers (all causal attention), 8 heads, Shaw position encoding",
            ha="center", va="center", fontsize=10, color="#555555")

    # input encoding
    box(ax, (0.5, 5.35), (1.8, 0.80), "PGN/FEN\nposition", "#F2F2F2", fs=10)
    box(ax, (0.5, 4.0), (1.8, 0.80), "piece_idx\n64 squares", pale_blue, edge=blue, fs=10)
    box(ax, (0.5, 2.9), (1.8, 0.80), "aux features\nside, clocks, rights", pale_blue, edge=blue, fs=9.5)

    # position encoder
    box(ax, (2.8, 3.8), (2.2, 1.6),
        "PositionEncoder\n3 layers, 8 heads\nShaw relative pos\nencodes 64 squares",
        "#D9EAF7", edge=blue, fs=10, weight="bold")

    # trunk
    box(ax, (5.5, 3.8), (2.4, 1.6),
        "Trunk\n10 blocks\nall CausalAttention\nno SSM (attn_every=1)",
        pale_purple, edge=purple, fs=10, weight="bold")

    # norm
    box(ax, (8.4, 4.15), (1.4, 0.9),
        "RMSNorm\nh in R256", "#F5F5F5", edge=gray, fs=10, weight="bold")

    # policy head
    box(ax, (10.4, 5.3), (2.5, 1.0),
        "Policy head\nLinear(256 -> 4672)\nmove logits",
        pale_orange, edge=orange, fs=9.5, weight="bold")

    # value head (legacy)
    box(ax, (10.4, 3.3), (2.5, 1.0),
        "Value head (legacy)\n256 -> 128 -> 1\n33,025 params",
        pale_red, edge=red, fs=9.5)

    # arrows
    arrow(ax, (2.3, 5.75), (2.8, 4.7), blue)  # fen -> encoder
    arrow(ax, (2.3, 4.4), (2.8, 4.4), blue)   # piece_idx -> encoder
    arrow(ax, (2.3, 3.3), (2.8, 3.9), blue)   # aux -> encoder
    arrow(ax, (5.0, 4.6), (5.5, 4.6), gray)    # encoder -> trunk
    arrow(ax, (7.9, 4.6), (8.4, 4.6), gray)    # trunk -> norm
    arrow(ax, (9.8, 4.6), (10.4, 5.8), orange) # norm -> policy
    arrow(ax, (9.8, 4.6), (10.4, 3.8), red)    # norm -> value

    # note about value head
    ax.text(12.7, 4.8, "main training target", ha="center", fontsize=8.5, color=orange)
    ax.text(12.7, 3.15, "D52 tried 4x bigger (132k),\nregressed in play", ha="center", fontsize=8.5, color=red)

    # stats box
    box(ax, (2.8, 1.0), (8.0, 1.3),
        "Training: 142m lichess elite positions  |  Elo: 2483 +/- 32 (64-sim PUCT vs Stockfish ladder)  |  "
        "Policy CE: 2.329 on held-out  |  Top-1 move match: 30.9%",
        "#F5F5F5", edge=gray, fs=9.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()