"""Draw the Kibitzer architecture with the scaled value-head variant.

Output: reports/value_head/fig_valuehead_architecture.png
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("reports/value_head/fig_valuehead_architecture.png")


def box(ax, xy, wh, text, face, edge="#333333", lw=1.4, fs=10.5, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight)
    return patch


def arrow(ax, start, end, color="#333333", lw=1.4, text=None, text_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )
    if text:
        x, y = text_xy if text_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    blue = "#4E79A7"
    pale_blue = "#DCE9F6"
    teal = "#0F9588"
    pale_teal = "#D8F0EC"
    orange = "#F28E2B"
    pale_orange = "#FCE8D0"
    red = "#B0413E"
    pale_red = "#F3DADA"
    gray = "#666666"
    pale_gray = "#F2F2F2"

    ax.text(
        7,
        7.55,
        "Kibitzer architecture with scaled value head",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        7,
        7.18,
        "D52 changed only the value head: trunk, encoder, and policy head stayed frozen during value-head retraining.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )

    box(ax, (0.55, 5.35), (2.0, 0.85), "PGN/FEN\nposition", pale_gray, fs=10)
    box(ax, (0.55, 3.95), (2.0, 0.85), "piece_idx\n64 squares", pale_blue, edge=blue, fs=10)
    box(ax, (0.55, 2.85), (2.0, 0.85), "aux features\nside, clocks, rights", pale_blue, edge=blue, fs=9.5)

    box(
        ax,
        (3.05, 3.5),
        (2.35, 1.55),
        "PositionEncoder\n3 layers, 8 heads\nShaw relative position",
        "#D9EAF7",
        edge=blue,
        fs=10,
        weight="bold",
    )
    box(
        ax,
        (5.95, 3.35),
        (2.35, 1.85),
        "Hybrid trunk\n10 blocks\nAttention every 3\nSSM otherwise",
        "#E8E8F7",
        edge="#6B5FB5",
        fs=10,
        weight="bold",
    )
    box(ax, (8.85, 3.75), (1.6, 1.05), "RMSNorm\nh ∈ R320", "#F5F5F5", edge=gray, fs=10, weight="bold")

    box(
        ax,
        (11.0, 5.15),
        (2.35, 0.95),
        "Policy head\nLinear(320 → 4672)\nmove logits",
        pale_orange,
        edge=orange,
        fs=9.5,
        weight="bold",
    )
    box(
        ax,
        (11.0, 3.35),
        (2.35, 0.95),
        "Legacy value head\n320 → 160 → 1\n33,025 params",
        pale_red,
        edge=red,
        fs=9.5,
    )
    box(
        ax,
        (11.0, 1.65),
        (2.35, 1.05),
        "Scaled value head\n320 → 256 → 256 → 1\n131,841 params",
        pale_teal,
        edge=teal,
        fs=9.5,
        weight="bold",
    )

    arrow(ax, (2.55, 4.38), (3.05, 4.28), blue)
    arrow(ax, (2.55, 3.28), (3.05, 4.05), blue)
    arrow(ax, (5.4, 4.28), (5.95, 4.28), gray)
    arrow(ax, (8.3, 4.28), (8.85, 4.28), gray)
    arrow(ax, (10.45, 4.28), (11.0, 5.62), orange)
    arrow(ax, (10.45, 4.28), (11.0, 3.82), red)
    arrow(ax, (10.45, 4.28), (11.0, 2.18), teal)

    ax.text(12.18, 4.82, "main training target", ha="center", fontsize=8.5, color=orange)
    ax.text(12.18, 3.05, "default / old checkpoints", ha="center", fontsize=8.5, color=red)
    ax.text(12.18, 1.30, "D52 experiment", ha="center", fontsize=8.5, color=teal, fontweight="bold")

    box(
        ax,
        (3.1, 0.65),
        (7.3, 1.35),
        "D52 training protocol\nfreeze encoder + trunk + policy head; replace only value_head; train against Stockfish depth-14 labels\n"
        "offline MSE improved: 100M base 0.0403→0.0196, 142M comp 0.0569→0.0178\n"
        "play regressed: SF-1900 and Leela tests moved down, so scaled value head is a negative strength lever",
        "#FFF7E8",
        edge="#C48A2C",
        fs=9.2,
    )
    arrow(ax, (12.15, 1.65), (10.4, 1.32), teal, lw=1.1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
