# figures for the "params vs strength" blog section. hardcoded table of kibitzer plus
# public reference points (deepmind searchless transformer paper, alphazero, maia, chess-gpt,
# lc0, stockfish, gpt-3.5). two figures: a log-params scatter (frontier framing) and a sorted
# horizontal bar chart. dark theme matches plot_failures.py.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

INK = "#e8e6e3"
BG = "#14161a"
PANEL = "#1b1e24"
ACC = "#e0533d"       # kibitzer accent (warm)
ACC2 = "#4d9de0"       # cool blue, no-search class
ACC3 = "#c9a227"       # gold, search class
TERTIARY = "#7d8290"   # footnote / muted text
GRID = "#2c313a"

OUT_DIR = Path("reports/model_landscape")

# id, label, params_m (None = undisclosed), elo, search (bool), open (bool), note
ROWS = [
    ("kib512", "Kibitzer @512-sim PUCT", 15.2, 2581, True, True, "ours, SF-anchored Ordo, 171 games"),
    ("kib64", "Kibitzer @64-sim", 15.2, 2483, True, True, "ours, SF ladder fit"),
    ("dm9", "DeepMind searchless 9M", 9, 2007, False, True, "tournament elo, arXiv:2402.04494"),
    ("dm136", "DeepMind searchless 136M", 136, 2224, False, True, "same paper"),
    ("dm270", "DeepMind searchless 270M", 270, 2299, False, True, "same paper (2895 blitz vs humans)"),
    ("azpol", "AlphaZero policy-only", 24, 1620, False, False, "params est., same paper's table"),
    ("azmcts", "AlphaZero + 400 MCTS", 24, 2502, True, False, "same paper's table"),
    ("maia", "Maia-1900", 0.5, 1900, False, True, "human-matching target, params est."),
    ("cgpt", "chess-GPT 50M (Karvonen)", 50, 1500, False, True, "approx, white/black asymmetric"),
    ("lc0t1", "lc0 t1-256x10 @1 node", 10, 2700, False, True, "params est., our calibration"),
    ("lc0bt4", "lc0 BT4 + full search", 191, 3500, True, True, "approx CCRL"),
    ("sf17", "Stockfish 17 (NNUE + alpha-beta)", 30, 3653, True, True, "net params est., CCRL 40/15"),
    ("sf16fast", "Stockfish 16 @50ms", 30, 2706, True, True, "same paper's table"),
    ("gpt35", "GPT-3.5-turbo-instruct", None, 1755, False, False, "size undisclosed, lichess elo"),
]

FOOTNOTE = ("elo figures as reported by each project (internal tournament, CCRL, lichess, "
            "SF-anchored Ordo), scales are not strictly comparable. params for engines = network only.")


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
    })


def by_id(rid):
    for r in ROWS:
        if r[0] == rid:
            return r
    raise KeyError(rid)


def fig_scatter(out):
    fig, ax = plt.subplots(figsize=(10.5, 7))

    scatter_rows = [r for r in ROWS if r[2] is not None]  # drop gpt35, no size

    # manual label offsets (dx points, dy points) to dodge overlaps
    offsets = {
        "kib512": (10, 8), "kib64": (10, -14), "dm9": (10, 10), "dm136": (10, 8),
        "dm270": (-10, 8), "azpol": (10, -14), "azmcts": (10, 8), "maia": (10, 8),
        "cgpt": (10, -14), "lc0t1": (12, -4), "lc0bt4": (10, 8), "sf17": (10, 8),
        "sf16fast": (10, -14), "azmcts": (10, -16),
    }

    for rid, label, params, elo, search, is_open, note in scatter_rows:
        if rid in ("kib512", "kib64"):
            continue  # drawn separately below
        color = ACC3 if search else ACC2
        marker = "D" if search else "o"
        ax.scatter(params, elo, s=70, marker=marker, color=color,
                   edgecolors=BG, linewidths=0.6, zorder=3)
        dx, dy = offsets.get(rid, (10, 8))
        ax.annotate(rid, (params, elo), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, ha="left" if dx > 0 else "right")

    # kibitzer points, stars, accent color, larger
    kib512 = by_id("kib512")
    kib64 = by_id("kib64")
    for rid, r in (("kib512", kib512), ("kib64", kib64)):
        ax.scatter(r[2], r[3], s=280, marker="*", color=ACC,
                   edgecolors=BG, linewidths=0.8, zorder=5)
        dx, dy = offsets.get(rid, (10, 8))
        ax.annotate(rid, (r[2], r[3]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=ACC, fontweight="bold", ha="left" if dx > 0 else "right")

    # arrow from kib64 to kib512, same weights deeper search
    arrow = FancyArrowPatch((kib64[2], kib64[3] + 25), (kib512[2], kib512[3] - 25),
                            arrowstyle="-|>", mutation_scale=14, color=ACC,
                            linewidth=1.4, alpha=0.7, zorder=4,
                            connectionstyle="arc3,rad=0.15")
    ax.add_patch(arrow)
    # annotation text in the empty region left of the stars, leader arrow to kib512
    ax.annotate("same weights,\ndeeper search (+~100 Elo)",
                xy=(kib512[2], kib512[3]), xytext=(3.2, 2600),
                fontsize=8.5, color=ACC, style="italic", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=ACC, alpha=0.5, linewidth=0.9,
                                shrinkA=14, shrinkB=10))

    # no-search pareto frontier: maia -> dm9 -> lc0t1 (by params, non-dominated elo)
    frontier_ids = ["maia", "dm9", "lc0t1"]
    fx = [by_id(i)[2] for i in frontier_ids]
    fy = [by_id(i)[3] for i in frontier_ids]
    ax.plot(fx, fy, linestyle="--", color=ACC2, alpha=0.5, linewidth=1.3, zorder=2,
            label="no-search frontier")

    ax.set_xscale("log")
    ax.set_xlim(0.28, 550)  # margin so edge labels (maia, dm270) do not clip
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("reported Elo")
    ax.set_title("small model, honest number: the params-vs-strength landscape")
    ax.grid(alpha=0.3, which="both")

    # legend: marker classes + frontier
    handles = [
        plt.Line2D([], [], marker="o", color=ACC2, linestyle="none", markersize=8, label="no search at play time"),
        plt.Line2D([], [], marker="D", color=ACC3, linestyle="none", markersize=8, label="plays with search"),
        plt.Line2D([], [], marker="*", color=ACC, linestyle="none", markersize=16, label="kibitzer"),
        plt.Line2D([], [], color=ACC2, linestyle="--", alpha=0.5, label="no-search frontier"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    fig.text(0.01, 0.005, FOOTNOTE, fontsize=7.5, color=TERTIARY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def fig_bars(out):
    rows = sorted(ROWS, key=lambda r: r[3])

    fig, ax = plt.subplots(figsize=(11, 7.5))
    y = range(len(rows))

    for i, (rid, label, params, elo, search, is_open, note) in enumerate(rows):
        if rid in ("kib512", "kib64"):
            color = ACC
        else:
            color = ACC3 if search else ACC2
        hatch = "///" if not is_open else None
        facecolor = "none" if not is_open else color
        edgecolor = color
        ax.barh(i, elo, height=0.62, facecolor=facecolor, edgecolor=edgecolor,
                hatch=hatch, linewidth=1.4)
        ax.text(elo + 25, i, str(elo), va="center", fontsize=9, color=INK, fontweight="bold")

    # display names without the size baked in, so it appears once in parentheses
    display = {
        "dm9": "DeepMind searchless", "dm136": "DeepMind searchless",
        "dm270": "DeepMind searchless", "cgpt": "chess-GPT, Karvonen",
    }
    ticklabels = []
    for rid, label, params, elo, search, is_open, note in rows:
        name = display.get(rid, label)
        if params is None:
            ticklabels.append(f"{name} (size n/a)")
        else:
            p = f"{params:g}M"
            ticklabels.append(f"{name} ({p})")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ticklabels, fontsize=9)
    ax.set_xlabel("reported Elo")
    ax.set_xlim(0, 4000)
    ax.set_title("reported strength of open chess models (network size in parentheses)")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # rows are sorted ascending by elo, smallest at top reading down

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ACC2, edgecolor=ACC2, label="no search at play time"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ACC3, edgecolor=ACC3, label="plays with search"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ACC, edgecolor=ACC, label="kibitzer"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=TERTIARY, hatch="///", label="not open source"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False, fontsize=9)

    fig.text(0.01, 0.005, FOOTNOTE, fontsize=7.5, color=TERTIARY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_scatter(OUT_DIR / "fig_landscape_scatter.png")
    fig_bars(OUT_DIR / "fig_landscape_bars.png")


if __name__ == "__main__":
    main()
