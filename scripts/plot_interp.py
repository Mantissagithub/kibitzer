# mechanistic-interpretability figures for the 100M... actually 15.2M single-position (ctx=1)
# shaw relative-attention chess model, from 3 replayed games vs Leela-2700 (win/draw/loss).
# data: interp/data/{win,draw,loss}.npz, *_plies.json, *_acts.json, summary.json
# run: uv run python scripts/plot_interp.py

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
import numpy as np  # noqa: E402

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
        "savefig.dpi": 180,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "interp" / "data"
FIG_DIR = REPO_ROOT / "interp" / "figures"

GAMES = ["win", "draw", "loss"]
GAME_COLOR = {"win": "#4c9a5b", "draw": "#888888", "loss": "#c0392b"}
GAME_LABEL = {"win": "win", "draw": "draw", "loss": "loss"}

FILES = "abcdefgh"


def sq_to_board(vec: np.ndarray) -> np.ndarray:
    """[64] vector -> [8,8] board matrix with rank 8 on top, a-file on left."""
    board = np.zeros((8, 8), dtype=vec.dtype)
    for s in range(64):
        file_ = s % 8
        rank = s // 8
        board[7 - rank, file_] = vec[s]
    return board


def style_board_axes(ax: plt.Axes) -> None:
    ax.set_xticks(range(8))
    ax.set_xticklabels(list(FILES), fontsize=6)
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(r) for r in range(8, 0, -1)], fontsize=6)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_npz(game: str) -> dict:
    return dict(np.load(DATA_DIR / f"{game}.npz"))


def load_json(name: str):
    with (DATA_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def fig1_head_atlas() -> None:
    data = load_npz("win")
    attn = data["attn_mean"]  # [3,8,64,64] layer, head, query, key

    received = attn.mean(axis=2)  # [3,8,64] average over queries -> attention received per key square
    boards = np.stack(
        [[sq_to_board(received[layer, head]) for head in range(8)] for layer in range(3)]
    )  # [3,8,8,8]

    vmin, vmax = boards.min(), boards.max()

    fig, axes = plt.subplots(3, 8, figsize=(16, 6.6))
    im = None
    for layer in range(3):
        for head in range(8):
            ax = axes[layer, head]
            im = ax.imshow(boards[layer, head], cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"L{layer + 1}·H{head + 1}", fontsize=8, pad=3)
            if head == 0:
                style_board_axes(ax)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    fig.suptitle(
        "what each of the 24 encoder heads attends to\n"
        "(attention received per square, averaged over the WIN game's model plies)",
        fontsize=12,
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("mean attention received", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(FIG_DIR / "fig1_head_atlas.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_head_atlas.png")


def fig2_activation_flow() -> None:
    stages = ["enc1", "enc2", "enc3", "pool"] + [f"trunk{i}" for i in range(1, 11)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for game in GAMES:
        acts = load_json(f"{game}_acts.json")
        means = [float(np.mean([p[s] for p in acts])) for s in stages]
        ax.plot(
            range(len(stages)),
            means,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=GAME_COLOR[game],
            label=GAME_LABEL[game],
        )

    boundary = stages.index("pool")
    ax.axvline(boundary + 0.5, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
    ax.text(
        boundary + 0.6,
        ax.get_ylim()[1] * 0.02 + ax.get_ylim()[0],
        "trunk →",
        fontsize=8,
        color="#555555",
        va="bottom",
    )

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean activation L2 norm")
    ax.set_title("activation norm by stage: sharp encoder dynamics, smooth trunk drift", pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.text(
        0.5,
        0.005,
        "encoder (enc1-3) does the large, non-monotonic spatial computation and mean-pool resets the norm; "
        "the trunk then rises smoothly and monotonically (no sharp jumps) across all 10 layers at ctx=1",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_DIR / "fig2_activation_flow.png")
    plt.close(fig)
    print("wrote fig2_activation_flow.png")


def fig3_value_trajectory() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    finals = {}
    for game in GAMES:
        plies = load_json(f"{game}_plies.json")
        values = [p["value"] for p in plies]
        ax.plot(
            range(len(values)),
            values,
            color=GAME_COLOR[game],
            linewidth=1.8,
            label=GAME_LABEL[game],
        )
        finals[game] = values[-1]
        ax.annotate(
            f"{values[-1]:+.2f}",
            xy=(len(values) - 1, values[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=GAME_COLOR[game],
            va="center",
            fontweight="bold",
        )

    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("model ply index")
    ax.set_ylabel("value head output")
    ax.set_title("the value head knows who is winning", pad=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_value_trajectory.png")
    plt.close(fig)
    print("wrote fig3_value_trajectory.png")


def fig4_behavior_profile() -> None:
    summary = load_json("summary.json")
    metrics = ["mean_capture_mass", "mean_check_mass", "mean_entropy", "top_capture_rate"]
    metric_labels = ["capture mass", "check mass", "entropy", "top-move\ncapture rate"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

    x = np.arange(len(metrics))
    width = 0.25
    for i, game in enumerate(GAMES):
        vals = [summary[game][m] for m in metrics]
        bars = ax_a.bar(x + (i - 1) * width, vals, width=width, color=GAME_COLOR[game], label=GAME_LABEL[game])
        for rect, val in zip(bars, vals):
            ax_a.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metric_labels, fontsize=8)
    ax_a.set_ylabel("value (entropy in nats; others are probabilities in [0,1])")
    ax_a.set_title("(a) aggregate behaviour by game outcome", pad=10)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.9)

    loss_plies = load_json("loss_plies.json")
    n = len(loss_plies)
    ax_b.plot(
        range(n),
        [p["capture_mass"] for p in loss_plies],
        color="#d68910",
        linewidth=1.6,
        label="capture mass",
    )
    ax_b.plot(
        range(n),
        [p["check_mass"] for p in loss_plies],
        color="#c0392b",
        linewidth=1.6,
        label="check mass",
    )
    ax_b.set_xlabel("model ply index (LOSS game)")
    ax_b.set_ylabel("probability mass")
    ax_b.set_title("(b) does it lash out as the LOSS game deteriorates?", pad=10)
    ax_b.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("aggressive or positional? capture/check tendencies across outcomes", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig4_behavior_profile.png")
    plt.close(fig)
    print("wrote fig4_behavior_profile.png")


def fig5_head_phase_evolution(layer: int, head: int) -> None:
    data = load_npz("win")
    keys = ["attn_sample_2", "attn_sample_12", "attn_sample_24"]
    phase_labels = ["opening (ply 2)", "middlegame (ply 12)", "late (ply 24)"]

    boards = []
    for key in keys:
        attn = data[key][layer, head]  # [64,64]
        received = attn.mean(axis=0)  # [64]
        boards.append(sq_to_board(received))
    boards = np.stack(boards)
    vmin, vmax = boards.min(), boards.max()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    im = None
    for i, ax in enumerate(axes):
        im = ax.imshow(boards[i], cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(phase_labels[i], fontsize=10)
        style_board_axes(ax)

    fig.suptitle(f"how head L{layer + 1}·H{head + 1} tracks the game (WIN game, attention received)", fontsize=12)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("attention received", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(FIG_DIR / "fig5_head_phase_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig5_head_phase_evolution.png (L{layer + 1}·H{head + 1})")


def pick_structured_head() -> tuple[int, int]:
    """pick the encoder head with the highest spatial variance of attention-received,
    as a proxy for 'most structured / least diffuse'."""
    data = load_npz("win")
    attn = data["attn_mean"]  # [3,8,64,64]
    received = attn.mean(axis=2)  # [3,8,64]
    variances = received.var(axis=2)  # [3,8]
    layer, head = np.unravel_index(np.argmax(variances), variances.shape)
    return int(layer), int(head)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_head_atlas()
    fig2_activation_flow()
    fig3_value_trajectory()
    fig4_behavior_profile()
    layer, head = pick_structured_head()
    fig5_head_phase_evolution(layer, head)


if __name__ == "__main__":
    main()
