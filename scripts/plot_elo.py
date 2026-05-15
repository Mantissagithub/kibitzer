"""plot estimated elo across checkpoints from hf repo names.

reads ``Pradheep1647/kibitzer-sft-elo-*-step-*`` repo names from hugging face,
parses estimated ratings (or computes them from legacy plus/minus diffs against
the stockfish-elo-1320 baseline), and renders a dark-themed plot.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


_RE_RATED  = re.compile(r"-elo-(\d{4})-step-(\d{6})$")
_RE_LEGACY = re.compile(r"-elo-(plus|minus)-(\d{4})-step-(\d{6})$")

_BASELINE_RATING = 1320  # stockfish-elo-1320 (default eval opponent)


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _env(key: str, file_env: dict[str, str]) -> str:
    return os.environ.get(key) or file_env.get(key, "")


def _parse(repo_id: str) -> tuple[int, int] | None:
    short = repo_id.split("/", 1)[-1]
    m = _RE_RATED.search(short)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = _RE_LEGACY.search(short)
    if m:
        sign = 1 if m.group(1) == "plus" else -1
        return int(m.group(3)), _BASELINE_RATING + sign * int(m.group(2))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hf-username", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-repo-prefix", default="kibitzer-sft")
    p.add_argument("--out", default="docs/elo_progress.png")
    p.add_argument("--baseline", type=int, default=_BASELINE_RATING,
                   help="stockfish reference rating to draw as a horizontal line")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env = _read_env(Path(".env"))
    username = args.hf_username or _env("HF_USERNAME", env)
    token = args.hf_token or _env("HF_TOKEN", env)
    if not username or not token:
        print("error: HF username/token missing; set .env or pass args")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub is not installed")
        return 1

    api = HfApi(token=token)
    needle = f"{args.hf_repo_prefix}-elo-"
    points: list[tuple[int, int]] = []
    pending_steps: list[int] = []
    for info in api.list_models(author=username, search=needle, token=token):
        rid = getattr(info, "id", None) or getattr(info, "modelId", None)
        if not (rid and rid.startswith(f"{username}/{needle}")):
            continue
        parsed = _parse(rid)
        if parsed is not None:
            points.append(parsed)
            continue
        m = re.search(r"-elo-pending-step-(\d{6})$", rid)
        if m:
            pending_steps.append(int(m.group(1)))

    points.sort()
    pending_steps.sort()
    if not points:
        print("no evaluated checkpoints found")
        return 1

    import matplotlib.pyplot as plt

    bg       = "#0d1117"
    panel    = "#161b22"
    grid     = "#21262d"
    fg       = "#e6edf3"
    muted    = "#8b97a8"
    accent   = "#7ee787"
    accent2  = "#79c0ff"
    warn     = "#f0883e"

    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor":   panel,
        "axes.edgecolor":   grid,
        "axes.labelcolor":  fg,
        "axes.titlecolor":  fg,
        "xtick.color":      muted,
        "ytick.color":      muted,
        "grid.color":       grid,
        "text.color":       fg,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
    })

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    steps   = [s for s, _ in points]
    ratings = [r for _, r in points]

    ax.axhline(args.baseline, color=warn, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Stockfish baseline ({args.baseline})")

    ax.plot(steps, ratings, color=accent, linewidth=2, marker="o",
            markersize=5.5, markerfacecolor=accent, markeredgecolor=panel,
            markeredgewidth=1.2, label="Kibitzer estimated Elo", zorder=3)

    for s, r in points:
        ax.annotate(f"{r}", xy=(s, r), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=muted)

    for s in pending_steps:
        ax.axvline(s, color=muted, linestyle=":", linewidth=0.6, alpha=0.35, zorder=1)

    ax.set_xlabel("training step")
    ax.set_ylabel("estimated Elo")
    ax.set_title("Kibitzer — estimated Elo across SFT checkpoints",
                 loc="left", fontsize=14, fontweight="600", pad=14)

    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(grid)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)

    if pending_steps:
        subtitle = (f"{len(points)} evaluated · {len(pending_steps)} pending "
                    f"(dotted) · baseline {args.baseline}")
    else:
        subtitle = f"{len(points)} evaluated · baseline {args.baseline}"
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            color=muted, fontsize=10)

    leg = ax.legend(loc="lower right", frameon=True, facecolor=panel,
                    edgecolor=grid, labelcolor=fg, framealpha=0.9)
    leg.get_frame().set_linewidth(0.8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=bg, bbox_inches="tight", dpi=160)
    print(f"wrote {out_path} ({len(points)} points, {len(pending_steps)} pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
