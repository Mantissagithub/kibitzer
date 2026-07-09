from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


REPORT_DIR = Path("reports/preference_repair")
BUFFER_PATH = Path("runs/preference/r1_teacher_pairs_sf12.jsonl")
CHECKPOINT_PATH = Path("runs/preference/preference_repair.pt")
GATE_PATH = REPORT_DIR / "preference_repair_vs2700_s128_g80_seed31.jsonl"
OPPONENT_ELO = 2700
TACTICAL_R1_SCORE = 0.294
TACTICAL_R1_ELO = 2548


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def elo_from_score(score: float) -> int:
    clipped = min(0.999, max(0.001, score))
    return round(OPPONENT_ELO + 400 * math.log10(clipped / (1.0 - clipped)))


def gate_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    score = 0.0
    wins = 0
    draws = 0
    losses = 0
    for i, row in enumerate(rows, start=1):
        point = float(row.get("score", 0.0))
        score += point
        wins += int(point == 1.0)
        draws += int(point == 0.5)
        losses += int(point == 0.0)
        rate = score / i
        out.append(
            {
                "game": i,
                "score": score,
                "rate": rate,
                "elo": elo_from_score(rate),
                "wins": wins,
                "draws": draws,
                "losses": losses,
            }
        )
    return out


def load_checkpoint_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("eval_metrics") or {}
    return {str(key): float(value) for key, value in metrics.items()}


def plot_gate(curve: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if curve:
        x = [row["game"] for row in curve]
        y = [row["rate"] for row in curve]
        ax.plot(x, y, color="#4c78a8", linewidth=2, marker="o", markersize=3)
        ax.axhline(TACTICAL_R1_SCORE, color="#f58518", linestyle="--", label="tactical R1 seed23")
        ax.text(x[-1], y[-1], f"  final {y[-1]:.3f}", va="center", fontsize=9)
    ax.set_title("Preference repair external gate score")
    ax.set_xlabel("games completed")
    ax.set_ylabel("score rate")
    ax.set_ylim(0.0, 0.55)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_gate_score_curve.png", dpi=160)
    plt.close(fig)


def plot_elo(curve: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if curve:
        x = [row["game"] for row in curve]
        y = [row["elo"] for row in curve]
        ax.plot(x, y, color="#54a24b", linewidth=2, marker="o", markersize=3)
        ax.axhline(TACTICAL_R1_ELO, color="#f58518", linestyle="--", label="tactical R1 seed23")
        ax.axhline(OPPONENT_ELO, color="#333333", linewidth=1, linestyle=":", label="opponent label")
        ax.text(x[-1], y[-1], f"  final {y[-1]:.0f}", va="center", fontsize=9)
    ax.set_title("Preference repair implied Elo during gate")
    ax.set_xlabel("games completed")
    ax.set_ylabel("implied Elo vs Maia/Leela-2700")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_gate_elo_curve.png", dpi=160)
    plt.close(fig)


def plot_buffer(rows: list[dict[str, Any]]) -> None:
    margins = [float(row.get("teacher_margin", 0.0)) for row in rows]
    probs = [float(row.get("bad_policy_prob", 0.0)) for row in rows]
    floor = sum(1 for row in rows if row.get("bad_score_is_floor"))
    direct = max(0, len(rows) - floor)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(margins, bins=50, color="#4c78a8", alpha=0.85)
    axes[0].axvline(percentile(margins, 0.5), color="#f58518", linestyle="--", label="p50")
    axes[0].axvline(percentile(margins, 0.9), color="#e45756", linestyle="--", label="p90")
    axes[0].set_title("Teacher margin distribution")
    axes[0].set_xlabel("good_score - bad_score")
    axes[0].set_ylabel("pairs")
    axes[0].legend()

    axes[1].hist(probs, bins=50, color="#72b7b2", alpha=0.85)
    axes[1].set_title("Policy probability on rejected move")
    axes[1].set_xlabel("bad_policy_prob")
    axes[1].set_ylabel("pairs")
    fig.suptitle(f"Preference buffer: {len(rows):,} pairs, {direct:,} direct / {floor:,} floor")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_buffer_diagnostics.png", dpi=160)
    plt.close(fig)


def plot_offline(metrics: dict[str, float]) -> None:
    keys = ["dpo_loss", "ce_loss", "anchor_kl", "pair_acc", "pair_margin"]
    values = [metrics.get(key, 0.0) for key in keys]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(keys, values, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"])
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set_title("Best checkpoint held-out preference metrics")
    ax.set_ylabel("metric value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_offline_metrics.png", dpi=160)
    plt.close(fig)


def write_readme(
    buffer_rows: list[dict[str, Any]],
    curve: list[dict[str, float]],
    metrics: dict[str, float],
) -> None:
    final = curve[-1] if curve else {"game": 0, "wins": 0, "draws": 0, "losses": 0, "rate": 0.0, "elo": 0.0}
    margins = [float(row.get("teacher_margin", 0.0)) for row in buffer_rows]
    probs = [float(row.get("bad_policy_prob", 0.0)) for row in buffer_rows]
    lines = [
        "# Preference repair report",
        "",
        "## Verdict",
        "",
        "- Rejected. The external gate regressed hard against the current tactical R1 checkpoint.",
        f"- Gate stopped at {int(final['game'])}/80 games: {int(final['wins'])}W/{int(final['draws'])}D/{int(final['losses'])}L, score rate {final['rate']:.3f}, implied Elo {final['elo']:.0f}.",
        f"- Tactical R1 reference is score rate {TACTICAL_R1_SCORE:.3f}, implied Elo {TACTICAL_R1_ELO}. Preference repair was already below the promotion band.",
        "- Offline preference metrics improved enough to save a checkpoint, but they did not predict external play.",
        "",
        "## Figures",
        "",
        "- ![gate score curve](fig_gate_score_curve.png)",
        "- ![gate elo curve](fig_gate_elo_curve.png)",
        "- ![buffer diagnostics](fig_buffer_diagnostics.png)",
        "- ![offline metrics](fig_offline_metrics.png)",
        "",
        "## Buffer",
        "",
        "| pairs | margin mean | margin p50 | margin p90 | bad-policy-prob mean | floor bad score |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(buffer_rows):,} | {sum(margins)/max(1,len(margins)):.3f} | {percentile(margins,0.5):.3f} | {percentile(margins,0.9):.3f} | {sum(probs)/max(1,len(probs)):.3f} | {sum(1 for row in buffer_rows if row.get('bad_score_is_floor')):,} |",
        "",
        "## Offline Best Checkpoint",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in ["dpo_loss", "ce_loss", "anchor_kl", "pair_acc", "pair_margin"]:
        lines.append(f"| `{key}` | {metrics.get(key, 0.0):.4f} |")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "Use a much more conservative retry only if continuing this branch:",
            "",
            "```bash",
            "ACTION=train \\",
            "PREFERENCE_JSONL=runs/preference/r1_teacher_pairs_sf12.jsonl \\",
            "OUTPUT_CHECKPOINT=runs/preference/preference_repair_anchor_r1.pt \\",
            "LEARNING_RATE=3e-6 \\",
            "EPOCHS=1 \\",
            "BETA=0.03 \\",
            "CE_WEIGHT=0.5 \\",
            "ANCHOR_WEIGHT=0.5 \\",
            "bash scripts/run_preference_repair.sh",
            "```",
            "",
        ]
    )
    (REPORT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    buffer_rows = read_jsonl(BUFFER_PATH)
    gate_rows = read_jsonl(GATE_PATH)
    curve = gate_curve(gate_rows)
    metrics = load_checkpoint_metrics(CHECKPOINT_PATH)

    plot_gate(curve)
    plot_elo(curve)
    plot_buffer(buffer_rows)
    plot_offline(metrics)
    write_readme(buffer_rows, curve, metrics)
    print(f"wrote preference report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
