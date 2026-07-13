"""Regenerate reports/run_analysis/ from local training/evaluation evidence.

Reads LOGBOOK.md, saved checkpoint metadata under runs/, the locked
common-oracle diagnostics reports under runs/diagnostics/, and the small
Stockfish match reports under eval_pgns/. Every number plotted traces back to
one of those repo-local files; metrics with no local evidence are drawn as an
explicit "not recorded" panel instead of being omitted silently or guessed.

Usage:
    uv run python scripts/plot_run_analysis.py
    uv run python scripts/plot_run_analysis.py --output-dir reports/run_analysis
"""

from __future__ import annotations

import argparse
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

from kibitzer.diagnostics import VALUE_BINS  # noqa: E402
from kibitzer.run_analysis import (  # noqa: E402
    EvidenceLog,
    assert_validation_split,
    collect_joint_stage_epochs,
    collect_match_result,
    collect_oracle_bin_metrics,
    collect_search_strategies,
    collect_value_repair_epochs,
    collect_value_stage_epochs,
    load_checkpoint_config,
    load_json,
    parse_strategy_key,
)


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
        "savefig.dpi": 150,
    }
)

STRATEGY_ORDER_CHECKPOINTS = ["phase2", "joint", "value_repair"]
STRATEGY_ORDER_MODES = ["raw", "s64:v0", "s64:v0.5", "s64:v1"]


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _footnote(fig: plt.Figure, lines: list[str]) -> None:
    text = "Source: " + "  |  ".join(lines)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=6.5, color="#444444")


def _unavailable_axis(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=9.5,
        wrap=True,
        transform=ax.transAxes,
        color="#a33a00",
        bbox={"boxstyle": "round", "facecolor": "#fff3e0", "edgecolor": "#a33a00"},
    )


def plot_value_metrics_by_epoch(
    *, decisions_text: str, run_dir: Path, output_dir: Path, log: EvidenceLog
) -> Path | None:
    value_epochs = collect_value_stage_epochs(decisions_text)
    repair_epochs = collect_value_repair_epochs(run_dir / "value_repair")
    log.record(
        "value stage epoch curve",
        available=bool(value_epochs),
        source="LOGBOOK.md D25 (value stage table)",
    )
    log.record(
        "value-repair stage epoch curve",
        available=bool(repair_epochs),
        source="runs/value_repair/value_repair_best_epoch_{1,2,3}.pt (eval_metrics)",
    )
    if not value_epochs and not repair_epochs:
        return None

    # Color encodes the training run; solid/circle vs. dashed/square encodes which
    # of the two metrics named in each subplot title is being drawn (first vs.
    # second). A single shared legend at the bottom explains the run colors so
    # per-panel legends never have to compete with the data for space.
    stage_colors = {
        "value stage (Stockfish d14 regression, 5 epochs)": "#1f77b4",
        "value-repair stage (Stage A balanced repair, 3 epochs)": "#d62728",
    }
    series = {
        "value stage (Stockfish d14 regression, 5 epochs)": value_epochs,
        "value-repair stage (Stage A balanced repair, 3 epochs)": repair_epochs,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("mse", "mae", "MSE (solid) / MAE (dashed) — value loss, error units"),
        ("pearson", "sign_accuracy", "Pearson r (solid) / sign accuracy (dashed)"),
        ("r2", None, "R²"),
    ]
    for ax, (metric_a, metric_b, title) in zip(axes, panels, strict=True):
        for label, rows in series.items():
            if not rows:
                continue
            color = stage_colors[label]
            epochs = [row["epoch"] for row in rows]
            if metric_a in rows[0]:
                ax.plot(epochs, [row[metric_a] for row in rows], marker="o", color=color)
            if metric_b is not None and metric_b in rows[0]:
                ax.plot(
                    epochs,
                    [row[metric_b] for row in rows],
                    marker="s",
                    linestyle="--",
                    color=color,
                )
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("epoch (within stage)")
        ax.set_xticks(sorted({row["epoch"] for rows in series.values() for row in rows}))

    legend_handles = [
        plt.Line2D([0], [0], color=color, marker="o", label=label)
        for label, color in stage_colors.items()
        if series[label]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.08))
    fig.suptitle("Value-head training loss and value metrics by epoch")
    _footnote(
        fig,
        [
            "LOGBOOK.md §D25 (value stage, hand-recorded stdout)",
            "runs/value_repair/value_repair_best_epoch_{1,2,3}.pt (programmatic eval_metrics)",
        ],
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    path = output_dir / "fig1_value_metrics_by_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_policy_metrics_by_epoch(
    *, decisions_text: str, run_dir: Path, output_dir: Path, log: EvidenceLog
) -> Path:
    joint_epochs = collect_joint_stage_epochs(decisions_text)
    policy_yaml = load_checkpoint_config(run_dir / "policy" / "policy_final.pt")
    log.record(
        "joint stage policy epoch curve",
        available=bool(joint_epochs),
        source="LOGBOOK.md D27 (joint stage table)",
    )
    log.record(
        "policy-only Phase-1 stage epoch curve",
        available=False,
        source="scripts/train_bc.py",
        note="train_bc.py does not compute held-out metrics or persist a per-epoch loss log locally",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax_ce, ax_acc, ax_unavailable = axes

    if joint_epochs:
        epochs = [row["epoch"] for row in joint_epochs]
        ax_ce.plot(epochs, [row["policy_cross_entropy"] for row in joint_epochs], marker="o", color="#1f77b4")
        ax_ce.set_xlabel("epoch")
        ax_ce.set_ylabel("policy cross-entropy")
        ax_ce.set_title("Joint stage: policy CE (lower is better)")
        ax_ce.set_xticks(epochs)

        ax_acc.plot(
            epochs,
            [row["policy_top1_accuracy"] for row in joint_epochs],
            marker="o",
            label="teacher top-1 accuracy",
        )
        ax_acc.plot(
            epochs,
            [row["policy_teacher_coverage"] for row in joint_epochs],
            marker="s",
            linestyle="--",
            label="teacher-set coverage",
        )
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("fraction")
        ax_acc.set_title("Joint stage: policy agreement with teacher")
        ax_acc.set_xticks(epochs)
        ax_acc.legend(fontsize=7)
    else:
        _unavailable_axis(ax_ce, "joint stage policy CE:\nnot found in LOGBOOK.md")
        _unavailable_axis(ax_acc, "joint stage policy agreement:\nnot found in LOGBOOK.md")

    policy_note = "not recorded locally\n(scripts/train_bc.py logs\nonly a live tqdm postfix;\nno eval or history is saved)"
    if policy_yaml is not None:
        epochs_trained = policy_yaml.get("training_objective")
        policy_note = (
            f"Phase-1 policy-only stage\n(objective={epochs_trained})\n"
            "has no per-epoch loss log:\n" + policy_note
        )
    _unavailable_axis(ax_unavailable, "Phase-1 policy-only epoch curve:\n" + policy_note)

    fig.suptitle("Policy metrics by epoch")
    _footnote(
        fig,
        [
            "LOGBOOK.md §D27 (joint stage, hand-recorded stdout)",
            "runs/policy/policy_final.yaml (final config only, no epoch history)",
        ],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = output_dir / "fig2_policy_metrics_by_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _merge_oracle_bins(
    payloads: list[tuple[dict, str]], checkpoints: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for checkpoint in checkpoints:
        for payload, _source in payloads:
            bins = collect_oracle_bin_metrics(payload, checkpoint=checkpoint)
            if bins is not None:
                merged[checkpoint] = bins
                break
    return merged


def plot_common_oracle_value_by_bin(
    *, diagnostics_dir: Path, output_dir: Path, log: EvidenceLog, repo_root: Path
) -> Path | None:
    validation = load_json(diagnostics_dir / "validation.json")
    value_repair_validation = load_json(diagnostics_dir / "value_repair_validation.json")
    payloads = []
    for payload, path in (
        (validation, diagnostics_dir / "validation.json"),
        (value_repair_validation, diagnostics_dir / "value_repair_validation.json"),
    ):
        if payload is None:
            continue
        assert_validation_split(payload, source=_rel(path, repo_root))
        payloads.append((payload, _rel(path, repo_root)))

    checkpoints = ["phase2", "joint", "value_repair"]
    bins_by_checkpoint = _merge_oracle_bins(payloads, checkpoints)
    log.record(
        "common-oracle value metrics by bin",
        available=bool(bins_by_checkpoint),
        source="runs/diagnostics/validation.json, runs/diagnostics/value_repair_validation.json",
    )
    if not bins_by_checkpoint:
        return None

    colors = {"phase2": "#1f77b4", "joint": "#ff7f0e", "value_repair": "#2ca02c"}
    x = range(len(VALUE_BINS))
    width = 0.8 / max(len(bins_by_checkpoint), 1)

    fig, (ax_mae, ax_sign) = plt.subplots(1, 2, figsize=(11, 4.5))
    for offset, checkpoint in enumerate(checkpoints):
        bins = bins_by_checkpoint.get(checkpoint)
        if bins is None:
            continue
        positions = [pos + offset * width for pos in x]
        mae_values = [bins[name]["mae"] for name in VALUE_BINS]
        sign_values = [bins[name]["sign_accuracy"] for name in VALUE_BINS]
        color = colors.get(checkpoint, None)
        ax_mae.bar(positions, mae_values, width=width, label=checkpoint, color=color)
        ax_sign.bar(positions, sign_values, width=width, label=checkpoint, color=color)

    centered = [pos + width * (len(bins_by_checkpoint) - 1) / 2 for pos in x]
    for ax, title, ylabel in (
        (ax_mae, "Value MAE by oracle bin (lower is better)", "MAE"),
        (ax_sign, "Value sign accuracy by oracle bin (higher is better)", "sign accuracy"),
    ):
        ax.set_xticks(centered)
        ax.set_xticklabels(VALUE_BINS)
        ax.set_xlabel("|Stockfish depth-20 evaluation| bin")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Common-oracle value metrics by value bin (validation split, 800 positions)")
    _footnote(
        fig,
        [
            "runs/diagnostics/validation.json (split=validation)",
            "runs/diagnostics/value_repair_validation.json (split=validation)",
        ],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path = output_dir / "fig3_common_oracle_value_by_bin.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _ordered_strategy_keys(strategies: dict[str, dict[str, float]]) -> list[str]:
    ordered = []
    for checkpoint in STRATEGY_ORDER_CHECKPOINTS:
        for mode in STRATEGY_ORDER_MODES:
            key = f"{checkpoint}:{mode}"
            if key in strategies:
                ordered.append(key)
    # Anything unexpected still gets shown, appended at the end.
    for key in strategies:
        if key not in ordered:
            ordered.append(key)
    return ordered


def plot_search_regret_by_checkpoint_sims_scale(
    *, diagnostics_dir: Path, output_dir: Path, log: EvidenceLog, repo_root: Path
) -> Path | None:
    validation = load_json(diagnostics_dir / "validation.json")
    value_repair_validation = load_json(diagnostics_dir / "value_repair_validation.json")
    strategies: dict[str, dict[str, float]] = {}
    sources = []
    for payload, path in (
        (validation, diagnostics_dir / "validation.json"),
        (value_repair_validation, diagnostics_dir / "value_repair_validation.json"),
    ):
        if payload is None:
            continue
        assert_validation_split(payload, source=_rel(path, repo_root))
        strategies.update(collect_search_strategies(payload))
        sources.append(_rel(path, repo_root))

    log.record(
        "search regret / near-best / best-move by checkpoint, sims, value scale",
        available=bool(strategies),
        source=", ".join(sources) if sources else "runs/diagnostics/",
    )
    if not strategies:
        return None

    keys = _ordered_strategy_keys(strategies)
    labels = []
    for key in keys:
        checkpoint, sims, scale = parse_strategy_key(key)
        labels.append(checkpoint if sims == "raw" else f"{checkpoint}\n{sims} {scale}")

    x = range(len(keys))
    fig, (ax_regret, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.27
    ax_regret.bar(
        [pos - width for pos in x], [strategies[k]["mean_cp"] for k in keys], width=width, label="mean regret"
    )
    ax_regret.bar([pos for pos in x], [strategies[k]["p90_cp"] for k in keys], width=width, label="p90 regret")
    ax_regret.bar(
        [pos + width for pos in x], [strategies[k]["p95_cp"] for k in keys], width=width, label="p95 regret"
    )
    ax_regret.set_ylabel("centipawns lost vs. depth-20 best move (lower is better)")
    ax_regret.set_title("Move regret by checkpoint / simulations / value scale")
    ax_regret.set_xticks(list(x))
    ax_regret.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax_regret.legend(fontsize=8)

    ax_acc.bar(
        [pos - width / 2 for pos in x],
        [strategies[k]["near_best_accuracy"] for k in keys],
        width=width,
        label="near-best (≤50cp)",
    )
    ax_acc.bar(
        [pos + width / 2 for pos in x],
        [strategies[k]["best_move_accuracy"] for k in keys],
        width=width,
        label="exact best move",
    )
    ax_acc.set_ylabel("fraction of positions")
    ax_acc.set_title("Move accuracy by checkpoint / simulations / value scale")
    ax_acc.set_xticks(list(x))
    ax_acc.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax_acc.legend(fontsize=8)

    fig.suptitle("Search regret and move accuracy (validation split, 800 positions)")
    _footnote(fig, sources)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = output_dir / "fig4_search_regret_by_checkpoint_sims_scale.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_match_wdl_score(
    *, eval_pgns_dir: Path, output_dir: Path, log: EvidenceLog, repo_root: Path
) -> Path | None:
    candidates = [
        "search_vs_stockfish_1320.json",
        "search_vs_stockfish_1320_s256.json",
        "joint_vs_stockfish_1320_s256.json",
    ]
    results = []
    sources = []
    for name in candidates:
        path = eval_pgns_dir / name
        result = collect_match_result(path)
        if result is not None:
            results.append(result)
            sources.append(_rel(path, repo_root))

    log.record(
        "match WDL / score vs Stockfish",
        available=bool(results),
        source=", ".join(sources) if sources else str(eval_pgns_dir),
        note="small-sample (10 games/config); noisy by construction",
    )
    if not results:
        return None

    # Label by the checkpoint/sims actually recorded inside each JSON report, not
    # the filename: e.g. eval_pgns/search_vs_stockfish_1320.json's own "checkpoint"
    # field is joint_best.pt, not a generic "search" checkpoint.
    labels = [
        f"{Path(result['checkpoint']).name}\nsims={result['simulations']}, n={result['games']}"
        for result in results
    ]
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(10, 5))
    wins = [r["wins"] for r in results]
    draws = [r["draws"] for r in results]
    losses = [r["losses"] for r in results]
    ax.bar(x, wins, label="wins", color="#2ca02c")
    ax.bar(x, draws, bottom=wins, label="draws", color="#7f7f7f")
    bottom_wd = [w + d for w, d in zip(wins, draws, strict=True)]
    ax.bar(x, losses, bottom=bottom_wd, label="losses", color="#d62728")
    for pos, result in zip(x, results, strict=True):
        ax.text(
            pos,
            result["games"] + 0.3,
            f"score {100 * result['score']:.0f}%",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("games")
    ax.set_ylim(0, max(r["games"] for r in results) + 2)
    ax.set_title(
        "Match result vs. Stockfish-1320 — NOISY, do not read as an Elo estimate\n"
        "(each point is a single 10-game match; see LOGBOOK.md §D27 caveat)"
    )
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Match WDL and score (separate, clearly noisy panel)")
    _footnote(fig, sources)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    path = output_dir / "fig5_match_wdl_score_noisy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def render_readme(
    *,
    output_dir: Path,
    figure_paths: dict[str, Path | None],
    log: EvidenceLog,
    diagnostics_dir: Path,
) -> Path:
    lines = [
        "# Run analysis report",
        "",
        "Regenerated deterministically from repo-local training/evaluation evidence by",
        "`scripts/plot_run_analysis.py`. Every figure lists its exact source file(s) in a",
        "footnote; metrics with no local evidence are drawn as an explicit orange",
        "\"not recorded\" panel rather than omitted or invented.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run python scripts/plot_run_analysis.py",
        "```",
        "",
        "## Figures",
        "",
    ]

    descriptions = {
        "fig1_value_metrics_by_epoch.png": (
            "Value-head training loss and value metrics by epoch",
            "Value stage (5 epochs, Stockfish depth-14 regression) and value-repair "
            "Stage A (3 epochs, balanced bin sampling) MSE/MAE, Pearson r/sign accuracy, "
            "and R² across training epochs. Two independent training runs, plotted on "
            "shared per-metric axes so scales are never mixed.",
        ),
        "fig2_policy_metrics_by_epoch.png": (
            "Policy metrics by epoch",
            "Joint distillation stage (5 epochs) policy cross-entropy and teacher "
            "agreement/coverage. The Phase-1 policy-only stage is drawn as an explicit "
            "unavailable panel: `scripts/train_bc.py` never computes held-out metrics or "
            "persists a per-epoch log, so no epoch curve exists locally for it.",
        ),
        "fig3_common_oracle_value_by_bin.png": (
            "Common-oracle value metrics by value bin",
            "Value MAE and sign accuracy for phase2, joint, and value-repair checkpoints "
            "against the locked common-oracle **validation** split (800 real-game "
            "positions, depth-20 Stockfish, game-disjoint from training), broken out by "
            "the quiet/edge/decisive/won magnitude bins.",
        ),
        "fig4_search_regret_by_checkpoint_sims_scale.png": (
            "Search regret, near-best, and best-move accuracy by checkpoint/sims/value-scale",
            "Mean/p90/p95 move regret (centipawns lost vs. the depth-20 best move) and "
            "near-best/exact-best-move accuracy, for each checkpoint at raw policy play "
            "and at 64 PUCT simulations across value_scale in {0, 0.5, 1}. Regret (cp) and "
            "accuracy (fraction) are kept on separate axes.",
        ),
        "fig5_match_wdl_score_noisy.png": (
            "Match WDL and score (noisy, separate panel)",
            "Full-game win/draw/loss results vs. Stockfish-1320 from the small local "
            "cutechess-style matches. Each bar is a single 10-game match — explicitly "
            "labeled as noisy and kept out of the metric-based figures above so it cannot "
            "be mistaken for a precise Elo comparison.",
        ),
    }

    for filename, (title, description) in descriptions.items():
        path = figure_paths.get(filename)
        lines.append(f"### {title}")
        lines.append("")
        if path is not None:
            lines.append(f"![{title}]({path.name})")
            lines.append("")
        else:
            lines.append("_Not generated: no local evidence was found for this figure._")
            lines.append("")
        lines.append(description)
        lines.append("")

    lines.append("## Evidence inventory")
    lines.append("")
    lines.append("| evidence | available | source |")
    lines.append("|---|---|---|")
    for entry in log.entries:
        status = "yes" if entry.available else "**no**"
        note = f" ({entry.note})" if entry.note else ""
        lines.append(f"| {entry.label} | {status} | {entry.source}{note} |")
    lines.append("")

    lines.append("## Inspected but not charted")
    lines.append("")
    lines.append(
        "- `runs/diagnostics/value_label_audit.json`: a one-shot depth-14-vs-depth-20 "
        "label-ceiling audit (overall MAE 0.0217, sign disagreement 2.90%, decision "
        "`cached_labels_have_headroom`). It is a single scalar verdict, not a metric "
        "series across epochs/checkpoints/bins, so it is reported here as text rather "
        "than forced into a chart."
    )
    lines.append(
        "- `runs/diagnostics/value_repair_head_to_head.json`: a narrower "
        "phase2-vs-value_repair comparison (only the `s64:v1` configuration) that is a "
        "strict subset of `value_repair_validation.json`, which is used for the figures "
        "above instead."
    )
    lines.append(
        "- `runs/policy/policy_final.yaml`, `runs/joint_distill/joint_best.yaml`, "
        "`runs/value/value_final.yaml`: final-checkpoint config/metric snapshots, used to "
        "cross-check the epoch curves above but not separately plotted."
    )
    lines.append("")

    lines.append("## Locked test split")
    lines.append("")
    lines.append(
        "Every `runs/diagnostics/*.json` report consumed by this script asserts "
        '`"split": "validation"` before use (see `assert_validation_split` in '
        "`kibitzer/run_analysis.py`); the script raises rather than plotting if a report "
        "is not the validation split. No locked test-split evaluation exists locally as "
        "of this run, so none is presented as evaluated here."
    )
    lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures and README (default: <repo-root>/reports/run_analysis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root: Path = args.repo_root
    output_dir: Path = args.output_dir or (repo_root / "reports" / "run_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = repo_root / "LOGBOOK.md"
    if not decisions_path.is_file():
        raise SystemExit(f"missing LOGBOOK.md: {decisions_path}")
    decisions_text = decisions_path.read_text(encoding="utf-8")

    run_dir = repo_root / "runs"
    diagnostics_dir = run_dir / "diagnostics"
    eval_pgns_dir = repo_root / "eval_pgns"

    log = EvidenceLog()
    figure_paths: dict[str, Path | None] = {}

    print("[1/6] value metrics by epoch", flush=True)
    figure_paths["fig1_value_metrics_by_epoch.png"] = plot_value_metrics_by_epoch(
        decisions_text=decisions_text, run_dir=run_dir, output_dir=output_dir, log=log
    )

    print("[2/6] policy metrics by epoch", flush=True)
    figure_paths["fig2_policy_metrics_by_epoch.png"] = plot_policy_metrics_by_epoch(
        decisions_text=decisions_text, run_dir=run_dir, output_dir=output_dir, log=log
    )

    print("[3/6] common-oracle value metrics by bin", flush=True)
    figure_paths["fig3_common_oracle_value_by_bin.png"] = plot_common_oracle_value_by_bin(
        diagnostics_dir=diagnostics_dir, output_dir=output_dir, log=log, repo_root=repo_root
    )

    print("[4/6] search regret by checkpoint/sims/value_scale", flush=True)
    figure_paths["fig4_search_regret_by_checkpoint_sims_scale.png"] = (
        plot_search_regret_by_checkpoint_sims_scale(
            diagnostics_dir=diagnostics_dir, output_dir=output_dir, log=log, repo_root=repo_root
        )
    )

    print("[5/6] match WDL / score (noisy panel)", flush=True)
    figure_paths["fig5_match_wdl_score_noisy.png"] = plot_match_wdl_score(
        eval_pgns_dir=eval_pgns_dir, output_dir=output_dir, log=log, repo_root=repo_root
    )

    print("[6/6] writing README index", flush=True)
    readme_path = render_readme(
        output_dir=output_dir, figure_paths=figure_paths, log=log, diagnostics_dir=diagnostics_dir
    )

    print("\nDONE", flush=True)
    for name, path in figure_paths.items():
        print(f"  {name}: {'generated -> ' + str(path) if path else 'skipped (no local evidence)'}")
    print(f"  README: {readme_path}")


if __name__ == "__main__":
    main()
