"""Evidence loading for the local training/evaluation run-analysis report.

Every function here reads a single kind of repo-local artifact (a markdown
table in LOGBOOK.md, a checkpoint's non-tensor metadata, a diagnostics JSON
report, or a match-result JSON report) and returns plain data. Nothing here
invents numbers: missing or malformed evidence is reported as unavailable
rather than guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


VALIDATION_SPLIT = "validation"


@dataclass(frozen=True)
class Evidence:
    """Tracks whether one evidence source was found and where it came from."""

    label: str
    available: bool
    source: str
    note: str = ""


@dataclass
class EvidenceLog:
    """Accumulates :class:`Evidence` entries as figures are assembled."""

    entries: list[Evidence] = field(default_factory=list)

    def record(self, label: str, *, available: bool, source: str, note: str = "") -> None:
        self.entries.append(Evidence(label, available=available, source=source, note=note))


def _clean_cell(cell: str) -> str:
    return cell.strip().replace("**", "")


def _parse_number(cell: str) -> float:
    text = _clean_cell(cell)
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_markdown_epoch_table(
    text: str,
    *,
    header_marker: str,
    columns: list[str],
) -> list[dict[str, float]]:
    """Parse a `| epoch | ... |` markdown table into per-epoch metric dicts.

    ``header_marker`` is a unique substring of the header row used to locate
    the table inside a larger markdown document. ``columns`` names the
    metric columns in the order they appear after the leading epoch column.
    Returns an empty list if the marker is not found (evidence unavailable,
    never fabricated).
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if header_marker in line:
            start = index
            break
    if start is None:
        return []

    rows: list[dict[str, float]] = []
    # start + 1 is the `|---:|...` alignment row; data begins at start + 2.
    for line in lines[start + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = [cell for cell in stripped.strip("|").split("|")]
        if len(cells) != 1 + len(columns):
            break
        epoch_cell, *metric_cells = cells
        try:
            epoch = int(_clean_cell(epoch_cell))
            values = [_parse_number(cell) for cell in metric_cells]
        except ValueError:
            break
        row = {"epoch": float(epoch)}
        row.update(dict(zip(columns, values, strict=True)))
        rows.append(row)
    return rows


def load_checkpoint_eval_metrics(path: Path) -> dict[str, float] | None:
    """Return the ``eval_metrics`` dict embedded in a saved checkpoint, if any."""
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("eval_metrics")
    if not isinstance(metrics, dict):
        return None
    return {key: float(value) for key, value in metrics.items()}


def load_checkpoint_config(path: Path) -> dict[str, Any] | None:
    """Return non-tensor checkpoint fields (training objective, best epoch, ...)."""
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    return {key: value for key, value in payload.items() if key not in ("model", "config")}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_validation_split(payload: dict[str, Any], *, source: str) -> None:
    """Refuse to plot a diagnostics report unless it is the validation split.

    The locked common-oracle test split may only be consumed once; this
    report must never present it as already-evaluated evidence.
    """
    split = payload.get("split")
    if split != VALIDATION_SPLIT:
        raise ValueError(
            f"{source} has split={split!r}, expected {VALIDATION_SPLIT!r}; "
            "refusing to plot a non-validation (possibly locked test) split"
        )


VALUE_STAGE_HEADER = "| epoch | MSE ↓ | MAE ↓ | Pearson ↑ | sign accuracy ↑ | R² ↑ |"
VALUE_STAGE_COLUMNS = ["mse", "mae", "pearson", "sign_accuracy", "r2"]

JOINT_STAGE_HEADER = (
    "| epoch | policy CE ↓ | teacher top-1 ↑ | teacher coverage ↑ | value MSE ↓ | "
    "value sign ↑ | value R² ↑ |"
)
JOINT_STAGE_COLUMNS = [
    "policy_cross_entropy",
    "policy_top1_accuracy",
    "policy_teacher_coverage",
    "value_mse",
    "value_sign_accuracy",
    "value_r2",
]


def collect_value_stage_epochs(decisions_text: str) -> list[dict[str, float]]:
    return parse_markdown_epoch_table(
        decisions_text, header_marker=VALUE_STAGE_HEADER, columns=VALUE_STAGE_COLUMNS
    )


def collect_joint_stage_epochs(decisions_text: str) -> list[dict[str, float]]:
    return parse_markdown_epoch_table(
        decisions_text, header_marker=JOINT_STAGE_HEADER, columns=JOINT_STAGE_COLUMNS
    )


VALUE_REPAIR_EPOCH_KEYS = [
    "mse",
    "mae",
    "pearson",
    "sign_accuracy",
    "r2",
    "quiet_mae",
    "quiet_sign_accuracy",
    "edge_mae",
    "edge_sign_accuracy",
    "decisive_mae",
    "decisive_sign_accuracy",
    "won_mae",
    "won_sign_accuracy",
]


def collect_value_repair_epochs(
    run_dir: Path, *, stem: str = "value_repair_best", epochs: tuple[int, ...] = (1, 2, 3)
) -> list[dict[str, float]]:
    """Load real per-epoch eval_metrics from the saved epoch checkpoints.

    Unlike the value/joint stages, value-repair training was run with
    ``--save-every-epoch``, so each epoch's checkpoint carries its own
    programmatically computed ``eval_metrics`` — no markdown transcription
    needed.
    """
    rows: list[dict[str, float]] = []
    for epoch in epochs:
        path = run_dir / f"{stem}_epoch_{epoch}.pt"
        metrics = load_checkpoint_eval_metrics(path)
        if metrics is None:
            continue
        row: dict[str, float] = {"epoch": float(epoch)}
        for key in VALUE_REPAIR_EPOCH_KEYS:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return rows


def collect_oracle_bin_metrics(
    payload: dict[str, Any], *, checkpoint: str
) -> dict[str, dict[str, float]] | None:
    """Per-bin MAE/sign-accuracy for one checkpoint from a diagnostics report."""
    checkpoints = payload.get("checkpoints", {})
    entry = checkpoints.get(checkpoint)
    if entry is None:
        return None
    value_by_bin = entry.get("value_by_bin", {})
    return {
        bin_name: {
            "mae": float(bin_metrics["mae"]),
            "sign_accuracy": float(bin_metrics["sign_accuracy"]),
            "count": int(bin_metrics["count"]),
        }
        for bin_name, bin_metrics in value_by_bin.items()
    }


def collect_search_strategies(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Regret/near-best/best-move metrics keyed by ``checkpoint:sims:vscale``."""
    strategies = payload.get("strategies", {})
    return {
        key: {
            "count": float(entry["count"]),
            "mean_cp": float(entry["mean_cp"]),
            "p90_cp": float(entry["p90_cp"]),
            "p95_cp": float(entry["p95_cp"]),
            "near_best_accuracy": float(entry["near_best_accuracy"]),
            "best_move_accuracy": float(entry["best_move_accuracy"]),
        }
        for key, entry in strategies.items()
    }


_STRATEGY_KEY_RE = re.compile(
    r"^(?P<checkpoint>[^:]+):(?:raw|s(?P<sims>\d+):v(?P<scale>[\d.]+))$"
)


def parse_strategy_key(key: str) -> tuple[str, str, str]:
    """Split ``joint:s64:v0.5`` into (checkpoint, sims_label, value_scale_label).

    Raw (non-searched) strategies are keyed ``checkpoint:raw`` and are
    reported as sims="raw", scale="n/a".
    """
    match = _STRATEGY_KEY_RE.match(key)
    if not match:
        raise ValueError(f"unrecognized strategy key: {key!r}")
    checkpoint = match.group("checkpoint")
    sims = match.group("sims")
    scale = match.group("scale")
    if sims is None:
        return checkpoint, "raw", "n/a"
    return checkpoint, f"s{sims}", f"v{scale}"


def collect_match_result(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    return {
        "checkpoint": payload["checkpoint"],
        "games": int(payload["games"]),
        "simulations": int(payload["simulations"]),
        "stockfish_elo": int(payload["stockfish_elo"]),
        "wins": int(payload["wins"]),
        "draws": int(payload["draws"]),
        "losses": int(payload["losses"]),
        "score": float(payload["score"]),
    }
