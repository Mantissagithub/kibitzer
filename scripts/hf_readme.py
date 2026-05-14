"""Hugging Face model card rendering for kibitzer checkpoints."""

from __future__ import annotations

import math
from typing import Any

import yaml


GITHUB_REPO = "https://github.com/Mantissagithub/kibitzer"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _elo_label(elo: float | None) -> str:
    if elo is None or not math.isfinite(elo):
        return "pending/unrated"
    return f"{elo:+.1f} vs configured Stockfish baseline"


def _front_matter(repo_id: str) -> str:
    data = {
        "library_name": "pytorch",
        "license": "mit",
        "tags": ["chess", "transformer", "policy-value", "kibitzer"],
        "model_name": repo_id,
    }
    return "---\n" + yaml.safe_dump(data, sort_keys=False).strip() + "\n---"


def render_hf_readme(
    *,
    repo_id: str,
    checkpoint_name: str,
    step: int,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    elo: float | None = None,
    post_eval: dict[str, Any] | None = None,
) -> str:
    """Render the README.md uploaded to each HF checkpoint repo."""
    config = config or {}
    metrics = metrics or {}
    post_eval = post_eval or {}

    d_model = config.get("d_model", 384)
    max_seq_len = config.get("max_seq_len", 256)
    eval_games = post_eval.get("n_games") or metrics.get("eval_n_games")
    eval_score = post_eval.get("score") or metrics.get("eval_score")
    eval_err = post_eval.get("elo_err") or metrics.get("eval_elo_err")

    lines = [
        _front_matter(repo_id),
        "",
        f"# {repo_id}",
        "",
        "Kibitzer checkpoint for a chess policy/value model trained on game",
        "sequences. Each board is encoded as 64 square tokens plus auxiliary",
        "state, summarized by a square-level encoder, then processed by a",
        "causal transformer over the game timeline.",
        "",
        "## Status",
        "",
        f"- `checkpoint`: `{checkpoint_name}`",
        f"- `step`: `{step:06d}`",
        f"- `elo_rating`: {_elo_label(elo)}",
        f"- `github_repo`: {GITHUB_REPO}",
    ]
    if eval_games is not None:
        lines.append(f"- `eval_games`: {eval_games}")
    if eval_score is not None:
        lines.append(f"- `eval_score`: {_fmt_float(eval_score, 2)}")
    if eval_err is not None:
        lines.append(f"- `elo_error`: {_fmt_float(eval_err, 1)}")

    lines.extend([
        "",
        "## Architecture",
        "",
        f"- Position encoder over 64 board squares and 7 auxiliary features.",
        f"- Causal transformer trunk with `d_model={d_model}` and sequence",
        f"  length `{max_seq_len}`.",
        "- RMSNorm, RoPE attention, SwiGLU MLPs, and PyTorch scaled",
        "  dot-product attention.",
        "- Policy head over 4,672 AlphaZero-style moves (`64 * 73`).",
        "- Tanh value head for bounded outcome prediction.",
        "",
        "## Files",
        "",
        f"- `{checkpoint_name}`: PyTorch checkpoint.",
        "- `training_metadata.yaml`: training config and checkpoint metrics.",
        "- `post_eval.yaml`: uploaded after local Stockfish/cutechess eval.",
        "",
        "## Usage",
        "",
        "```bash",
        "git clone https://github.com/Mantissagithub/kibitzer.git",
        "cd kibitzer",
        f"uv run python scripts/uci.py --checkpoint <path-to>/{checkpoint_name}",
        "```",
        "",
        "This checkpoint is one artifact in a sequence. Repos with",
        "`elo-pending` in the name have not been evaluated yet; rated repos are",
        "renamed after `scripts/eval_and_rename_hf.py --from-hf` completes.",
        "",
    ])
    return "\n".join(lines)
