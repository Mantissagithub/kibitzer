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


def _opponent_rating(opponent: str | None) -> int | None:
    if opponent is None or not opponent.startswith("stockfish-elo-"):
        return None
    return int(opponent.rsplit("-", 1)[1])


def _estimated_rating(elo: float | None, opponent: str | None) -> int | None:
    opponent_rating = _opponent_rating(opponent)
    if opponent_rating is None or elo is None or not math.isfinite(elo):
        return None
    return int(round(opponent_rating + elo))


def _elo_label(elo: float | None, opponent: str | None) -> str:
    if elo is None or not math.isfinite(elo):
        return "pending/unrated"
    rating = _estimated_rating(elo, opponent)
    if rating is not None:
        return f"{rating} estimated vs {opponent} ({elo:+.1f} Elo diff)"
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
    opponent: str | None = None,
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
    eval_opponent = opponent or post_eval.get("opponent")

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
        f"- `elo_rating`: {_elo_label(elo, eval_opponent)}",
        f"- `github_repo`: {GITHUB_REPO}",
    ]
    if eval_opponent is not None:
        lines.append(f"- `eval_opponent`: `{eval_opponent}`")
    if elo is not None and math.isfinite(elo):
        lines.append(f"- `elo_diff`: {_fmt_float(elo, 1)}")
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
        f"- Input: 64 chess-square piece tokens plus 7 auxiliary scalars",
        "  covering side to move, castling rights, en-passant file, and",
        "  halfmove clock.",
        f"- Position encoder: bidirectional square-level transformer that",
        "  compresses each board into one timeline token.",
        f"- Timeline trunk: causal transformer with `d_model={d_model}`,",
        f"  sequence length `{max_seq_len}`, 12 layers, and 8 attention heads.",
        "- Transformer blocks: pre-norm RMSNorm, RoPE on causal Q/K attention,",
        "  PyTorch scaled dot-product attention, and SwiGLU feed-forward layers.",
        "- Outputs: policy head over 4,672 AlphaZero-style moves (`64 * 73`)",
        "  plus a tanh value head for bounded outcome prediction.",
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
