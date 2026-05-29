"""hf push stuff extracted from train.py so rl can use it too."""

from __future__ import annotations

import getpass
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

GITHUB_REPO = "https://github.com/Mantissagithub/kibitzer"


@dataclass
class HFPushConfig:
    username: str
    token: str
    repo_prefix: str
    private: bool


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _write_env_values(path: Path, updates: dict[str, str]) -> None:
    existing = _read_env_file(path)
    existing.update({k: v for k, v in updates.items() if v})
    lines = [f"{k}={v}" for k, v in sorted(existing.items())]
    path.write_text("\n".join(lines) + "\n")


def _env_value(file_env: dict[str, str], key: str) -> str:
    return os.environ.get(key) or file_env.get(key, "")


def prepare_hf_push(cfg: Any, use_tui: bool = False) -> HFPushConfig | None:
    # reads .env, returns config or none if disabled/missing creds
    if not cfg.hf_push:
        return None

    env_path = Path(".env")
    file_env = _read_env_file(env_path)
    username = _env_value(file_env, "HF_USERNAME")
    token = _env_value(file_env, "HF_TOKEN")

    if use_tui and (not username or not token):
        print("[header]Hugging Face checkpoint push[/]")
        print("[muted]credentials are saved to .env for this project[/]")
        if not username:
            username = input("HF username: ").strip()
        if not token:
            token = getpass.getpass("HF token: ").strip()
        _write_env_values(env_path, {"HF_USERNAME": username, "HF_TOKEN": token})
        try:
            env_path.chmod(0o600)
        except OSError:
            pass

    if not username or not token:
        print("[warning]HF push enabled but HF_USERNAME/HF_TOKEN are missing; checkpoint uploads disabled[/]")
        return None

    return HFPushConfig(
        username=username,
        token=token,
        repo_prefix=cfg.hf_repo_prefix,
        private=cfg.hf_private,
    )


def _elo_tag(elo: float | None) -> str:
    # "elo-pending" or "elo-plus-1234"
    if elo is None or not math.isfinite(elo):
        return "elo-pending"
    sign = "plus" if elo >= 0 else "minus"
    return f"elo-{sign}-{abs(int(round(elo))):04d}"


def hf_repo_id(hf: HFPushConfig, step: int, elo: float | None) -> str:
    # e.g., Pradheep1647/kibitzer-rl-elo-pending-step-000020
    return f"{hf.username}/{hf.repo_prefix}-{_elo_tag(elo)}-step-{step:06d}"


# ── readme rendering ────────────────────────────────────────────


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
        return f"{rating} estimated vs {opponent} ({elo:+.1f} elo diff)"
    return f"{elo:+.1f} vs configured stockfish baseline"


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
    # generates the README.md that goes up to hf with each checkpoint
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
        "kibitzer checkpoint for a chess policy/value model trained on game",
        "sequences. each board is encoded as 64 square tokens plus auxiliary",
        "state, summarized by a square-level encoder, then processed by a",
        "causal transformer over the game timeline.",
        "",
        "## status",
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
        "## architecture",
        "",
        "- input: 64 chess-square piece tokens plus 7 auxiliary scalars",
        "  covering side to move, castling rights, en-passant file, and",
        "  halfmove clock.",
        "- position encoder: bidirectional square-level transformer that",
        "  compresses each board into one timeline token.",
        "- timeline trunk: causal transformer with",
        f"  `d_model={d_model}`,",
        f"  sequence length `{max_seq_len}`, 12 layers, and 8 attention heads.",
        "- transformer blocks: pre-norm rmsnorm, rope on causal q/k attention,",
        "  pytorch scaled dot-product attention, and swiglu feed-forward layers.",
        "- outputs: policy head over 4,672 alphazero-style moves (`64 * 73`)",
        "  plus a tanh value head for bounded outcome prediction.",
        "",
        "## files",
        "",
        f"- `{checkpoint_name}`: pytorch checkpoint.",
        "- `training_metadata.yaml`: training config and checkpoint metrics.",
        "- `post_eval.yaml`: uploaded after local stockfish/cutechess eval.",
        "",
        "## usage",
        "",
        "```bash",
        "git clone https://github.com/Mantissagithub/kibitzer.git",
        "cd kibitzer",
        f"uv run python scripts/uci.py --checkpoint <path-to>/{checkpoint_name}",
        "```",
        "",
        "this checkpoint is one artifact in a sequence. repos with",
        "`elo-pending` in the name have not been evaluated yet; rated repos are",
        "renamed after `scripts/eval_and_rename_hf.py --from-hf` completes.",
        "",
    ])
    return "\n".join(lines)


def push_checkpoint_to_hf(
    hf: HFPushConfig | None,
    ckpt_path: Path,
    step: int,
    cfg: Any,
    metrics: dict[str, Any],
    elo: float | None = None,
    *,
    training_stage: str = "sft",
    eval_opponent: str | None = None,
) -> str | None:
    # uploads checkpoint + metadata + readme to hf, returns repo id or error
    if hf is None:
        return None
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return "huggingface_hub is not installed"

    repo_id = hf_repo_id(hf, step, elo)
    metadata_path = ckpt_path.with_suffix(".yaml")
    metadata = {
        "model": "kibitzer",
        "training_stage": training_stage,
        "step": step,
        "elo_diff": elo,
        "checkpoint": ckpt_path.name,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else {},
        "metrics": metrics,
    }
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=True))
    readme_path = ckpt_path.with_suffix(".README.md")
    readme_path.write_text(render_hf_readme(
        repo_id=repo_id,
        checkpoint_name=ckpt_path.name,
        step=step,
        config=metadata["config"],
        metrics=metrics,
        elo=elo,
        opponent=eval_opponent,
    ))
    try:
        api = HfApi(token=hf.token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=hf.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=ckpt_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=hf.token,
        )
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="training_metadata.yaml",
            repo_id=repo_id,
            repo_type="model",
            token=hf.token,
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=hf.token,
        )
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return repo_id