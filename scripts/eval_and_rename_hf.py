"""Evaluate local checkpoints and rename their Hugging Face repos with Elo.

Use this after remote training has pushed checkpoint repos named like:
``Pradheep1647/kibitzer-sft-elo-pending-step-002000``.

Run it on a machine with ``stockfish`` and ``cutechess-cli`` installed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import yaml

from kibitzer.eval import evaluate_checkpoint
from kibitzer import tui


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


def _step_from_checkpoint(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"checkpoint name must look like step_002000.pt: {path}")


def _elo_tag(elo: float) -> str:
    if not math.isfinite(elo):
        return "elo-unrated"
    sign = "plus" if elo >= 0 else "minus"
    return f"elo-{sign}-{abs(int(round(elo))):04d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--hf-username", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-repo-prefix", default="kibitzer-sft")
    p.add_argument("--opponent", default="stockfish-0",
                   choices=["stockfish-0", "stockfish-3", "stockfish-5", "stockfish-10"])
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--time-per-move-ms", type=int, default=200)
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--cutechess-path", default="cutechess-cli")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env = _read_env(Path(".env"))
    username = args.hf_username or _env("HF_USERNAME", env)
    token = args.hf_token or _env("HF_TOKEN", env)
    if not username or not token:
        tui.console.print("[error]HF username/token missing; set .env or pass args[/]")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        tui.console.print("[error]huggingface_hub is not installed[/]")
        return 1

    ckpts = sorted(Path(args.checkpoint_dir).glob("step_*.pt"))
    if not ckpts:
        tui.console.print(f"[error]no step_*.pt checkpoints under {args.checkpoint_dir}[/]")
        return 1

    api = HfApi(token=token)
    for ckpt in ckpts:
        step = _step_from_checkpoint(ckpt)
        old_repo = f"{username}/{args.hf_repo_prefix}-elo-pending-step-{step:06d}"
        tui.console.print(f"[muted]evaluating {ckpt.name}[/]")
        result = evaluate_checkpoint(
            checkpoint_path=str(ckpt),
            opponent=args.opponent,
            n_games=args.n_games,
            time_per_move_ms=args.time_per_move_ms,
            stockfish_path=args.stockfish_path,
            cutechess_path=args.cutechess_path,
        )
        elo = float(result["elo_diff"])
        new_repo = f"{username}/{args.hf_repo_prefix}-{_elo_tag(elo)}-step-{step:06d}"
        tui.console.print(
            f"[success]{ckpt.name}[/] elo={elo:+.1f} "
            f"[muted]{old_repo} -> {new_repo}[/]"
        )
        if args.dry_run:
            continue

        api.move_repo(from_id=old_repo, to_id=new_repo, repo_type="model", token=token)
        metadata_path = ckpt.with_suffix(".post_eval.yaml")
        metadata_path.write_text(yaml.safe_dump(result, sort_keys=True))
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="post_eval.yaml",
            repo_id=new_repo,
            repo_type="model",
            token=token,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
