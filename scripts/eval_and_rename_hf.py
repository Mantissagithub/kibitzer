"""evaluate checkpoints and rename their hugging face repos with elo.

use this after remote training has pushed checkpoint repos named like:
``pradheep1647/kibitzer-sft-elo-pending-step-002000``.

run it on a machine with ``stockfish`` and ``cutechess-cli`` installed. the
``--from-hf`` mode downloads one pending checkpoint at a time, evaluates it,
renames the hf repo, uploads post-eval metadata, then deletes the local copy.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from kibitzer.eval import evaluate_checkpoint
from scripts.hf_readme import render_hf_readme


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


def _step_from_repo(repo_id: str) -> int:
    marker = "-step-"
    if marker not in repo_id:
        raise ValueError(f"repo name has no step marker: {repo_id}")
    return int(repo_id.rsplit(marker, 1)[1])


def _elo_tag(elo: float) -> str:
    if not math.isfinite(elo):
        return "elo-unrated"
    sign = "plus" if elo >= 0 else "minus"
    return f"elo-{sign}-{abs(int(round(elo))):04d}"


def _opponent_rating(opponent: str) -> int | None:
    if not opponent.startswith("stockfish-elo-"):
        return None
    return int(opponent.rsplit("-", 1)[1])


def _estimated_rating(elo_diff: float, opponent: str) -> int | None:
    opponent_rating = _opponent_rating(opponent)
    if opponent_rating is None or not math.isfinite(elo_diff):
        return None
    return int(round(opponent_rating + elo_diff))


def _rating_tag(elo_diff: float, opponent: str) -> str:
    rating = _estimated_rating(elo_diff, opponent)
    if rating is not None:
        return f"elo-{rating:04d}"
    return _elo_tag(elo_diff)


def _log(message: str) -> None:
    tqdm.write(message)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint-dir", default=None,
                   help="local run dir containing step_*.pt checkpoints")
    p.add_argument("--from-hf", action="store_true",
                   help="stream pending checkpoint repos from HF one at a time")
    p.add_argument("--work-dir", default=".hf_eval_work",
                   help="temporary download dir used with --from-hf")
    p.add_argument("--keep-local", action="store_true",
                   help="do not delete downloaded checkpoint dirs after eval")
    p.add_argument("--limit", type=int, default=None,
                   help="optional max number of checkpoints to process")
    p.add_argument("--hf-username", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-repo-prefix", default="kibitzer-sft")
    p.add_argument("--opponent", default="stockfish-elo-1320",
                   choices=[
                       "stockfish-elo-1320",
                       "stockfish-elo-1500",
                       "stockfish-elo-1800",
                       "stockfish-full",
                   ])
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--time-per-move-ms", type=int, default=200)
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--cutechess-path", default="cutechess-cli")
    p.add_argument("--pgn-dir", default="eval_pgns",
                   help="directory for generated cutechess PGNs")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _pending_repo_ids(api, username: str, prefix: str, token: str) -> list[str]:
    needle = f"{prefix}-elo-pending-step-"
    repos = []
    for info in api.list_models(author=username, search=needle, token=token):
        repo_id = getattr(info, "id", None) or getattr(info, "modelId", None)
        if repo_id and repo_id.startswith(f"{username}/{needle}"):
            repos.append(repo_id)
    return sorted(repos, key=_step_from_repo)


def _download_one_checkpoint(api, repo_id: str, token: str, work_dir: Path) -> Path:
    local_dir = work_dir / repo_id.split("/", 1)[1]
    if local_dir.exists():
        shutil.rmtree(local_dir)
    api.snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=["*.pt", "training_metadata.yaml"],
        token=token,
    )
    ckpts = sorted(local_dir.glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no step_*.pt found after downloading {repo_id}")
    return ckpts[0]


def _evaluate_and_rename(
    *,
    api,
    ckpt: Path,
    old_repo: str,
    username: str,
    token: str,
    args: argparse.Namespace,
) -> None:
    step = _step_from_checkpoint(ckpt)
    _log(f"eval: {ckpt.name} vs {args.opponent} ({args.n_games} games)")
    completed_games = 0
    with tqdm(
        total=args.n_games,
        desc=f"games {ckpt.stem}",
        unit="game",
        leave=False,
    ) as game_bar:

        def on_progress(progress: dict) -> None:
            nonlocal completed_games
            completed = int(progress.get("completed", completed_games))
            if completed > completed_games:
                game_bar.update(completed - completed_games)
                completed_games = completed
            if "elo_diff" in progress:
                elo_diff = float(progress["elo_diff"])
                elo_err = float(progress.get("elo_err", math.nan))
                game_bar.set_postfix_str(f"elo={elo_diff:+.1f}±{elo_err:.1f}")

        result = evaluate_checkpoint(
            checkpoint_path=str(ckpt),
            opponent=args.opponent,
            n_games=args.n_games,
            time_per_move_ms=args.time_per_move_ms,
            stockfish_path=args.stockfish_path,
            cutechess_path=args.cutechess_path,
            pgn_dir=args.pgn_dir,
            on_progress=on_progress,
    )
    elo = float(result["elo_diff"])
    new_repo = f"{username}/{args.hf_repo_prefix}-{_rating_tag(elo, args.opponent)}-step-{step:06d}"
    _log(f"done: {ckpt.name} elo={elo:+.1f} {old_repo} -> {new_repo}")
    if args.dry_run:
        _log(f"dry-run: skipped HF rename/upload for {new_repo}")
        return

    _log(f"hf: renaming {old_repo} -> {new_repo}")
    api.move_repo(from_id=old_repo, to_id=new_repo, repo_type="model", token=token)
    metadata_path = ckpt.with_suffix(".post_eval.yaml")
    metadata_path.write_text(yaml.safe_dump(result, sort_keys=True))
    training_metadata_path = ckpt.with_name("training_metadata.yaml")
    training_metadata = {}
    if training_metadata_path.exists():
        training_metadata = yaml.safe_load(training_metadata_path.read_text()) or {}
    readme_path = ckpt.with_suffix(".README.md")
    readme_path.write_text(render_hf_readme(
        repo_id=new_repo,
        checkpoint_name=ckpt.name,
        step=step,
        config=training_metadata.get("config"),
        metrics=training_metadata.get("metrics"),
        elo=elo,
        opponent=args.opponent,
        post_eval=result,
    ))
    _log(f"hf: uploading post_eval.yaml and README.md to {new_repo}")
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="post_eval.yaml",
        repo_id=new_repo,
        repo_type="model",
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=new_repo,
        repo_type="model",
        token=token,
    )


def main() -> int:
    args = parse_args()
    env = _read_env(Path(".env"))
    username = args.hf_username or _env("HF_USERNAME", env)
    token = args.hf_token or _env("HF_TOKEN", env)
    if not username or not token:
        _log("error: HF username/token missing; set .env or pass args")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        _log("error: huggingface_hub is not installed")
        return 1

    api = HfApi(token=token)
    mode = "from-hf" if args.from_hf else "local"
    suffix = " dry-run" if args.dry_run else ""
    _log(
        f"kibitzer HF eval: mode={mode}{suffix} user={username} "
        f"opponent={args.opponent} games={args.n_games}"
    )

    if args.from_hf:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _log(f"hf: listing pending {username}/{args.hf_repo_prefix} repos")
        repo_ids = _pending_repo_ids(api, username, args.hf_repo_prefix, token)
        if args.limit is not None:
            repo_ids = repo_ids[:args.limit]
        if not repo_ids:
            _log("error: no pending HF checkpoint repos found")
            return 1
        for repo_id in tqdm(repo_ids, desc="HF repos", unit="repo"):
            local_root = work_dir / repo_id.split("/", 1)[1]
            try:
                _log(f"hf: downloading {repo_id}")
                ckpt = _download_one_checkpoint(api, repo_id, token, work_dir)
                _evaluate_and_rename(
                    api=api,
                    ckpt=ckpt,
                    old_repo=repo_id,
                    username=username,
                    token=token,
                    args=args,
                )
            finally:
                if not args.keep_local and local_root.exists():
                    shutil.rmtree(local_root)
                    _log(f"clean: removed {local_root}")
        return 0

    if not args.checkpoint_dir:
        _log("error: pass --checkpoint-dir or use --from-hf")
        return 1
    ckpts = sorted(Path(args.checkpoint_dir).glob("step_*.pt"))
    if args.limit is not None:
        ckpts = ckpts[:args.limit]
    if not ckpts:
        _log(f"error: no step_*.pt checkpoints under {args.checkpoint_dir}")
        return 1
    for ckpt in tqdm(ckpts, desc="local checkpoints", unit="ckpt"):
        step = _step_from_checkpoint(ckpt)
        old_repo = f"{username}/{args.hf_repo_prefix}-elo-pending-step-{step:06d}"
        _evaluate_and_rename(
            api=api,
            ckpt=ckpt,
            old_repo=old_repo,
            username=username,
            token=token,
            args=args,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
