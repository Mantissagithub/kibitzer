"""bulk-delete pending/plus/minus checkpoint repos from hugging face.

run after the forward eval has produced rated-elo repos and you no longer
need the original pending/plus/minus-named repos. lists matching repos,
prints them, and deletes them after a confirmation prompt (or ``--yes``).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


_LEGACY_REPO_RE = re.compile(
    r"^[^/]+/[^/]+-elo-(pending|plus-\d{4}|minus-\d{4})-step-\d{6}$"
)


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


def _step_from_repo(repo_id: str) -> int:
    return int(repo_id.rsplit("-step-", 1)[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hf-username", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-repo-prefix", default="kibitzer-sft")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.add_argument("--dry-run", action="store_true", help="list only, do not delete")
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
    repos: list[str] = []
    for info in api.list_models(author=username, search=needle, token=token):
        rid = getattr(info, "id", None) or getattr(info, "modelId", None)
        if (
            rid
            and rid.startswith(f"{username}/{needle}")
            and _LEGACY_REPO_RE.match(rid)
        ):
            repos.append(rid)
    repos.sort(key=_step_from_repo)

    if not repos:
        print("no pending/plus/minus repos to delete")
        return 0

    print(f"found {len(repos)} legacy repos under {username}/{args.hf_repo_prefix}:")
    for rid in repos:
        print(f"  {rid}")

    if args.dry_run:
        print("dry-run: nothing deleted")
        return 0

    if not args.yes:
        ans = input(f"\ndelete all {len(repos)} repos above? [y/N] ").strip().lower()
        if ans != "y":
            print("aborted")
            return 1

    for rid in repos:
        print(f"deleting {rid}")
        api.delete_repo(repo_id=rid, repo_type="model", token=token, missing_ok=True)
    print(f"done: deleted {len(repos)} repos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
