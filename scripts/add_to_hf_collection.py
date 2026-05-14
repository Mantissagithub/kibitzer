"""add all kibitzer hugging face model repos to a named collection.

finds the user's collection by title (default ``Kibitzer``) and adds every
model repo under ``{username}/{prefix}-*`` to it. idempotent: re-running is
safe (``exists_ok=True``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hf-username", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-repo-prefix", default="kibitzer",
                   help="match repos named {username}/{prefix}-* (default: kibitzer)")
    p.add_argument("--collection-title", default="Kibitzer")
    p.add_argument("--dry-run", action="store_true")
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

    target = args.collection_title.strip().lower()
    collection = None
    for c in api.list_collections(owner=username, token=token):
        if (c.title or "").strip().lower() == target:
            collection = c
            break
    if collection is None:
        print(f"error: no collection titled {args.collection_title!r} under {username}")
        return 1
    print(f"collection: {collection.title} ({collection.slug})")

    needle = f"{args.hf_repo_prefix}-"
    repos: list[str] = []
    for info in api.list_models(author=username, search=needle, token=token):
        rid = getattr(info, "id", None) or getattr(info, "modelId", None)
        if rid and rid.startswith(f"{username}/{needle}"):
            repos.append(rid)
    repos.sort()
    if not repos:
        print(f"no repos match {username}/{needle}*")
        return 0
    print(f"found {len(repos)} repos to add")

    for rid in repos:
        if args.dry_run:
            print(f"dry-run: would add {rid}")
            continue
        print(f"adding {rid}")
        api.add_collection_item(
            collection_slug=collection.slug,
            item_id=rid,
            item_type="model",
            exists_ok=True,
            token=token,
        )
    print(f"done: {len(repos)} repos {'previewed' if args.dry_run else 'added'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
