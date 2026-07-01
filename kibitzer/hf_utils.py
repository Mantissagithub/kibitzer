from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def validate_hf_push(*, enabled: bool, repo_id: str | None) -> None:
    if not enabled:
        return
    if not repo_id or "/" not in repo_id:
        raise SystemExit("--hf-push true requires --hf-repo USER_OR_ORG/REPO")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("--hf-push true requires HF_TOKEN in the environment")


def push_checkpoint_to_hf(
    checkpoint_path: Path,
    *,
    repo_id: str,
    training_objective: str,
    metadata: dict[str, Any],
) -> None:
    token = os.environ["HF_TOKEN"]
    metadata_path = checkpoint_path.with_suffix(".yaml")
    readme_path = checkpoint_path.with_suffix(".README.md")
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "model": "kibitzer",
                "architecture": "clean-rebuild-transformer-ssm",
                "training_objective": training_objective,
                "checkpoint": checkpoint_path.name,
                **metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            [
                "---",
                "library_name: pytorch",
                "license: mit",
                "tags:",
                "- chess",
                "- policy-value",
                "- transformer",
                "- state-space-model",
                "---",
                "",
                f"# {repo_id}",
                "",
                "Kibitzer clean-rebuild checkpoint.",
                "",
                f"- Training objective: `{training_objective}`",
                f"- Checkpoint: `{checkpoint_path.name}`",
                "- Architecture: 32.1M-parameter position-only transformer/SSM hybrid",
                "- Policy output: 4,672 AlphaZero-style move logits",
                "- Value output: side-to-move scalar in `[-1, 1]`",
                "- Source: https://github.com/Mantissagithub/kibitzer/tree/clean-rebuild",
                "",
            ]
        ),
        encoding="utf-8",
    )

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="training_metadata.yaml",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
