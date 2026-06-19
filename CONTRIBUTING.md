# CONTRIBUTING.md

Use this repo with small, reviewable changes. This is especially important for
PRs prepared with coding agents, where reviewers need a clear trail from request
to diff to verification.

## PR Expectations

- Keep each PR focused on one task or closely related set of changes.
- Explain why the change is needed, not just what files changed.
- Call out any behavior, API, checkpoint, artifact, or dependency impact.
- Do not mix generated artifacts with source changes unless the PR is explicitly
  about those artifacts.
- Do not include local secrets, `.env` values, private machine paths, or agent
  scratch files.

## Agent-Friendly Workflow

1. Inspect existing code, tests, and docs before editing.
2. Make the smallest patch that solves the requested problem.
3. Preserve public APIs and training behavior unless the task asks to change them.
4. Run the smallest relevant test or explain why it was not run.
5. In the PR description, include a concise justification and verification notes.

## Suggested PR Description

```md
## Why

Brief justification for the change.

## What Changed

- Main change 1
- Main change 2

## Verification

- `uv run pytest tests/...`
- Not run: reason
```

## Commands

- Tests: `uv run pytest`
- Focused tests: `uv run pytest tests/test_name.py` or
  `uv run pytest tests/test_name.py::test_case`
- Lint/typecheck: not confirmed.
