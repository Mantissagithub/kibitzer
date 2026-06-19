# AGENTS.md

Lightweight context for coding agents working in this repo.

## Project Overview

- Python 3.11 chess policy/value model named `kibitzer`.
- Learns from chess game histories and predicts AlphaZero-style moves plus a bounded value head.
- Includes supervised training, evaluation against Stockfish/cutechess, RL fine-tuning, AWR/PPO paths, and AlphaZero-style search/training.
- Uses Hugging Face for checkpoints/artifacts when the relevant environment variables are set.
- Heavy artifacts such as checkpoints, runs, PGNs, and local env files are intentionally gitignored.

## Important Directories

- `kibitzer/`: package code for model, encoding, inference, evaluation, search, RL, distillation, and utilities.
- `kibitzer/cenv/`: C-backed/vectorized chess environment pieces.
- `scripts/`: training, evaluation, Hugging Face, play, UCI, and plotting entrypoints.
- `tests/`: pytest suite for model/data/encoding/loss/eval/RL/search behavior.
- `docs/`: rendered architecture and result images used by the README.
- `checkpoints/`, `runs/`, `runs_rl/`, `eval_pgns*/`: generated local artifacts; do not treat as source.
- `resources/`: local resource data; contents may be large or generated, inspect before relying on it.

## Setup

- Python: `>=3.11` from `pyproject.toml`.
- Main dependency manager in repo docs: `uv`.
- Install/sync command: not confirmed.
- Hugging Face operations use `HF_USERNAME` and `HF_TOKEN` from the environment or local `.env`.

## Confirmed Commands

From `README.md`:

```bash
uv run python scripts/train.py
uv run python scripts/eval_and_rename_hf.py --from-hf
uv run pytest
uv run python scripts/train_rl.py --algorithm awr --init-checkpoint runs/best.pt
uv run python scripts/train_az.py --opponent stockfish --stockfish-levels 1320,1500,1800 --sims 32 --material-weight 0.85 --context-window 128 --batch-size 8 --grad-accum-steps 4 --hf-push true
uv run python scripts/train_ppo.py --total-timesteps 200000 --num-envs 8
uv run python scripts/search_vs_raw.py --sims 64 --material 0.85
uv run python scripts/eval_vs_stockfish.py --elo 1320 --sims 64
uv run tensorboard --logdir runs_rl/tb
```

## Tests, Lint, Typecheck

- Tests: `uv run pytest`
- Focused tests: `uv run pytest tests/test_name.py` or `uv run pytest tests/test_name.py::test_case`
- Lint command: not confirmed.
- Typecheck command: not confirmed.

## Coding Conventions

- Use Python type hints and `from __future__ import annotations`.
- Keep package imports absolute, e.g. `from kibitzer.model import Kibitzer`.
- Prefer dataclasses for compact configuration objects.
- Tests are plain pytest functions with direct tensor shape/value assertions.
- Existing comments are sparse and explain non-obvious behavior or accepted bounds.
- Match the existing lower-case, concise prose style in README-style docs.

## Agent Rules

- Inspect existing patterns before editing; read nearby code and tests first.
- Prefer small patches that directly serve the requested change.
- Do not change public APIs, checkpoints, artifact layout, or training semantics unless asked.
- Do not modify generated/heavy artifacts unless the task explicitly requires it.
- Run the smallest relevant test after code changes.
- For docs-only changes, inspect the resulting files and skip expensive training/test runs unless requested.
- Explain clearly if a command cannot be run, is not confirmed, or would need external tools/secrets.
