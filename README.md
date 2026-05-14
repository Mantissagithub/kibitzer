# kibitzer

Chess model that treats board states as tokens, learns from game sequences, and
plays through policy/value predictions.

- GitHub: https://github.com/Mantissagithub/kibitzer
- Hugging Face: https://huggingface.co/Pradheep1647
- `elo_rating`: pending/unrated
- Current checkpoint prefix: `Pradheep1647/kibitzer-sft`
- Current checkpoint status: pending HF eval/rename through step `050000`

## Architecture

Kibitzer is built as a reusable chess policy/value stack:

- `PositionEncoder` embeds each board state from 64 square tokens plus 7
  auxiliary features for side to move, castling rights, en-passant file, and
  halfmove clock.
- A bidirectional square-level encoder summarizes each board into one timeline
  token.
- A causal Llama-style transformer trunk models the game sequence with RMSNorm,
  RoPE attention, SwiGLU MLPs, and PyTorch scaled dot-product attention.
- The policy head predicts one of 4,672 AlphaZero-style moves (`64 * 73`).
- The value head predicts a bounded scalar outcome estimate with `tanh`.

## Run

Train locally and push checkpoints to Hugging Face:

```bash
uv run python scripts/train.py
```

Evaluate pending Hugging Face checkpoints, rename repos with ELO, and upload
post-eval metadata:

```bash
uv run python scripts/eval_and_rename_hf.py --from-hf
```

Run tests:

```bash
uv run pytest
```

## Notes

- Credentials are read from `HF_USERNAME` and `HF_TOKEN`, either from the
  environment or an ignored local `.env` file.
- Training checkpoints are saved under `runs/` and pushed as individual HF model
  repos named like `kibitzer-sft-elo-pending-step-002000`.
- ELO is intentionally listed as pending until checkpoints are evaluated against
  Stockfish via `cutechess-cli`.
