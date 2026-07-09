#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_FILE="${DATASET_FILE:-runs/regret_start/targeted_selfplay.jsonl}"
DATASET_CARD="${DATASET_CARD:-runs/regret_start/README.md}"
HF_REPO="${HF_REPO:-}"
WAIT_FOR_GEN="${WAIT_FOR_GEN:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ ! -f .env ]]; then
  echo "error: .env is required for HF_TOKEN/HF_USERNAME" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
: "${HF_TOKEN:?error: HF_TOKEN is missing from .env}"
: "${HF_USERNAME:?error: HF_USERNAME is missing from .env}"

if [[ -z "$HF_REPO" ]]; then
  HF_REPO="${HF_USERNAME}/kibitzer-regret-start-az"
fi

while pgrep -f 'train_regret_start_az.py gen' >/dev/null; do
  if [[ "$WAIT_FOR_GEN" != "1" ]]; then
    echo "error: regret-start generation is still running; set WAIT_FOR_GEN=1 to wait" >&2
    exit 1
  fi
  lines="$(wc -l < "$DATASET_FILE" 2>/dev/null || printf '0')"
  echo "generation still running; $DATASET_FILE has $lines rows; sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

if [[ ! -s "$DATASET_FILE" ]]; then
  echo "error: dataset file is missing or empty: $DATASET_FILE" >&2
  exit 1
fi

rows="$(wc -l < "$DATASET_FILE")"
bytes="$(wc -c < "$DATASET_FILE")"
mkdir -p "$(dirname "$DATASET_CARD")"
cat > "$DATASET_CARD" <<EOF
---
license: mit
task_categories:
- reinforcement-learning
- text-generation
language:
- en
tags:
- chess
- alphazero
- self-play
- kibitzer
---

# Kibitzer Regret-Start Mini Self-Play

Targeted AlphaZero-style continuation data generated from high-regret Kibitzer
positions.

- source checkpoint: \`runs/regret/policy_regret_repair.pt\`
- start buffer: \`runs/regret/az12_policy_regret_sf12.jsonl\`
- generation defaults: 1,000 starts, 128 PUCT sims, 32 plies
- value target: root search value
- rows at upload: ${rows}
- bytes at upload: ${bytes}

Each JSONL row contains a FEN, MCTS visit counts, root value, selected value
target, source start FEN, and start-regret metadata.
EOF

echo "============================================================"
echo " KIBITZER REGRET-START DATASET UPLOAD"
echo "============================================================"
echo "Repo:        $HF_REPO"
echo "Dataset:     $DATASET_FILE"
echo "Rows:        $rows"
echo "Bytes:       $bytes"
echo "Card:        $DATASET_CARD"
echo

hf upload "$HF_REPO" "$DATASET_FILE" "$(basename "$DATASET_FILE")" \
  --repo-type dataset \
  --private \
  --token "$HF_TOKEN" \
  --commit-message "Upload regret-start self-play dataset"

hf upload "$HF_REPO" "$DATASET_CARD" README.md \
  --repo-type dataset \
  --private \
  --token "$HF_TOKEN" \
  --commit-message "Add dataset card"

echo "HF_DATASET_UPLOAD_DONE https://huggingface.co/datasets/$HF_REPO"
