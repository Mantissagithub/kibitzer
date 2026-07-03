#!/usr/bin/env bash
set -euo pipefail

# From-scratch JOINT policy+value training.
#
# Unlike train_policy_and_value.sh (policy first, then a frozen-trunk value head),
# this trains the whole model from random init with the policy AND value losses
# together, so the shared trunk is shaped by value gradients from the start.
# Policy target: elite human moves. Value target: side-to-move game result.
# The joint value weight is fixed at 0.25 inside train_bc.py (no flag for it).

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "error: .env is required; copy .env.example and fill HF_USERNAME/HF_TOKEN" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${HF_USERNAME:?error: HF_USERNAME is missing from .env}"
: "${HF_TOKEN:?error: HF_TOKEN is missing from .env}"

if [[ ! -x .venv/bin/python ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: install uv or create .venv with the project dependencies" >&2
    exit 1
  fi
  uv sync
fi
PYTHON="$ROOT_DIR/.venv/bin/python"

PGN_PATHS=()
if [[ $# -gt 0 ]]; then
  PGN_PATHS=("$@")
else
  DATA_YEAR="${DATA_YEAR:-2025}"
  DATA_MONTHS="${DATA_MONTHS:-06,07,08,09,10,11}"
  MIN_FREE_GB="${MIN_FREE_GB:-8}"
  FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 {print $4}')"
  if (( FREE_KB < MIN_FREE_GB * 1024 * 1024 )); then
    echo "error: automatic dataset download requires at least ${MIN_FREE_GB}GB free" >&2
    exit 1
  fi
  echo "stage 0/1: download/cache Lichess Elite ${DATA_YEAR} months ${DATA_MONTHS}"
  "$PYTHON" scripts/download_lichess_elite.py \
    --year "$DATA_YEAR" \
    --months "$DATA_MONTHS" \
    --output-dir data/raw
  IFS=',' read -r -a REQUESTED_MONTHS <<< "$DATA_MONTHS"
  for month in "${REQUESTED_MONTHS[@]}"; do
    printf -v normalized_month '%02d' "$((10#$month))"
    PGN_PATHS+=("data/raw/lichess_elite_${DATA_YEAR}-${normalized_month}.pgn")
  done
fi

if [[ ${#PGN_PATHS[@]} -eq 0 ]]; then
  echo "error: no PGN files are available" >&2
  exit 1
fi
PGN_ARGS=()
for pgn_path in "${PGN_PATHS[@]}"; do
  if [[ ! -s "$pgn_path" ]]; then
    echo "error: PGN does not exist or is empty: $pgn_path" >&2
    exit 1
  fi
  PGN_ARGS+=(--pgn "$pgn_path")
done
echo "using ${#PGN_PATHS[@]} PGN source files"

"$PYTHON" - <<'PY'
import os

import torch
from huggingface_hub import HfApi

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is required for this launcher")

expected = os.environ["HF_USERNAME"]
actual = HfApi(token=os.environ["HF_TOKEN"]).whoami().get("name")
if actual != expected:
    raise SystemExit(f"error: HF token belongs to {actual!r}, expected {expected!r}")
print(f"prerequisites ok: cuda={torch.cuda.get_device_name(0)} hf_user={actual}")
PY

OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/joint_scratch/joint_scratch_final.pt}"
HF_REPO="${HF_REPO:-${HF_USERNAME}/kibitzer-clean-joint-scratch}"
MAX_POSITIONS="${MAX_POSITIONS:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HF_PUSH="${HF_PUSH:-true}"

mkdir -p "$(dirname "$OUTPUT_CHECKPOINT")"

echo "stage 1/1: from-scratch joint policy+value training"
echo "  positions/epoch: $MAX_POSITIONS"
echo "  epochs:          $EPOCHS"
echo "  output:          $OUTPUT_CHECKPOINT"
echo "  hf repo:         https://huggingface.co/$HF_REPO (push=$HF_PUSH)"

"$PYTHON" scripts/train_bc.py \
  "${PGN_ARGS[@]}" \
  --out "$OUTPUT_CHECKPOINT" \
  --max-positions "$MAX_POSITIONS" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --num-workers "$NUM_WORKERS" \
  --shuffle-buffer-size 8192 \
  --device cuda \
  --hf-push "$HF_PUSH" \
  --hf-repo "$HF_REPO"

if [[ ! -s "$OUTPUT_CHECKPOINT" ]]; then
  echo "error: joint checkpoint was not created: $OUTPUT_CHECKPOINT" >&2
  exit 1
fi

echo "training complete"
echo "joint checkpoint: $OUTPUT_CHECKPOINT"
echo "hf repo:          https://huggingface.co/$HF_REPO"

GATE="${GATE:-true}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-runs/value/value_final.pt}"
VALUE_LABEL_CACHE="${VALUE_LABEL_CACHE:-data/stockfish/joint_d14_mpv8_250000.pt}"
if [[ "$GATE" == "true" ]]; then
  echo
  if [[ ! -s "$BASELINE_CHECKPOINT" ]]; then
    echo "warn: baseline $BASELINE_CHECKPOINT missing; skipping value gate" >&2
  elif [[ ! -s "$VALUE_LABEL_CACHE" ]]; then
    echo "warn: label cache $VALUE_LABEL_CACHE missing; skipping value gate" >&2
  else
    echo "gate: value head vs frozen-trunk baseline on held-out Stockfish targets"
    "$PYTHON" scripts/gate_value.py \
      --checkpoint "baseline=$BASELINE_CHECKPOINT" \
      --checkpoint "joint_scratch=$OUTPUT_CHECKPOINT" \
      --label-cache "$VALUE_LABEL_CACHE" \
      --device cuda
  fi
fi
echo
echo "NEXT: if the value gate shows Pearson/decisive-sign clearly above baseline,"
echo "      run the search gate before promoting."
