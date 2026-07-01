#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

TRAIN_STAGE="${TRAIN_STAGE:-value}"
if [[ "$TRAIN_STAGE" != "all" && "$TRAIN_STAGE" != "policy" && "$TRAIN_STAGE" != "value" ]]; then
  echo "error: TRAIN_STAGE must be all, policy, or value" >&2
  exit 1
fi

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
  echo "stage 0/2: download/cache Lichess Elite ${DATA_YEAR} months ${DATA_MONTHS}"
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

STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

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

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-runs/policy/policy_final.pt}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-runs/value/value_final.pt}"
POLICY_HF_REPO="${POLICY_HF_REPO:-${HF_USERNAME}/kibitzer-clean-policy}"
VALUE_HF_REPO="${VALUE_HF_REPO:-${HF_USERNAME}/kibitzer-clean-value}"

POLICY_MAX_POSITIONS="${POLICY_MAX_POSITIONS:-5000000}"
VALUE_MAX_POSITIONS="${VALUE_MAX_POSITIONS:-250000}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-128}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-128}"
POLICY_EPOCHS="${POLICY_EPOCHS:-3}"
VALUE_EPOCHS="${VALUE_EPOCHS:-5}"
POLICY_NUM_WORKERS="${POLICY_NUM_WORKERS:-4}"
VALUE_DEPTH="${VALUE_DEPTH:-14}"
VALUE_EVAL_FRACTION="${VALUE_EVAL_FRACTION:-0.1}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"

readonly HF_PUSH=true

mkdir -p "$(dirname "$POLICY_CHECKPOINT")" "$(dirname "$VALUE_CHECKPOINT")"

if [[ "$TRAIN_STAGE" == "all" || "$TRAIN_STAGE" == "policy" ]]; then
  echo "stage 1/2: policy-only human move cloning"
  "$PYTHON" scripts/train_bc.py \
    --policy-only \
    "${PGN_ARGS[@]}" \
    --out "$POLICY_CHECKPOINT" \
    --max-positions "$POLICY_MAX_POSITIONS" \
    --batch-size "$POLICY_BATCH_SIZE" \
    --epochs "$POLICY_EPOCHS" \
    --lr 3e-4 \
    --num-workers "$POLICY_NUM_WORKERS" \
    --shuffle-buffer-size 8192 \
    --device cuda \
    --hf-push "$HF_PUSH" \
    --hf-repo "$POLICY_HF_REPO"
fi

if [[ "$TRAIN_STAGE" == "all" || "$TRAIN_STAGE" == "value" ]]; then
  if [[ ! -s "$POLICY_CHECKPOINT" ]]; then
    echo "error: policy checkpoint does not exist: $POLICY_CHECKPOINT" >&2
    exit 1
  fi

  echo "stage 2/2: value-head-only Stockfish regression"
  "$PYTHON" scripts/distill_stockfish.py \
    --value-only \
    "${PGN_ARGS[@]}" \
    --init "$POLICY_CHECKPOINT" \
    --out "$VALUE_CHECKPOINT" \
    --stockfish-path "$STOCKFISH_PATH" \
    --stockfish-workers "${STOCKFISH_WORKERS:-8}" \
    --depth "$VALUE_DEPTH" \
    --eval-fraction "$VALUE_EVAL_FRACTION" \
    --max-positions "$VALUE_MAX_POSITIONS" \
    --batch-size "$VALUE_BATCH_SIZE" \
    --epochs "$VALUE_EPOCHS" \
    --lr 3e-4 \
    --device cuda \
    --hf-push "$HF_PUSH" \
    --hf-repo "$VALUE_HF_REPO"

  if [[ ! -s "$VALUE_CHECKPOINT" ]]; then
    echo "error: value checkpoint was not created: $VALUE_CHECKPOINT" >&2
    exit 1
  fi
fi

echo "training complete"
echo "policy checkpoint: $POLICY_CHECKPOINT"
echo "value checkpoint:  $VALUE_CHECKPOINT"
echo "policy HF repo:    https://huggingface.co/$POLICY_HF_REPO"
echo "value HF repo:     https://huggingface.co/$VALUE_HF_REPO"
