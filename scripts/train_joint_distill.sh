#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

if [[ ! -f .env ]]; then
  echo "error: .env is required for the mandatory Hugging Face upload" >&2
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
    echo "error: install uv or create .venv with project dependencies" >&2
    exit 1
  fi
  echo "[setup] creating project environment"
  uv sync
fi
PYTHON="$ROOT_DIR/.venv/bin/python"

DATA_YEAR="${DATA_YEAR:-2025}"
DATA_MONTHS="${DATA_MONTHS:-06,07,08,09,10,11}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/value/value_final.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/joint_distill/joint_best.pt}"
MAX_POSITIONS="${MAX_POSITIONS:-250000}"
STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-14}"
STOCKFISH_MULTIPV="${STOCKFISH_MULTIPV:-8}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VALUE_WEIGHT="${VALUE_WEIGHT:-1.0}"
UNFREEZE_TRUNK_BLOCKS="${UNFREEZE_TRUNK_BLOCKS:-3}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.02}"
LABEL_CACHE="${LABEL_CACHE:-data/stockfish/joint_d${STOCKFISH_DEPTH}_mpv${STOCKFISH_MULTIPV}_${MAX_POSITIONS}.pt}"
HF_REPO="${HF_REPO:-${HF_USERNAME}/kibitzer-clean-joint-distill}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
readonly HF_PUSH=true

echo "============================================================"
echo " KIBITZER PHASE 3 — JOINT STOCKFISH DISTILLATION"
echo "============================================================"
echo "Purpose: teach the policy tactical engine moves while refining value."
echo "Initialization:       $INIT_CHECKPOINT"
echo "Teacher labels:       depth=$STOCKFISH_DEPTH MultiPV=$STOCKFISH_MULTIPV positions=$MAX_POSITIONS"
echo "Adaptation scope:     both heads + final norm + last $UNFREEZE_TRUNK_BLOCKS trunk blocks"
echo "Label cache:          $LABEL_CACHE"
echo "Output checkpoint:    $OUTPUT_CHECKPOINT"
echo "Hugging Face target:  https://huggingface.co/$HF_REPO"
echo

if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "error: trained policy/value checkpoint does not exist: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

echo "[setup 1/3] checking cached PGN sources"
"$PYTHON" scripts/download_lichess_elite.py \
  --year "$DATA_YEAR" \
  --months "$DATA_MONTHS" \
  --output-dir data/raw

PGN_ARGS=()
IFS=',' read -r -a REQUESTED_MONTHS <<< "$DATA_MONTHS"
for month in "${REQUESTED_MONTHS[@]}"; do
  printf -v normalized_month '%02d' "$((10#$month))"
  pgn_path="data/raw/lichess_elite_${DATA_YEAR}-${normalized_month}.pgn"
  if [[ ! -s "$pgn_path" ]]; then
    echo "error: PGN does not exist or is empty: $pgn_path" >&2
    exit 1
  fi
  PGN_ARGS+=(--pgn "$pgn_path")
done
echo "  ready: ${#PGN_ARGS[@]} PGN files"

echo "[setup 2/3] checking CUDA and Hugging Face identity"
"$PYTHON" - <<'PY'
import os

import torch
from huggingface_hub import HfApi

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is required for joint distillation")
expected = os.environ["HF_USERNAME"]
actual = HfApi(token=os.environ["HF_TOKEN"]).whoami().get("name")
if actual != expected:
    raise SystemExit(f"error: HF token belongs to {actual!r}, expected {expected!r}")
print(f"  CUDA: {torch.cuda.get_device_name(0)}")
print(f"  Hugging Face: authenticated as {actual}")
PY

echo "[setup 3/3] prerequisites complete"
echo

"$PYTHON" scripts/distill_stockfish.py \
  "${PGN_ARGS[@]}" \
  --init "$INIT_CHECKPOINT" \
  --out "$OUTPUT_CHECKPOINT" \
  --label-cache "$LABEL_CACHE" \
  --stockfish-path "$STOCKFISH_PATH" \
  --stockfish-workers "$STOCKFISH_WORKERS" \
  --depth "$STOCKFISH_DEPTH" \
  --multipv "$STOCKFISH_MULTIPV" \
  --temperature "$TEMPERATURE" \
  --max-positions "$MAX_POSITIONS" \
  --eval-fraction "$EVAL_FRACTION" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --value-weight "$VALUE_WEIGHT" \
  --unfreeze-last-trunk-blocks "$UNFREEZE_TRUNK_BLOCKS" \
  --device cuda \
  --hf-push "$HF_PUSH" \
  --hf-repo "$HF_REPO"

echo
echo "NEXT GATE"
echo "  CHECKPOINT=$OUTPUT_CHECKPOINT bash scripts/run_search_eval.sh"
