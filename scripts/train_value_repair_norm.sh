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
  echo "error: .venv is missing; run uv sync first" >&2
  exit 1
fi
PYTHON="$ROOT_DIR/.venv/bin/python"

DATA_YEAR="${DATA_YEAR:-2025}"
DATA_MONTHS="${DATA_MONTHS:-06,07,08,09,10,11}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/value_repair/value_repair_best.pt}"
POLICY_ANCHOR_CHECKPOINT="${POLICY_ANCHOR_CHECKPOINT:-runs/value/value_final.pt}"
LABEL_CACHE="${LABEL_CACHE:-data/stockfish/joint_d14_mpv8_250000.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/value_repair_norm/value_repair_norm_best.pt}"
MAX_POSITIONS="${MAX_POSITIONS:-250000}"
STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-14}"
CACHE_MULTIPV="${CACHE_MULTIPV:-8}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
NORM_LEARNING_RATE="${NORM_LEARNING_RATE:-2e-5}"
UNFREEZE_TRUNK_BLOCKS="${UNFREEZE_TRUNK_BLOCKS:-}"
TRUNK_LEARNING_RATE="${TRUNK_LEARNING_RATE:-5e-6}"
POLICY_KL_WEIGHT="${POLICY_KL_WEIGHT:-1.0}"
MAX_POLICY_KL="${MAX_POLICY_KL:-0.01}"
MIN_POLICY_TOP1_AGREEMENT="${MIN_POLICY_TOP1_AGREEMENT:-0.98}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.5}"
MAX_SAMPLING_WEIGHT="${MAX_SAMPLING_WEIGHT:-4.0}"
HF_REPO="${HF_REPO:-${HF_USERNAME}/kibitzer-clean-value-repair-norm}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
readonly HF_PUSH=true

TRUNK_ARGS=()
TRAINABLE_SCOPE="value head + final RMSNorm only"
if [[ -n "$UNFREEZE_TRUNK_BLOCKS" ]]; then
  TRUNK_ARGS+=(
    --unfreeze-last-trunk-blocks "$UNFREEZE_TRUNK_BLOCKS"
    --trunk-lr "$TRUNK_LEARNING_RATE"
  )
  TRAINABLE_SCOPE="value head + final RMSNorm + last $UNFREEZE_TRUNK_BLOCKS trunk block(s)"
fi

echo "============================================================"
echo " KIBITZER ${STAGE_LABEL:-STAGE B — VALUE HEAD + FINAL NORM REPAIR}"
echo "============================================================"
echo "Purpose: give value learning limited representation capacity while"
echo "         preserving the Phase-2 policy distribution."
echo "Initialization:       $INIT_CHECKPOINT"
echo "Policy anchor:        $POLICY_ANCHOR_CHECKPOINT"
echo "Teacher cache:        $LABEL_CACHE"
echo "Trainable scope:      $TRAINABLE_SCOPE"
echo "Learning rates:       head=$HEAD_LEARNING_RATE norm=$NORM_LEARNING_RATE"
if [[ -n "$UNFREEZE_TRUNK_BLOCKS" ]]; then
  echo "Trunk learning rate:  $TRUNK_LEARNING_RATE"
fi
echo "Policy anchor:        KL weight=$POLICY_KL_WEIGHT"
echo "Policy drift floors:  KL <= $MAX_POLICY_KL, top-1 agreement >= $MIN_POLICY_TOP1_AGREEMENT"
echo "Sampling:             inverse-frequency alpha=$SAMPLING_ALPHA cap=${MAX_SAMPLING_WEIGHT}x"
echo "Epochs:               $EPOCHS (every epoch retained)"
echo "Output checkpoint:    $OUTPUT_CHECKPOINT"
echo "Hugging Face target:  https://huggingface.co/$HF_REPO"
echo "Safety:               locked diagnostic test split is not used"
echo

for checkpoint in "$INIT_CHECKPOINT" "$POLICY_ANCHOR_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "error: required checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
done
if [[ ! -s "$LABEL_CACHE" ]]; then
  echo "error: audited teacher cache is missing: $LABEL_CACHE" >&2
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
    raise SystemExit("error: CUDA is required for norm repair")
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
  --value-only \
  --unfreeze-final-norm \
  "${PGN_ARGS[@]}" \
  --init "$INIT_CHECKPOINT" \
  --policy-anchor-checkpoint "$POLICY_ANCHOR_CHECKPOINT" \
  --policy-kl-weight "$POLICY_KL_WEIGHT" \
  --max-policy-kl "$MAX_POLICY_KL" \
  --min-policy-top1-agreement "$MIN_POLICY_TOP1_AGREEMENT" \
  --out "$OUTPUT_CHECKPOINT" \
  --label-cache "$LABEL_CACHE" \
  --stockfish-path "$STOCKFISH_PATH" \
  --stockfish-workers "$STOCKFISH_WORKERS" \
  --depth "$STOCKFISH_DEPTH" \
  --multipv "$CACHE_MULTIPV" \
  --value-cache-multipv "$CACHE_MULTIPV" \
  --max-positions "$MAX_POSITIONS" \
  --eval-fraction "$EVAL_FRACTION" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$HEAD_LEARNING_RATE" \
  --norm-lr "$NORM_LEARNING_RATE" \
  "${TRUNK_ARGS[@]}" \
  --value-bin-sampling-alpha "$SAMPLING_ALPHA" \
  --max-sampling-weight "$MAX_SAMPLING_WEIGHT" \
  --save-every-epoch \
  --value-repair-selection \
  --device cuda \
  --hf-push "$HF_PUSH" \
  --hf-repo "$HF_REPO"

echo
echo "NEXT GATE"
echo "  Do not consume the locked test split. Send the epoch summaries first."
