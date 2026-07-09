#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ACTION="${ACTION:-all}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-runs/scaling_shaw_comp/S2_shaw_142M_comp.pt}"
AZ_JSONL="${AZ_JSONL:-runs/az/az_data_1.jsonl}"
AZ_JSONLS="${AZ_JSONLS:-}"
REGRET_JSONL="${REGRET_JSONL:-runs/regret/az1_sf12.jsonl}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/regret/regret_repair.pt}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-12}"
STOCKFISH_MULTIPV="${STOCKFISH_MULTIPV:-8}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
MAX_POSITIONS="${MAX_POSITIONS:-}"
MIN_REGRET="${MIN_REGRET:-0.20}"
MIN_OUTCOME_GAP="${MIN_OUTCOME_GAP:-0.75}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TEMPERATURE="${TEMPERATURE:-0.05}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.5}"
VALUE_WEIGHT="${VALUE_WEIGHT:-0.25}"
UNFREEZE_LAST_TRUNK_BLOCKS="${UNFREEZE_LAST_TRUNK_BLOCKS:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ "$ACTION" != "all" && "$ACTION" != "label" && "$ACTION" != "train" ]]; then
  echo "error: ACTION must be all, label, or train" >&2
  exit 1
fi
if [[ ! -s "$BASE_CHECKPOINT" ]]; then
  echo "error: base checkpoint is missing: $BASE_CHECKPOINT" >&2
  exit 1
fi
if [[ "$ACTION" != "train" && ! -s "$AZ_JSONL" ]]; then
  if [[ -z "$AZ_JSONLS" ]]; then
    echo "error: AZ JSONL is missing: $AZ_JSONL" >&2
    exit 1
  fi
fi
if [[ "$ACTION" != "train" && ( -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ) ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi
if [[ "$ACTION" != "label" && ! -s "$REGRET_JSONL" && "$ACTION" == "train" ]]; then
  echo "error: regret buffer is missing: $REGRET_JSONL" >&2
  exit 1
fi

MAX_POSITION_ARGS=()
if [[ -n "$MAX_POSITIONS" ]]; then
  MAX_POSITION_ARGS=(--max-positions "$MAX_POSITIONS")
fi
AZ_ARGS=()
if [[ -n "$AZ_JSONLS" ]]; then
  IFS=',' read -r -a REQUESTED_AZ_JSONLS <<< "$AZ_JSONLS"
  for path in "${REQUESTED_AZ_JSONLS[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "error: AZ JSONL is missing: $path" >&2
      exit 1
    fi
    AZ_ARGS+=(--az-jsonl "$path")
  done
else
  AZ_ARGS=(--az-jsonl "$AZ_JSONL")
fi

echo "============================================================"
echo " KIBITZER REGRET-GUIDED TEACHER REPAIR"
echo "============================================================"
echo "Action:              $ACTION"
echo "Base checkpoint:     $BASE_CHECKPOINT"
echo "AZ source:           ${AZ_JSONLS:-$AZ_JSONL}"
echo "Regret buffer:       $REGRET_JSONL"
echo "Output checkpoint:   $OUTPUT_CHECKPOINT"
echo "Stockfish:           ${STOCKFISH_PATH:-none} depth=$STOCKFISH_DEPTH multipv=$STOCKFISH_MULTIPV workers=$STOCKFISH_WORKERS"
echo "Filters:             regret>=$MIN_REGRET or outcome_gap>=$MIN_OUTCOME_GAP"
echo "Train:               epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE anchor=$ANCHOR_WEIGHT value=$VALUE_WEIGHT"
echo

if [[ "$ACTION" == "all" || "$ACTION" == "label" ]]; then
  uv run python scripts/train_regret_repair.py label \
    "${AZ_ARGS[@]}" \
    --checkpoint "$BASE_CHECKPOINT" \
    --out-jsonl "$REGRET_JSONL" \
    --stockfish-path "$STOCKFISH_PATH" \
    --stockfish-workers "$STOCKFISH_WORKERS" \
    --depth "$STOCKFISH_DEPTH" \
    --multipv "$STOCKFISH_MULTIPV" \
    --min-regret "$MIN_REGRET" \
    --min-outcome-gap "$MIN_OUTCOME_GAP" \
    --device "$DEVICE" \
    "${MAX_POSITION_ARGS[@]}"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "train" ]]; then
  uv run python scripts/train_regret_repair.py train \
    --checkpoint "$BASE_CHECKPOINT" \
    --data "$REGRET_JSONL" \
    --out "$OUTPUT_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --temperature "$TEMPERATURE" \
    --anchor-weight "$ANCHOR_WEIGHT" \
    --value-weight "$VALUE_WEIGHT" \
    --unfreeze-last-trunk-blocks "$UNFREEZE_LAST_TRUNK_BLOCKS" \
    --device "$DEVICE"
fi

echo "REGRET_REPAIR_DONE"
