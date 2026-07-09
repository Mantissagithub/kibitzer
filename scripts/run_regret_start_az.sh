#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ACTION="${ACTION:-all}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/regret/policy_regret_repair.pt}"
STARTS_JSONL="${STARTS_JSONL:-runs/regret/az12_policy_regret_sf12.jsonl}"
OUT_JSONL="${OUT_JSONL:-runs/regret_start/targeted_selfplay.jsonl}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/regret_start/regret_start_az.pt}"
MAX_STARTS="${MAX_STARTS:-1000}"
MIN_REGRET="${MIN_REGRET:-0.05}"
SIMS="${SIMS:-128}"
PLIES="${PLIES:-32}"
TEMP_PLIES="${TEMP_PLIES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.15}"
VALUE_TARGET="${VALUE_TARGET:-root}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.75}"
VALUE_WEIGHT="${VALUE_WEIGHT:-0.0}"
UNFREEZE_LAST_TRUNK_BLOCKS="${UNFREEZE_LAST_TRUNK_BLOCKS:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-37}"

if [[ "$ACTION" != "all" && "$ACTION" != "gen" && "$ACTION" != "train" ]]; then
  echo "error: ACTION must be all, gen, or train" >&2
  exit 1
fi
if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "error: init checkpoint is missing: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ "$ACTION" != "train" && ! -s "$STARTS_JSONL" ]]; then
  echo "error: start buffer is missing: $STARTS_JSONL" >&2
  exit 1
fi
if [[ "$ACTION" == "train" && ! -s "$OUT_JSONL" ]]; then
  echo "error: generated self-play data is missing: $OUT_JSONL" >&2
  exit 1
fi

echo "============================================================"
echo " KIBITZER REGRET-START MINI SELF-PLAY"
echo "============================================================"
echo "Action:              $ACTION"
echo "Init checkpoint:     $INIT_CHECKPOINT"
echo "Start buffer:        $STARTS_JSONL"
echo "Self-play data:      $OUT_JSONL"
echo "Output checkpoint:   $OUTPUT_CHECKPOINT"
echo "Generation:          starts=$MAX_STARTS sims=$SIMS plies=$PLIES temp_plies=$TEMP_PLIES eps=$DIRICHLET_EPSILON"
echo "Train:               epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE anchor=$ANCHOR_WEIGHT value=$VALUE_WEIGHT"
echo

if [[ "$ACTION" == "all" || "$ACTION" == "gen" ]]; then
  uv run python scripts/train_regret_start_az.py gen \
    --checkpoint "$INIT_CHECKPOINT" \
    --starts "$STARTS_JSONL" \
    --out-jsonl "$OUT_JSONL" \
    --max-starts "$MAX_STARTS" \
    --min-regret "$MIN_REGRET" \
    --sims "$SIMS" \
    --plies "$PLIES" \
    --temp-plies "$TEMP_PLIES" \
    --temperature "$TEMPERATURE" \
    --dirichlet-epsilon "$DIRICHLET_EPSILON" \
    --value-target "$VALUE_TARGET" \
    --shuffle \
    --seed "$SEED" \
    --device "$DEVICE"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "train" ]]; then
  uv run python scripts/train_regret_start_az.py train \
    --checkpoint "$INIT_CHECKPOINT" \
    --data "$OUT_JSONL" \
    --out "$OUTPUT_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --anchor-weight "$ANCHOR_WEIGHT" \
    --value-weight "$VALUE_WEIGHT" \
    --unfreeze-last-trunk-blocks "$UNFREEZE_LAST_TRUNK_BLOCKS" \
    --seed "$SEED" \
    --device "$DEVICE"
fi

echo "REGRET_START_AZ_DONE"
