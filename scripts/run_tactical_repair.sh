#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/regret/policy_regret_repair.pt}"
PUZZLE_CSV="${PUZZLE_CSV:-data/puzzles/lichess_db_puzzle.csv}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
DATA_YEAR="${DATA_YEAR:-2025}"
DATA_MONTHS="${DATA_MONTHS:-06,07,08,09,10}"
EVAL_MONTH="${EVAL_MONTH:-11}"
MAX_POSITIONS="${MAX_POSITIONS:-200000}"
MIX_RATIO="${MIX_RATIO:-0.25}"
RATING_MIN="${RATING_MIN:-1600}"
RATING_MAX="${RATING_MAX:-2600}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VALUE_WEIGHT="${VALUE_WEIGHT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_POSITIONS="${EVAL_POSITIONS:-50000}"
SAVE_EVERY="${SAVE_EVERY:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  echo "error: init checkpoint is missing: $INIT_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -s "$PUZZLE_CSV" ]]; then
  echo "error: puzzle csv is missing: $PUZZLE_CSV" >&2
  exit 1
fi

PGN_ARGS=()
IFS=',' read -r -a REQUESTED_MONTHS <<< "$DATA_MONTHS"
for month in "${REQUESTED_MONTHS[@]}"; do
  printf -v normalized_month '%02d' "$((10#$month))"
  pgn_path="data/raw/lichess_elite_${DATA_YEAR}-${normalized_month}.pgn"
  if [[ ! -s "$pgn_path" ]]; then
    echo "error: PGN does not exist or is empty: $pgn_path" >&2
    exit 1
  fi
  PGN_ARGS+=(--game-pgn "$pgn_path")
done

printf -v normalized_eval_month '%02d' "$((10#$EVAL_MONTH))"
EVAL_PGN="data/raw/lichess_elite_${DATA_YEAR}-${normalized_eval_month}.pgn"
if [[ ! -s "$EVAL_PGN" ]]; then
  echo "error: eval PGN does not exist or is empty: $EVAL_PGN" >&2
  exit 1
fi

echo "============================================================"
echo " KIBITZER TACTICAL SUPERVISED REPAIR"
echo "============================================================"
echo "Init checkpoint:     $INIT_CHECKPOINT"
echo "Puzzle CSV:          $PUZZLE_CSV"
echo "Game months:         $DATA_YEAR-$DATA_MONTHS"
echo "Eval PGN:            $EVAL_PGN"
echo "Output checkpoint:   $OUTPUT_CHECKPOINT"
echo "Training:            max_positions=$MAX_POSITIONS mix=$MIX_RATIO rating=${RATING_MIN}-${RATING_MAX}"
echo "Optimizer:           lr=$LEARNING_RATE value_weight=$VALUE_WEIGHT batch=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Gate:                eval_positions=$EVAL_POSITIONS; warn if held-out top1 drops >0.5pp"
echo
echo "Interpretation:"
echo "  - early training loss should move down, but that alone is not a pass"
echo "  - final held-out top1 must not drop by more than 0.5pp"
echo "  - even with PASS_HELDOUT_TOP1, promote only after Leela/Maia eval"
echo

uv run python scripts/train_midtrain.py \
  --checkpoint "$INIT_CHECKPOINT" \
  --puzzle-csv "$PUZZLE_CSV" \
  "${PGN_ARGS[@]}" \
  --eval-pgn "$EVAL_PGN" \
  --out "$OUTPUT_CHECKPOINT" \
  --max-positions "$MAX_POSITIONS" \
  --mix-ratio "$MIX_RATIO" \
  --rating-min "$RATING_MIN" \
  --rating-max "$RATING_MAX" \
  --lr "$LEARNING_RATE" \
  --value-weight "$VALUE_WEIGHT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --eval-positions "$EVAL_POSITIONS" \
  --save-every "$SAVE_EVERY" \
  --device "$DEVICE"

echo "TACTICAL_REPAIR_DONE"
