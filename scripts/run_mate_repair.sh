#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

# narrow patch for the mate-blindness seen in the official-Elo PGN. this is not
# another broad tactical repair; it only lets mate-tagged puzzle lines dominate,
# with a small game-corpus anchor so the model does not learn "check everything".

INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
PUZZLE_CSV="${PUZZLE_CSV:-data/puzzles/lichess_db_puzzle.csv}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/mate/mate_repair.pt}"
DATA_YEAR="${DATA_YEAR:-2025}"
DATA_MONTHS="${DATA_MONTHS:-06,07,08,09,10}"
EVAL_MONTH="${EVAL_MONTH:-11}"
MAX_POSITIONS="${MAX_POSITIONS:-120000}"
MIX_RATIO="${MIX_RATIO:-0.85}"
MATE_THEMES="${MATE_THEMES:-mate mateIn1 mateIn2 mateIn3 mateIn4}"
RATING_MIN="${RATING_MIN:-900}"
RATING_MAX="${RATING_MAX:-2800}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
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
echo " KIBITZER MATE-ONLY REPAIR PATCH"
echo "============================================================"
echo "Init checkpoint:     $INIT_CHECKPOINT"
echo "Puzzle CSV:          $PUZZLE_CSV"
echo "Mate themes:         $MATE_THEMES"
echo "Game anchor months:  $DATA_YEAR-$DATA_MONTHS"
echo "Eval PGN:            $EVAL_PGN"
echo "Output checkpoint:   $OUTPUT_CHECKPOINT"
echo "Training:            max_positions=$MAX_POSITIONS mate_mix=$MIX_RATIO rating=${RATING_MIN}-${RATING_MAX}"
echo "Optimizer:           lr=$LEARNING_RATE value_weight=$VALUE_WEIGHT batch=$BATCH_SIZE workers=$NUM_WORKERS"
echo "Gate:                held-out game top1 should not drop by >0.5pp"
echo
echo "Read:"
echo "  - this patch targets forced mate delivery / mate-line forcing moves"
echo "  - it is not a promotion by itself"
echo "  - after training, run a 128-sim gate first, then only test 512 if it is not worse"
echo

uv run python scripts/train_midtrain.py \
  --checkpoint "$INIT_CHECKPOINT" \
  --puzzle-csv "$PUZZLE_CSV" \
  --theme-any "$MATE_THEMES" \
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

echo
echo "MATE_REPAIR_DONE -> $OUTPUT_CHECKPOINT"
echo
echo "Next quick gate:"
echo "  uv run python scripts/maia_gauntlet.py \\"
echo "    --checkpoint $OUTPUT_CHECKPOINT \\"
echo "    --maia-weights data/leela/t1-256x10-distilled.pb.gz --maia-elo 2700 \\"
echo "    --lc0-path data/leela/lc0 --backend cuda --maia-nodes 1 \\"
echo "    --games 40 --simulations 128 --seed 23 \\"
echo "    --out-jsonl reports/mate_repair/mate_repair_vs2700_s128_g40_seed23.jsonl \\"
echo "    --out-pgn reports/mate_repair/mate_repair_vs2700_s128_g40_seed23.pgn --device cuda"
