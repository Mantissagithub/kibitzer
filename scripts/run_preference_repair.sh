#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

ACTION="${ACTION:-all}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
PREFERENCE_JSONL="${PREFERENCE_JSONL:-runs/preference/r1_teacher_pairs_sf12.jsonl}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/preference/preference_repair.pt}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
STOCKFISH_DEPTH="${STOCKFISH_DEPTH:-12}"
STOCKFISH_MULTIPV="${STOCKFISH_MULTIPV:-8}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
MAX_POSITIONS="${MAX_POSITIONS:-100000}"
MIN_MARGIN="${MIN_MARGIN:-0.08}"
SOURCE_JSONLS="${SOURCE_JSONLS:-runs/regret/az12_policy_regret_sf12.jsonl,runs/regret/az12_policy_regret_sf12_bigger.jsonl}"
SOURCE_PGNS="${SOURCE_PGNS:-data/raw/lichess_elite_2025-11.pgn}"
POSITION_STRIDE="${POSITION_STRIDE:-6}"
MIN_PLY="${MIN_PLY:-8}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TEMPERATURE="${TEMPERATURE:-0.05}"
BETA="${BETA:-0.1}"
CE_WEIGHT="${CE_WEIGHT:-0.25}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.05}"
UNFREEZE_LAST_TRUNK_BLOCKS="${UNFREEZE_LAST_TRUNK_BLOCKS:-0}"
SEED="${SEED:-31}"
DEVICE="${DEVICE:-cuda}"

if [[ "$ACTION" != "all" && "$ACTION" != "label" && "$ACTION" != "train" ]]; then
  echo "error: ACTION must be all, label, or train" >&2
  exit 1
fi
if [[ ! -s "$BASE_CHECKPOINT" ]]; then
  echo "error: base checkpoint is missing: $BASE_CHECKPOINT" >&2
  exit 1
fi
if [[ "$ACTION" != "train" && ( -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ) ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi
if [[ "$ACTION" != "label" && ! -s "$PREFERENCE_JSONL" && "$ACTION" == "train" ]]; then
  echo "error: preference buffer is missing: $PREFERENCE_JSONL" >&2
  exit 1
fi

MAX_POSITION_ARGS=()
if [[ -n "$MAX_POSITIONS" ]]; then
  MAX_POSITION_ARGS=(--max-positions "$MAX_POSITIONS")
fi

SOURCE_ARGS=()
if [[ -n "$SOURCE_JSONLS" ]]; then
  IFS=',' read -r -a REQUESTED_JSONLS <<< "$SOURCE_JSONLS"
  for path in "${REQUESTED_JSONLS[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "error: source JSONL is missing: $path" >&2
      exit 1
    fi
    SOURCE_ARGS+=(--jsonl "$path")
  done
fi
if [[ -n "$SOURCE_PGNS" ]]; then
  IFS=',' read -r -a REQUESTED_PGNS <<< "$SOURCE_PGNS"
  for path in "${REQUESTED_PGNS[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "error: source PGN is missing: $path" >&2
      exit 1
    fi
    SOURCE_ARGS+=(--pgn "$path")
  done
fi
if [[ "$ACTION" != "train" && "${#SOURCE_ARGS[@]}" -eq 0 ]]; then
  echo "error: at least one SOURCE_JSONLS or SOURCE_PGNS entry is required" >&2
  exit 1
fi

echo "============================================================"
echo " KIBITZER TEACHER-PREFERENCE REPAIR"
echo "============================================================"
echo "Action:              $ACTION"
echo "Base/ref checkpoint: $BASE_CHECKPOINT"
echo "Preference buffer:   $PREFERENCE_JSONL"
echo "Output checkpoint:   $OUTPUT_CHECKPOINT"
echo "Stockfish:           ${STOCKFISH_PATH:-none} depth=$STOCKFISH_DEPTH multipv=$STOCKFISH_MULTIPV workers=$STOCKFISH_WORKERS"
echo "Sources JSONL:       ${SOURCE_JSONLS:-none}"
echo "Sources PGN:         ${SOURCE_PGNS:-none}"
echo "Label filters:       max_positions=${MAX_POSITIONS:-all} min_margin=$MIN_MARGIN stride=$POSITION_STRIDE min_ply=$MIN_PLY"
echo "Train:               epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE beta=$BETA ce=$CE_WEIGHT anchor=$ANCHOR_WEIGHT"
echo
echo "Interpretation:"
echo "  - kept pairs are positions where the teacher clearly prefers one move over a model-attractive mistake"
echo "  - pair_acc should rise, but this is only an offline sanity check"
echo "  - promote only after the 80-game Leela/Maia external gate beats tactical R1"
echo

if [[ "$ACTION" == "all" || "$ACTION" == "label" ]]; then
  uv run python scripts/train_preference_repair.py label \
    "${SOURCE_ARGS[@]}" \
    --checkpoint "$BASE_CHECKPOINT" \
    --out-jsonl "$PREFERENCE_JSONL" \
    --stockfish-path "$STOCKFISH_PATH" \
    --stockfish-workers "$STOCKFISH_WORKERS" \
    --depth "$STOCKFISH_DEPTH" \
    --multipv "$STOCKFISH_MULTIPV" \
    --min-margin "$MIN_MARGIN" \
    --min-ply "$MIN_PLY" \
    --position-stride "$POSITION_STRIDE" \
    --seed "$SEED" \
    --shuffle \
    --device "$DEVICE" \
    "${MAX_POSITION_ARGS[@]}"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "train" ]]; then
  uv run python scripts/train_preference_repair.py train \
    --checkpoint "$BASE_CHECKPOINT" \
    --data "$PREFERENCE_JSONL" \
    --out "$OUTPUT_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --temperature "$TEMPERATURE" \
    --beta "$BETA" \
    --ce-weight "$CE_WEIGHT" \
    --anchor-weight "$ANCHOR_WEIGHT" \
    --unfreeze-last-trunk-blocks "$UNFREEZE_LAST_TRUNK_BLOCKS" \
    --seed "$SEED" \
    --device "$DEVICE"
fi

echo
echo "PREFERENCE_REPAIR_DONE"
echo "External gate command:"
echo "  CANDIDATE_NAME=preference_repair CANDIDATE_CHECKPOINT=$OUTPUT_CHECKPOINT CANDIDATE_REPORT_DIR=reports/preference_repair SEED=31 bash scripts/run_repair_eval_gate.sh"
