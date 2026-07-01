#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
ACTION="${ACTION:-build}"
DATA_YEAR="${DATA_YEAR:-2025}"
UNSEEN_MONTHS="${UNSEEN_MONTHS:-01,02,03,04,05}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
ORACLE_CACHE="${ORACLE_CACHE:-data/diagnostics/unseen_${DATA_YEAR}_d20.pt}"
CHOSEN_MOVE_CACHE="${CHOSEN_MOVE_CACHE:-data/diagnostics/chosen_moves_d20.pt}"
PHASE2_CHECKPOINT="${PHASE2_CHECKPOINT:-runs/value/value_final.pt}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-runs/joint_distill/joint_best.pt}"
SIMULATIONS="${SIMULATIONS:-64}"
VALUE_SCALES="${VALUE_SCALES:-0,0.5,1}"
TACTICS_EPD="${TACTICS_EPD:-resources/mate_in_one_sanity.epd}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: project Python does not exist: $PYTHON" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH" >&2
  exit 1
fi

PGN_ARGS=()
IFS=',' read -r -a MONTHS <<< "$UNSEEN_MONTHS"
for month in "${MONTHS[@]}"; do
  printf -v normalized_month '%02d' "$((10#$month))"
  PGN_ARGS+=(--pgn "data/raw/lichess_elite_${DATA_YEAR}-${normalized_month}.pgn")
done

case "$ACTION" in
  build)
    echo "============================================================"
    echo " SEARCH DIAGNOSTICS — BUILD LOCKED COMMON ORACLE"
    echo "============================================================"
    echo "Unseen data:      Lichess Elite $DATA_YEAR months $UNSEEN_MONTHS"
    echo "Oracle:           Stockfish depth 20, MultiPV 1"
    echo "Split:            game-disjoint validation/test"
    echo "Target:           200 positions per value bin per split"
    echo "Output:           $ORACLE_CACHE"
    echo
    "$PYTHON" scripts/download_lichess_elite.py \
      --year "$DATA_YEAR" \
      --months "$UNSEEN_MONTHS" \
      --output-dir data/raw
    "$PYTHON" scripts/diagnose_search.py build \
      "${PGN_ARGS[@]}" \
      --output "$ORACLE_CACHE" \
      --stockfish-path "$STOCKFISH_PATH" \
      --stockfish-workers "$STOCKFISH_WORKERS" \
      --prescan-depth 10 \
      --oracle-depth 20 \
      --positions-per-bin 200 \
      --oversample 3
    echo
    echo "NEXT: ACTION=validate bash scripts/run_search_diagnostics.sh"
    ;;
  validate)
    if [[ ! -s "$ORACLE_CACHE" ]]; then
      echo "error: build the oracle first: $ORACLE_CACHE" >&2
      exit 1
    fi
    echo "============================================================"
    echo " SEARCH DIAGNOSTICS — VALIDATION / CONFIG SELECTION"
    echo "============================================================"
    "$PYTHON" scripts/diagnose_search.py evaluate \
      --oracle "$ORACLE_CACHE" \
      --split validation \
      --checkpoint "phase2=$PHASE2_CHECKPOINT" \
      --checkpoint "joint=$JOINT_CHECKPOINT" \
      --output runs/diagnostics/validation.json \
      --chosen-move-cache "$CHOSEN_MOVE_CACHE" \
      --stockfish-path "$STOCKFISH_PATH" \
      --stockfish-workers "$STOCKFISH_WORKERS" \
      --simulations "$SIMULATIONS" \
      --value-scales "$VALUE_SCALES" \
      --tactics-epd "$TACTICS_EPD" \
      --device cuda
    echo
    echo "The test split stays locked until one checkpoint/value-scale is selected."
    ;;
  test)
    : "${SELECTED_NAME:?error: SELECTED_NAME is required for ACTION=test}"
    : "${SELECTED_CHECKPOINT:?error: SELECTED_CHECKPOINT is required for ACTION=test}"
    : "${SELECTED_VALUE_SCALE:?error: SELECTED_VALUE_SCALE is required for ACTION=test}"
    if [[ ! -s "$ORACLE_CACHE" ]]; then
      echo "error: oracle does not exist: $ORACLE_CACHE" >&2
      exit 1
    fi
    "$PYTHON" scripts/diagnose_search.py evaluate \
      --oracle "$ORACLE_CACHE" \
      --split test \
      --checkpoint "phase2=$PHASE2_CHECKPOINT" \
      --checkpoint "$SELECTED_NAME=$SELECTED_CHECKPOINT" \
      --output runs/diagnostics/test.json \
      --chosen-move-cache "$CHOSEN_MOVE_CACHE" \
      --stockfish-path "$STOCKFISH_PATH" \
      --stockfish-workers "$STOCKFISH_WORKERS" \
      --simulations "$SIMULATIONS" \
      --value-scales "$SELECTED_VALUE_SCALE" \
      --tactics-epd "$TACTICS_EPD" \
      --device cuda
    ;;
  *)
    echo "error: ACTION must be build, validate, or test" >&2
    exit 1
    ;;
esac
