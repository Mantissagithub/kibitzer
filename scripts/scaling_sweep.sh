#!/usr/bin/env bash
set -euo pipefail

# Scaling-law sweep launcher (docs/scaling_study/, LOGBOOK.md D36).
#
# Downloads/caches Lichess Elite months 06-10 for training and month 11 as the
# held-out eval set, then drives scripts/scaling_sweep.py over the ladder.

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -x .venv/bin/python ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: install uv or create .venv with the project dependencies" >&2
    exit 1
  fi
  uv sync
fi
PYTHON="$ROOT_DIR/.venv/bin/python"

DATA_YEAR="${DATA_YEAR:-2025}"
TRAIN_MONTHS="${TRAIN_MONTHS:-06,07,08,09,10}"
EVAL_MONTH="${EVAL_MONTH:-11}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 {print $4}')"
if (( FREE_KB < MIN_FREE_GB * 1024 * 1024 )); then
  echo "error: automatic dataset download requires at least ${MIN_FREE_GB}GB free" >&2
  exit 1
fi

echo "stage 1/2: download/cache Lichess Elite ${DATA_YEAR} months ${TRAIN_MONTHS},${EVAL_MONTH}"
"$PYTHON" scripts/download_lichess_elite.py \
  --year "$DATA_YEAR" \
  --months "${TRAIN_MONTHS},${EVAL_MONTH}" \
  --output-dir data/raw

TRAIN_PGN_ARGS=()
IFS=',' read -r -a TRAIN_MONTH_LIST <<< "$TRAIN_MONTHS"
for month in "${TRAIN_MONTH_LIST[@]}"; do
  printf -v normalized_month '%02d' "$((10#$month))"
  path="data/raw/lichess_elite_${DATA_YEAR}-${normalized_month}.pgn"
  if [[ ! -s "$path" ]]; then
    echo "error: PGN does not exist or is empty: $path" >&2
    exit 1
  fi
  TRAIN_PGN_ARGS+=(--train-pgn "$path")
done

printf -v normalized_eval_month '%02d' "$((10#$EVAL_MONTH))"
EVAL_PGN="data/raw/lichess_elite_${DATA_YEAR}-${normalized_eval_month}.pgn"
if [[ ! -s "$EVAL_PGN" ]]; then
  echo "error: PGN does not exist or is empty: $EVAL_PGN" >&2
  exit 1
fi

SIZES="${SIZES:-S0,S1,S2,S3}"
MAX_POSITIONS="${MAX_POSITIONS:-5000000}"
EVAL_POSITIONS="${EVAL_POSITIONS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BASE_LR="${BASE_LR:-3e-4}"
LR_TRANSFER="${LR_TRANSFER:-mup}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-reports/scaling_law/results.json}"
CKPT_DIR="${CKPT_DIR:-runs/scaling}"

echo "stage 2/2: scaling sweep"
echo "  sizes:           $SIZES"
echo "  positions/rung:  $MAX_POSITIONS"
echo "  train months:    $TRAIN_MONTHS"
echo "  eval month:       $EVAL_MONTH"
echo "  out:             $OUT"

"$PYTHON" scripts/scaling_sweep.py \
  "${TRAIN_PGN_ARGS[@]}" \
  --eval-pgn "$EVAL_PGN" \
  --sizes "$SIZES" \
  --max-positions "$MAX_POSITIONS" \
  --eval-positions "$EVAL_POSITIONS" \
  --batch-size "$BATCH_SIZE" \
  --base-lr "$BASE_LR" \
  --lr-transfer "$LR_TRANSFER" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --out "$OUT" \
  --ckpt-dir "$CKPT_DIR"

echo
echo "sweep complete: $OUT"
