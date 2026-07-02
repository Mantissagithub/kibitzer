#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv is missing; run uv sync first" >&2
  exit 1
fi
PYTHON="$ROOT_DIR/.venv/bin/python"

LABEL_CACHE="${LABEL_CACHE:-data/stockfish/joint_d14_mpv8_250000.pt}"
AUDIT_POSITIONS="${AUDIT_POSITIONS:-3000}"
AUDIT_DEPTH="${AUDIT_DEPTH:-20}"
STOCKFISH_WORKERS="${STOCKFISH_WORKERS:-8}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-128}"
AUDIT_SEED="${AUDIT_SEED:-42}"
AUDIT_CACHE="${AUDIT_CACHE:-data/diagnostics/value_labels_d14_vs_d${AUDIT_DEPTH}_${AUDIT_POSITIONS}.pt}"
AUDIT_REPORT="${AUDIT_REPORT:-runs/diagnostics/value_label_audit.json}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"

if [[ ! -s "$LABEL_CACHE" ]]; then
  echo "error: cached joint labels are missing: $LABEL_CACHE" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

echo "============================================================"
echo " KIBITZER VALUE-LABEL CEILING AUDIT"
echo "============================================================"
echo "Purpose: determine whether depth-14 labels are accurate enough"
echo "         for another value-head training run."
echo "Source labels:      $LABEL_CACHE"
echo "Audit:              $AUDIT_POSITIONS positions at depth $AUDIT_DEPTH"
echo "Stockfish workers:  $STOCKFISH_WORKERS"
echo "Resumable cache:    $AUDIT_CACHE"
echo "Report:             $AUDIT_REPORT"
echo "Safety:             locked validation test split is not used"
echo

"$PYTHON" scripts/audit_value_labels.py \
  --label-cache "$LABEL_CACHE" \
  --audit-cache "$AUDIT_CACHE" \
  --output "$AUDIT_REPORT" \
  --stockfish-path "$STOCKFISH_PATH" \
  --stockfish-workers "$STOCKFISH_WORKERS" \
  --depth "$AUDIT_DEPTH" \
  --positions "$AUDIT_POSITIONS" \
  --batch-size "$AUDIT_BATCH_SIZE" \
  --seed "$AUDIT_SEED"

echo
echo "NEXT STEP"
echo "  Send the final [3/3] AUDIT RESULT block before starting more training."
