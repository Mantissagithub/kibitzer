#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

# grpo + exact-dppo external-reward loop. SMOKE=1 runs a tiny cpu-friendly single
# iteration to sanity-check the pipeline before committing gpu time.
BASE_CHECKPOINT="${BASE_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-$BASE_CHECKPOINT}"
OUT_DIR="${OUT_DIR:-runs/grpo}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
ITERATIONS="${ITERATIONS:-30}"
ELO="${ELO:-1900}"
ELO_LO="${ELO_LO:-1600}"
ELO_HI="${ELO_HI:-2600}"
PROBE_ELO="${PROBE_ELO:-2000}"
PROBE_EVERY="${PROBE_EVERY:-5}"
PROBE_SIMS="${PROBE_SIMS:-128}"
NUM_GROUPS="${NUM_GROUPS:-10}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SIMS="${SIMS:-128}"
TEMP="${TEMP:-0.8}"
TEMP_PLIES="${TEMP_PLIES:-16}"
TEMP_LATE="${TEMP_LATE:-0.0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
DELTA="${DELTA:-0.2}"
BETA="${BETA:-0.05}"
SCOPE="${SCOPE:-heads}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-1}"
ENGINE_TIME="${ENGINE_TIME:-0.01}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  ITERATIONS=1
  NUM_GROUPS=2
  GROUP_SIZE=3
  SIMS=8
  ELO=1600
  BATCH_SIZE=64
  PROBE_EVERY=999
  DEVICE=cpu
  OUT_DIR=runs/grpo_smoke
fi

if [[ ! -s "$BASE_CHECKPOINT" ]]; then
  echo "error: base checkpoint is missing: $BASE_CHECKPOINT" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

echo "============================================================"
echo " KIBITZER GRPO + EXACT-DPPO (external reward)"
echo "============================================================"
echo "Base checkpoint:   $BASE_CHECKPOINT"
echo "Frozen anchor:     $ANCHOR_CHECKPOINT"
echo "Out dir:           $OUT_DIR"
echo "Stockfish:         ${STOCKFISH_PATH:-none}"
echo "Ladder:            elo=$ELO in [$ELO_LO,$ELO_HI]  probe@$PROBE_ELO (s$PROBE_SIMS) every $PROBE_EVERY"
echo "Rollout:           groups=$NUM_GROUPS x G=$GROUP_SIZE  sims=$SIMS  temp=$TEMP/${TEMP_PLIES}ply->$TEMP_LATE  dir=$DIRICHLET_ALPHA/$DIRICHLET_EPSILON"
echo "Update:            dppo-tv delta=$DELTA beta=$BETA scope=$SCOPE lr=$LEARNING_RATE epochs=$EPOCHS batch=$BATCH_SIZE"
echo "Smoke:             ${SMOKE:-0}   Iterations: $ITERATIONS"
echo

uv run python scripts/train_grpo.py loop \
  --checkpoint "$BASE_CHECKPOINT" \
  --anchor "$ANCHOR_CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --iterations "$ITERATIONS" \
  --stockfish "$STOCKFISH_PATH" \
  --elo "$ELO" \
  --elo-lo "$ELO_LO" \
  --elo-hi "$ELO_HI" \
  --probe-elo "$PROBE_ELO" \
  --probe-every "$PROBE_EVERY" \
  --probe-sims "$PROBE_SIMS" \
  --groups "$NUM_GROUPS" \
  --group-size "$GROUP_SIZE" \
  --sims "$SIMS" \
  --temp "$TEMP" \
  --temp-plies "$TEMP_PLIES" \
  --temp-late "$TEMP_LATE" \
  --dirichlet-alpha "$DIRICHLET_ALPHA" \
  --dirichlet-epsilon "$DIRICHLET_EPSILON" \
  --delta "$DELTA" \
  --beta "$BETA" \
  --scope "$SCOPE" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --epochs "$EPOCHS" \
  --engine-time "$ENGINE_TIME" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "GRPO_DONE"
