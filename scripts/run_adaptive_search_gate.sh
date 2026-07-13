#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
DEVICE="${DEVICE:-cuda}"
OPPONENTS="${OPPONENTS:-2700:1 2850:8 2950:32}"
GAMES="${GAMES:-40}"
MAX_PLIES="${MAX_PLIES:-200}"
SEED="${SEED:-23}"
FIXED_SIMS="${FIXED_SIMS:-512}"
ADAPTIVE_STAGES="${ADAPTIVE_STAGES:-128,256,512,1024}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-0.55}"
TOP2_RATIO_THRESHOLD="${TOP2_RATIO_THRESHOLD:-0.75}"
VALUE_DELTA_THRESHOLD="${VALUE_DELTA_THRESHOLD:-0.10}"
MAX_AVG_SIMS="${MAX_AVG_SIMS:-512}"
OUT="${OUT:-reports/adaptive_search}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  OPPONENTS="${SMOKE_OPPONENTS:-2700:1}"
  GAMES="${SMOKE_GAMES:-2}"
  MAX_PLIES="${SMOKE_MAX_PLIES:-1}"
  FIXED_SIMS="${SMOKE_FIXED_SIMS:-1}"
  ADAPTIVE_STAGES="${SMOKE_ADAPTIVE_STAGES:-1,2}"
  MAX_AVG_SIMS="${SMOKE_MAX_AVG_SIMS:-2}"
  BACKEND="${SMOKE_BACKEND:-eigen}"
  DEVICE="${SMOKE_DEVICE:-cpu}"
  OUT="${SMOKE_OUT:-/tmp/kibitzer_adaptive_search_smoke}"
fi

for path in "$CHECKPOINT" "$MAIA_WEIGHTS"; do
  [[ -s "$path" ]] || { echo "error: missing $path" >&2; exit 1; }
done
[[ -x "$LC0_PATH" ]] || { echo "error: lc0 missing/not executable: $LC0_PATH" >&2; exit 1; }
mkdir -p "$OUT"

echo "============================================================"
echo " KIBITZER ADAPTIVE SEARCH GATE"
echo "============================================================"
echo "checkpoint:    $CHECKPOINT"
echo "opponents:     $OPPONENTS  (label:lc0_nodes)"
echo "games/seed:    $GAMES / $SEED, paired openings and colors"
echo "max plies:     $MAX_PLIES"
echo "control:       uniform $FIXED_SIMS sims"
echo "adaptive:      stages $ADAPTIVE_STAGES"
echo "uncertainty:   entropy>=$ENTROPY_THRESHOLD top2>=$TOP2_RATIO_THRESHOLD value_delta>=$VALUE_DELTA_THRESHOLD"
echo "budget:        average sims <= $MAX_AVG_SIMS"
echo "adopt:         score delta >= +0.03, or within -0.02 with >=25% lower search time"
echo "outputs:       $OUT"
echo

passed=0
tested=0
for opponent in $OPPONENTS; do
  elo="${opponent%%:*}"
  nodes="${opponent##*:}"
  fixed="$OUT/fixed_s${FIXED_SIMS}_vs${elo}_n${nodes}_g${GAMES}_seed${SEED}"
  adaptive="$OUT/adaptive_vs${elo}_n${nodes}_g${GAMES}_seed${SEED}"

  echo "------------------------------------------------------------"
  echo " OPPONENT $elo  LC0 NODES=$nodes  CONTROL"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CHECKPOINT" --maia-weights "$MAIA_WEIGHTS" --maia-elo "$elo" \
    --lc0-path "$LC0_PATH" --backend "$BACKEND" --maia-nodes "$nodes" \
    --games "$GAMES" --max-plies "$MAX_PLIES" --simulations "$FIXED_SIMS" --seed "$SEED" \
    --out-jsonl "$fixed.jsonl" --out-pgn "$fixed.pgn" --device "$DEVICE"

  echo "------------------------------------------------------------"
  echo " OPPONENT $elo  LC0 NODES=$nodes  ADAPTIVE"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CHECKPOINT" --maia-weights "$MAIA_WEIGHTS" --maia-elo "$elo" \
    --lc0-path "$LC0_PATH" --backend "$BACKEND" --maia-nodes "$nodes" \
    --games "$GAMES" --max-plies "$MAX_PLIES" --simulations "$FIXED_SIMS" --adaptive-stages "$ADAPTIVE_STAGES" \
    --entropy-threshold "$ENTROPY_THRESHOLD" --top2-ratio-threshold "$TOP2_RATIO_THRESHOLD" \
    --value-delta-threshold "$VALUE_DELTA_THRESHOLD" --seed "$SEED" \
    --out-jsonl "$adaptive.jsonl" --out-pgn "$adaptive.pgn" --device "$DEVICE"

  set +e
  uv run python scripts/compare_search_gates.py \
    --fixed "$fixed.jsonl" --adaptive "$adaptive.jsonl" \
    --max-average-sims "$MAX_AVG_SIMS" --seed "$SEED" \
    --out "$OUT/decision_vs${elo}_n${nodes}.json"
  verdict=$?
  set -e
  tested=$((tested + 1))
  if (( verdict == 0 )); then
    passed=$((passed + 1))
  fi
  echo
done

echo "============================================================"
echo " ADAPTIVE SEARCH FINAL VERDICT"
echo "============================================================"
echo "accepted opponents: $passed / $tested"
if (( passed >= 2 )); then
  echo "RESULT: ADOPT  adaptive search passed against at least two external levels"
else
  echo "RESULT: REJECT  keep uniform 512-sim PUCT"
fi
echo "ADAPTIVE_SEARCH_GATE_DONE -> $OUT"
