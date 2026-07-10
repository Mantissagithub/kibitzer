#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1

# D63: search simulation-count sweep on the EXTERNAL Leela-2700 gate. the gate normally
# runs 128 sims; this asks whether more search buys strength or whether it plateaus
# because the value leaf is weak (value-capped). no code change, just varies
# --simulations. 128 is the control and should reproduce the ~0.294 reference.
#
# optional SF-ladder cross-check: LADDER=1 bash scripts/run_sims_sweep.sh

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
SIMS_LIST="${SIMS_LIST:-128 64 256 512}"   # control (128) first
GAMES="${GAMES:-40}"
SEED="${SEED:-23}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
LADDER_ELOS="${LADDER_ELOS:-1900 2300}"
LADDER_GAMES="${LADDER_GAMES:-20}"
OUT="${OUT:-reports/sims_sweep}"
DEVICE="${DEVICE:-cuda}"

for f in "$CHECKPOINT" "$MAIA_WEIGHTS"; do
  [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done
[[ -x "$LC0_PATH" ]] || { echo "error: lc0 missing/not executable: $LC0_PATH" >&2; exit 1; }
mkdir -p "$OUT"

echo "============================================================"
echo " D63  SIMULATION-COUNT SWEEP vs Leela-2700  (does search have headroom?)"
echo "============================================================"
echo "checkpoint:  $CHECKPOINT"
echo "sims:        $SIMS_LIST   (128 = control, reference score ~0.294)"
echo "gate:        $GAMES games, seed $SEED, opponent Leela-2700 (nodes 1, $BACKEND)"
echo "outputs:     $OUT"
echo "read:        rising score = free strength (use more sims); flat = value-capped ceiling"
echo

for sims in $SIMS_LIST; do
  jsonl="$OUT/kibitzer_vs2700_s${sims}_g${GAMES}_seed${SEED}.jsonl"
  pgn="$OUT/kibitzer_vs2700_s${sims}_g${GAMES}_seed${SEED}.pgn"
  echo "------------------------------------------------------------"
  echo " simulations = $sims   $( [[ "$sims" == "128" ]] && echo '(control)' || echo '' )"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CHECKPOINT" \
    --maia-weights "$MAIA_WEIGHTS" \
    --maia-elo 2700 \
    --lc0-path "$LC0_PATH" \
    --backend "$BACKEND" \
    --maia-nodes 1 \
    --games "$GAMES" \
    --simulations "$sims" \
    --seed "$SEED" \
    --out-jsonl "$jsonl" \
    --out-pgn "$pgn" \
    --device "$DEVICE"
  echo
done

if [[ "${LADDER:-0}" == "1" && -n "$STOCKFISH_PATH" ]]; then
  echo "============================================================"
  echo " SF-LADDER CROSS-CHECK ($LADDER_GAMES games/elo)"
  echo "============================================================"
  for elo in $LADDER_ELOS; do
    for sims in $SIMS_LIST; do
      out="$OUT/sf${elo}_s${sims}_g${LADDER_GAMES}.json"
      echo "--- SF-$elo  sims=$sims ---"
      uv run python scripts/eval_search_vs_stockfish.py \
        --checkpoint "$CHECKPOINT" --out "$out" \
        --games "$LADDER_GAMES" --simulations "$sims" \
        --stockfish-path "$STOCKFISH_PATH" --stockfish-elo "$elo" --device "$DEVICE"
    done
  done
fi

echo
echo "============================================================"
echo " SWEEP SUMMARY  (vs Leela-2700, control sims=128 ~ 0.294)"
echo "============================================================"
printf "%-8s %-10s %-8s %-8s\n" "sims" "W/D/L" "score" "impliedElo"
for sims in $SIMS_LIST; do
  jsonl="$OUT/kibitzer_vs2700_s${sims}_g${GAMES}_seed${SEED}.jsonl"
  line="$(uv run python scripts/monitor_match_jsonl.py --path "$jsonl" --expected-games "$GAMES" --once 2>/dev/null || echo '?')"
  wdl="$(echo "$line" | grep -oE '[0-9]+W/[0-9]+D/[0-9]+L' || echo '?')"
  rate="$(echo "$line" | grep -oE 'rate=[0-9.]+' | cut -d= -f2 || echo '?')"
  elo="$(echo "$line" | grep -oE 'elo=[0-9]+' | tail -1 | cut -d= -f2 || echo '?')"
  mark="$( [[ "$sims" == "128" ]] && echo ' <- control' || echo '' )"
  printf "%-8s %-10s %-8s %-8s%s\n" "$sims" "$wdl" "$rate" "$elo" "$mark"
done
echo
echo "read: sims beating the 128 control by >= 0.03 = free strength; flat = search is value-capped."
echo "SIMS_SWEEP_DONE"
