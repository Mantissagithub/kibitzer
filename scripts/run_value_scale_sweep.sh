#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1

# D62: interp-motivated value_scale sweep on the EXTERNAL Leela-2700 gate. the
# interp showed the value head is noisy/late; puct backs it up at full weight and
# the gate hardcodes value_scale=1.0. this sweeps value_scale down (trust the value
# less, lean on the policy prior) and prints a clean per-setting + summary log.
# value_scale=1.0 is the control and should reproduce the ~0.294 reference.
#
# optional SF-ladder cross-check: LADDER=1 bash scripts/run_value_scale_sweep.sh

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
VALUE_SCALES="${VALUE_SCALES:-1.0 0.75 0.5 0.25 0.0}"   # control first
GAMES="${GAMES:-40}"
SIMS="${SIMS:-128}"
SEED="${SEED:-23}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
LADDER_ELOS="${LADDER_ELOS:-1900 2300}"
LADDER_GAMES="${LADDER_GAMES:-20}"
OUT="${OUT:-reports/value_scale_sweep}"
DEVICE="${DEVICE:-cuda}"

for f in "$CHECKPOINT" "$MAIA_WEIGHTS"; do
  [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done
[[ -x "$LC0_PATH" ]] || { echo "error: lc0 missing/not executable: $LC0_PATH" >&2; exit 1; }
mkdir -p "$OUT"

tag() { echo "vs$(echo "$1" | tr -d '.')"; }   # 0.75 -> vs075, 1.0 -> vs10

echo "============================================================"
echo " D62  VALUE_SCALE SWEEP vs Leela-2700  (down-weight the noisy value head)"
echo "============================================================"
echo "checkpoint:    $CHECKPOINT"
echo "value_scales:  $VALUE_SCALES   (1.0 = control, reference score ~0.294)"
echo "gate:          $GAMES games @ $SIMS sims, seed $SEED, opponent Leela-2700 (nodes 1, $BACKEND)"
echo "outputs:       $OUT"
echo "decision:      a value_scale<1.0 beating 1.0 by >= +0.03 is a real (free) gain"
echo

for vs in $VALUE_SCALES; do
  t="$(tag "$vs")"
  jsonl="$OUT/kibitzer_vs2700_s${SIMS}_g${GAMES}_seed${SEED}_${t}.jsonl"
  pgn="$OUT/kibitzer_vs2700_s${SIMS}_g${GAMES}_seed${SEED}_${t}.pgn"
  echo "------------------------------------------------------------"
  echo " value_scale = $vs   $( [[ "$vs" == "1.0" ]] && echo '(control)' || echo '' )"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CHECKPOINT" \
    --maia-weights "$MAIA_WEIGHTS" \
    --maia-elo 2700 \
    --lc0-path "$LC0_PATH" \
    --backend "$BACKEND" \
    --maia-nodes 1 \
    --games "$GAMES" \
    --simulations "$SIMS" \
    --value-scale "$vs" \
    --seed "$SEED" \
    --out-jsonl "$jsonl" \
    --out-pgn "$pgn" \
    --device "$DEVICE"
  echo
done

# optional stockfish-ladder cross-check (LADDER=1)
if [[ "${LADDER:-0}" == "1" && -n "$STOCKFISH_PATH" ]]; then
  echo "============================================================"
  echo " SF-LADDER CROSS-CHECK ($LADDER_GAMES games/elo @ $SIMS sims)"
  echo "============================================================"
  for elo in $LADDER_ELOS; do
    for vs in $VALUE_SCALES; do
      t="$(tag "$vs")"
      out="$OUT/sf${elo}_s${SIMS}_g${LADDER_GAMES}_${t}.json"
      echo "--- SF-$elo  value_scale=$vs ---"
      uv run python scripts/eval_search_vs_stockfish.py \
        --checkpoint "$CHECKPOINT" --out "$out" \
        --games "$LADDER_GAMES" --simulations "$SIMS" --value-scale "$vs" \
        --stockfish-path "$STOCKFISH_PATH" --stockfish-elo "$elo" --device "$DEVICE"
    done
  done
fi

# ---- final comparison table (the thing to read) ----
echo
echo "============================================================"
echo " SWEEP SUMMARY  (vs Leela-2700, control value_scale=1.0 ~ 0.294)"
echo "============================================================"
printf "%-13s %-10s %-8s %-8s\n" "value_scale" "W/D/L" "score" "impliedElo"
for vs in $VALUE_SCALES; do
  t="$(tag "$vs")"
  jsonl="$OUT/kibitzer_vs2700_s${SIMS}_g${GAMES}_seed${SEED}_${t}.jsonl"
  line="$(uv run python scripts/monitor_match_jsonl.py --path "$jsonl" --expected-games "$GAMES" --once 2>/dev/null || echo '?')"
  # line looks like: "40/40 games: 12W/20D/48L score=22.0 rate=0.275 ... elo=2532"
  wdl="$(echo "$line" | grep -oE '[0-9]+W/[0-9]+D/[0-9]+L' || echo '?')"
  rate="$(echo "$line" | grep -oE 'rate=[0-9.]+' | cut -d= -f2 || echo '?')"
  elo="$(echo "$line" | grep -oE 'elo=[0-9]+' | tail -1 | cut -d= -f2 || echo '?')"
  mark="$( [[ "$vs" == "1.0" ]] && echo ' <- control' || echo '' )"
  printf "%-13s %-10s %-8s %-8s%s\n" "$vs" "$wdl" "$rate" "$elo" "$mark"
done
echo
echo "read: any value_scale beating the 1.0 control's rate by >= 0.03 is a free gain."
echo "VALUE_SCALE_SWEEP_DONE"
