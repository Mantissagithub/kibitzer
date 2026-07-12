#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# gate a checkpoint vs the external Maia/Leela-2700 opponent at one OR MORE sim counts,
# each scored against the BASE model's own score at that sim count (D63 sims_sweep). this
# is why we can gate at 512 too: the base already scores 0.825 at 512 (near ceiling vs
# maia), so 512 is mostly a "did it regress" check; 128 (base 0.287) is where a real gain
# actually shows because there is headroom. prints a score + verdict per sim count.
#
#   bash scripts/gate_sims.sh <candidate.pt> [outdir]
#   GATE_SIMS_LIST="128 512" bash scripts/gate_sims.sh runs/x.pt reports/selfplay_ei

CAND="${1:?usage: gate_sims.sh <candidate.pt> [outdir]}"
OUT="${2:-reports/selfplay_ei}"
GATE_SIMS_LIST="${GATE_SIMS_LIST:-128 512}"
GATE_SEED="${GATE_SEED:-23}"
MARGIN="${MARGIN:-0.03}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
MAIA_ELO="${MAIA_ELO:-2700}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
DEVICE="${DEVICE:-cuda}"

# base-model score per sim count (from reports/sims_sweep, seed 23, 40 games) + games to play.
base_score() { case "$1" in 64) echo 0.087;; 128) echo 0.287;; 256) echo 0.325;; 512) echo 0.825;; *) echo 0.500;; esac; }
gate_games() { case "$1" in 128) echo "${GATE_GAMES_128:-80}";; *) echo "${GATE_GAMES_OTHER:-40}";; esac; }

mkdir -p "$OUT"
[[ -s "$CAND" ]] || { echo "error: missing candidate $CAND" >&2; exit 1; }

declare -A RESULT
for sims in $GATE_SIMS_LIST; do
  games="$(gate_games "$sims")"
  baseline="$(base_score "$sims")"
  thresh="$(python3 -c "print(f'{$baseline+$MARGIN:.3f}')")"
  stem="$OUT/ei_hard_vs${MAIA_ELO}_s${sims}_g${games}_seed${GATE_SEED}"
  echo "------------------------------------------------------------"
  echo " GATE @ $sims sims : $games games vs Maia/Leela-$MAIA_ELO"
  echo "   base baseline $baseline  ->  pass if candidate >= $thresh"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CAND" --maia-weights "$MAIA_WEIGHTS" --maia-elo "$MAIA_ELO" \
    --lc0-path "$LC0_PATH" --backend "$BACKEND" --maia-nodes 1 \
    --games "$games" --simulations "$sims" --seed "$GATE_SEED" \
    --out-jsonl "$stem.jsonl" --out-pgn "$stem.pgn" --device "$DEVICE"
  score="$(python3 -c "
import json
s=[json.loads(l)['score'] for l in open('$stem.jsonl') if l.strip()]
print(f'{sum(s)/len(s):.3f}' if s else '0.000')
")"
  if python3 -c "import sys; sys.exit(0 if $score >= $thresh else 1)"; then
    v="PASS  (+beats base $baseline)"
  elif python3 -c "import sys; sys.exit(0 if $score < $baseline else 1)"; then
    v="REGRESSED (below base $baseline)"
  else
    v="FLAT  (within noise of base $baseline)"
  fi
  RESULT[$sims]="score $score  vs thresh $thresh  ->  $v"
done

echo
echo "============================================================"
echo " MULTI-SIM GATE SUMMARY   candidate: $CAND"
echo "============================================================"
for sims in $GATE_SIMS_LIST; do
  printf "  %4s sims : %s\n" "$sims" "${RESULT[$sims]}"
done
echo "read: 128 sims is the sensitive test (base 0.287, headroom); 512 sims is the"
echo "real-strength / no-regression check (base already 0.825, near ceiling vs Maia)."
