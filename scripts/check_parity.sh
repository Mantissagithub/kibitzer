#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# strength-parity check for batched search: run the SAME model vs the SAME lc0 opponent
# with the SAME seed/openings, once sequential (batch=1, the exact old search) and once
# batched (batch=32). if batching did not cost strength the two scores land within noise.
# both arms use 512 sims (the official-run sim count), so this validates the setting we ship.
#
#   bash scripts/check_parity.sh                 # 40 games/arm vs lc0-2700, ~1.5h
#   GAMES=60 SIMS=512 bash scripts/check_parity.sh

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
SIMS="${SIMS:-512}"
GAMES="${GAMES:-40}"
SEED="${SEED:-23}"
NODES="${NODES:-1}"                  # lc0-2700 at nodes=1 (the standard external gate)
ELO_LABEL="${ELO_LABEL:-2700}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-cuda}"
NET="${NET:-data/leela/t1-256x10-distilled.pb.gz}"
LC0="${LC0:-data/leela/lc0}"
OUT="${OUT:-reports/parity}"
TOL="${TOL:-0.05}"                   # allowed |batched - sequential| score gap

for f in "$CHECKPOINT" "$NET"; do [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }; done
[[ -x "$LC0" ]] || { echo "error: lc0 not executable: $LC0" >&2; exit 1; }
mkdir -p "$OUT"

run_arm() {  # $1 = batch_size, $2 = label
  local bs="$1" label="$2"
  local stem="$OUT/${label}_s${SIMS}_b${bs}_vs${ELO_LABEL}_g${GAMES}_seed${SEED}"
  echo "------------------------------------------------------------"
  echo " ARM: $label   (batch=$bs, $SIMS sims, $GAMES games vs lc0-$ELO_LABEL)"
  echo "------------------------------------------------------------"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$CHECKPOINT" --maia-weights "$NET" --maia-elo "$ELO_LABEL" \
    --lc0-path "$LC0" --backend "$BACKEND" --maia-nodes "$NODES" \
    --games "$GAMES" --simulations "$SIMS" --batch-size "$bs" --seed "$SEED" \
    --out-jsonl "$stem.jsonl" --out-pgn "$stem.pgn" --device "$DEVICE"
  python3 -c "
import json
s=[json.loads(l)['score'] for l in open('$stem.jsonl') if l.strip()]
print(f'{sum(s)/len(s):.4f}' if s else '0')
" > "$stem.score"
}

run_arm 1  sequential
run_arm 32 batched

SEQ=$(cat "$OUT/sequential_s${SIMS}_b1_vs${ELO_LABEL}_g${GAMES}_seed${SEED}.score")
BAT=$(cat "$OUT/batched_s${SIMS}_b32_vs${ELO_LABEL}_g${GAMES}_seed${SEED}.score")

echo
echo "============================================================"
echo " PARITY RESULT   (vs lc0-$ELO_LABEL, $GAMES games/arm, $SIMS sims, seed $SEED)"
echo "============================================================"
printf "  sequential (batch=1) : %s\n" "$SEQ"
printf "  batched    (batch=32): %s\n" "$BAT"
python3 -c "
seq, bat, tol = $SEQ, $BAT, $TOL
d = bat - seq
verdict = 'PARITY OK (batching is safe)' if abs(d) <= tol else 'REGRESSED / DRIFT (investigate before trusting batched numbers)'
print(f'  delta (batched - sequential): {d:+.4f}   tolerance +/-{tol}')
print(f'  -> {verdict}')
"
echo "note: ~$GAMES games/arm has ~+/-0.05-0.08 sampling noise; a small delta within tolerance is expected."