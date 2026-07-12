#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# deep-search confirmation for the D65 EI checkpoint. the 128-sim gate produced
# more wins but a lower match score; this checks whether 512 sims recovers the
# lost draw control and turns that aggression into real strength.

CAND="${CAND:-reports/selfplay_ei/ei_hard_s512.pt}"
BASELINE_JSONL="${BASELINE_JSONL:-reports/sims_sweep/kibitzer_vs2700_s512_g40_seed23.jsonl}"
BASELINE_SCORE="${BASELINE_SCORE:-0.825}"
GAMES="${GAMES:-40}"
SIMS="${SIMS:-512}"
SEED="${SEED:-23}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
MAIA_ELO="${MAIA_ELO:-2700}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-reports/selfplay_ei}"
STEM="$OUT/ei_hard_vs${MAIA_ELO}_s${SIMS}_g${GAMES}_seed${SEED}"

mkdir -p "$OUT"
for f in "$CAND" "$MAIA_WEIGHTS" "$LC0_PATH"; do
  [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done
[[ -x "$LC0_PATH" ]] || { echo "error: lc0 not executable: $LC0_PATH" >&2; exit 1; }

echo "============================================================"
echo " SELF-PLAY EI 512-SIM CONFIRMATION GATE"
echo "============================================================"
echo "candidate:       $CAND"
echo "opponent:        Maia/Leela-$MAIA_ELO nodes=1 backend=$BACKEND"
echo "gate:            $GAMES games @ $SIMS sims, seed $SEED"
echo "baseline file:   $BASELINE_JSONL"
echo "baseline score:  $BASELINE_SCORE  (tactical R1 @512 sims, 40 games)"
echo "pass read:       > baseline = real deep-search improvement"
echo "flat read:       ~= baseline = style changed, strength did not"
echo "fail read:       < 0.800 = EI hurt even with deep search"
echo "jsonl:           $STEM.jsonl"
echo "pgn:             $STEM.pgn"
echo

uv run python scripts/maia_gauntlet.py \
  --checkpoint "$CAND" \
  --maia-weights "$MAIA_WEIGHTS" \
  --maia-elo "$MAIA_ELO" \
  --lc0-path "$LC0_PATH" \
  --backend "$BACKEND" \
  --maia-nodes 1 \
  --games "$GAMES" \
  --simulations "$SIMS" \
  --seed "$SEED" \
  --out-jsonl "$STEM.jsonl" \
  --out-pgn "$STEM.pgn" \
  --device "$DEVICE"

python3 - <<PY
import json, math
from pathlib import Path

path = Path("$STEM.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
w = d = l = 0
points = 0.0
for row in rows:
    score = float(row["score"])
    points += score
    if score == 1.0:
        w += 1
    elif score == 0.5:
        d += 1
    else:
        l += 1
n = len(rows)
rate = points / n if n else 0.0
elo = 2700 + 400 * math.log10(rate / (1.0 - rate)) if 0.0 < rate < 1.0 else float("inf")
baseline = float("$BASELINE_SCORE")

print()
print("============================================================")
print(" VERDICT")
print("============================================================")
print(f"candidate: W/D/L={w}/{d}/{l} score={rate:.3f} points={points:.1f}/{n}")
print(f"baseline:  score={baseline:.3f}  file=$BASELINE_JSONL")
print(f"implied proxy Elo: {elo:.0f}")
if rate > baseline:
    print("RESULT: BEATS 512-SIM BASELINE -> EI improved deep-search play.")
elif rate >= 0.800:
    print("RESULT: NEAR BASELINE -> style changed, no clean strength gain.")
else:
    print("RESULT: BELOW BASELINE -> reject EI for promotion.")
print(f"SELFPLAY_EI_512_GATE_DONE -> {path}")
PY
