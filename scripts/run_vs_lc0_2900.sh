#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# batched model @1024 sims vs a strong lc0 opponent (~2900). lc0 strength is set by node
# count on the t1-256x10 distilled net; the calibration from run_adaptive_search_gate.sh is
# 2700:1, 2850:8, 2950:32, so ~2900 ~= nodes 16. every model move uses leaf-parallel search
# (batch 64), so a 1024-sim move is ~1.1s instead of the old serial ~3s.
#
#   bash scripts/run_vs_lc0_2900.sh                 # 1024 sims vs lc0 ~2900, 40 games
#   GAMES=60 NODES=32 ELO_LABEL=2950 bash scripts/run_vs_lc0_2900.sh   # push toward ~2950

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
SIMS="${SIMS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NODES="${NODES:-16}"                 # lc0 nodes -> strength (2700:1  2850:8  2950:32); 16 ~= 2900
ELO_LABEL="${ELO_LABEL:-2900}"       # label only, for output naming
GAMES="${GAMES:-40}"                 # both colors
SEED="${SEED:-23}"
MAX_PLIES="${MAX_PLIES:-200}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-cuda}"           # lc0 backend; at 16 nodes eigen is also instant
NET="${NET:-data/leela/t1-256x10-distilled.pb.gz}"
LC0="${LC0:-data/leela/lc0}"
OUT="${OUT:-reports/vs_lc0_2900}"

for f in "$CHECKPOINT" "$NET"; do [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }; done
[[ -x "$LC0" ]] || { echo "error: lc0 not executable: $LC0" >&2; exit 1; }
mkdir -p "$OUT"
STEM="$OUT/kibitzer_s${SIMS}_b${BATCH_SIZE}_vs${ELO_LABEL}_n${NODES}_g${GAMES}_seed${SEED}"

echo "============================================================"
echo " BATCHED MODEL @${SIMS} sims  vs  lc0-${ELO_LABEL} (nodes=$NODES)"
echo "============================================================"
echo "model:    $CHECKPOINT  sims=$SIMS batch=$BATCH_SIZE device=$DEVICE"
echo "opponent: lc0 $(basename "$NET")  nodes=$NODES backend=$BACKEND  (~${ELO_LABEL} elo)"
echo "games:    $GAMES (both colors), seed $SEED"
echo "calib:    lc0 nodes->elo (run_adaptive_search_gate): 2700:1  2850:8  2950:32"
echo "out ->    $STEM.{jsonl,pgn}"
echo

uv run python scripts/maia_gauntlet.py \
  --checkpoint "$CHECKPOINT" --maia-weights "$NET" --maia-elo "$ELO_LABEL" \
  --lc0-path "$LC0" --backend "$BACKEND" --maia-nodes "$NODES" \
  --games "$GAMES" --simulations "$SIMS" --batch-size "$BATCH_SIZE" \
  --max-plies "$MAX_PLIES" --seed "$SEED" \
  --out-jsonl "$STEM.jsonl" --out-pgn "$STEM.pgn" --device "$DEVICE"

score=$(python3 -c "
import json
s=[json.loads(l)['score'] for l in open('$STEM.jsonl') if l.strip()]
print(f'{sum(s)/len(s):.3f} over {len(s)} games' if s else 'no games')
")
echo
echo "============================================================"
echo " RESULT vs lc0-${ELO_LABEL} (nodes=$NODES):  $score"
echo "============================================================"
echo "pgn/jsonl: $STEM.{pgn,jsonl}"
