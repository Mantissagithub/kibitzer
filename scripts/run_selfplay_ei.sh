#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# one clean iteration of 512-sim hard-target self-play expert iteration (DECISIONS.md D65).
# the whole bet: 512-sim search is the strong teacher (D63), so we (1) self-play at 512 sims,
# (2) train the policy head toward the move search actually chose (hard argmax, keeps the
# decisiveness that soft targets diluted in D49), value head frozen, weak KL-to-base anchor,
# then (3) gate the result vs an EXTERNAL Leela/Maia-2700 opponent. h2h vs its own base is not
# the signal (D53/D54). pass = beat the 0.294 baseline by >= +0.03.
#
#   bash scripts/run_selfplay_ei.sh            # real run (overnight, holds the GPU)
#   SMOKE=1 bash scripts/run_selfplay_ei.sh    # tiny cpu plumbing check

BASE="${BASE:-runs/tactical/tactical_repair.pt}"     # frozen base = teacher start + kl anchor
SIMS="${SIMS:-512}"                                  # teacher strength for self-play (the point)
GEN_GAMES="${GEN_GAMES:-200}"                         # self-play games to generate
TEMP_PLIES="${TEMP_PLIES:-20}"                        # opening moves sampled for diversity
MAX_PLIES="${MAX_PLIES:-160}"
GEN_SEED="${GEN_SEED:-7}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-128}"
KL_BETA="${KL_BETA:-0.05}"
DEVICE="${DEVICE:-cuda}"

BASELINE="${BASELINE:-0.294}"                         # tactical R1 external-gate score @128 sims
MARGIN="${MARGIN:-0.03}"                              # required improvement (~+25 Elo)
GATE_GAMES="${GATE_GAMES:-80}"
GATE_SIMS="${GATE_SIMS:-128}"                         # 128 = apples-to-apples with BASELINE
GATE_SEED="${GATE_SEED:-23}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
MAIA_ELO="${MAIA_ELO:-2700}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"

OUT="${OUT:-reports/selfplay_ei}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  SIMS="${SMOKE_SIMS:-16}"; GEN_GAMES=4; TEMP_PLIES=4; MAX_PLIES=40
  EPOCHS=1; BATCH=32; GATE_GAMES=2; GATE_SIMS=16; DEVICE=cpu; BACKEND=eigen
  OUT="reports/selfplay_ei_smoke"
fi

DATA="$OUT/selfplay_s${SIMS}_g${GEN_GAMES}.jsonl"
CAND="$OUT/ei_hard_s${SIMS}.pt"
GATE_STEM="$OUT/ei_hard_vs${MAIA_ELO}_s${GATE_SIMS}_g${GATE_GAMES}_seed${GATE_SEED}"
mkdir -p "$OUT"

for f in "$BASE" "$MAIA_WEIGHTS"; do [[ -s "$f" ]] || { echo "error: missing $f" >&2; exit 1; }; done
[[ -x "$LC0_PATH" ]] || { echo "error: lc0 not executable: $LC0_PATH" >&2; exit 1; }

echo "============================================================"
echo " 512-SIM HARD-TARGET SELF-PLAY EXPERT ITERATION  (D65)"
echo "============================================================"
echo "base/teacher:  $BASE"
echo "self-play:     $GEN_GAMES games @ $SIMS sims  (dirichlet + temp $TEMP_PLIES)"
echo "train:         hard argmax targets | policy-head+norm only | value FROZEN | kl-base beta=$KL_BETA | lr $LR x$EPOCHS"
echo "gate:          vs Maia/Leela-$MAIA_ELO, $GATE_GAMES games @ $GATE_SIMS sims, seed $GATE_SEED"
echo "pass if:       gate_score >= $BASELINE + $MARGIN  (= $(python3 -c "print(f'{$BASELINE+$MARGIN:.3f}')"))"
echo "device:        $DEVICE"
echo "outputs ->     $OUT"
echo

echo "------------------------------------------------------------"
echo " STAGE 1/3  self-play generation @ $SIMS sims  (the slow part)"
echo "------------------------------------------------------------"
uv run python scripts/selfplay_az.py gen \
  --checkpoint "$BASE" --games "$GEN_GAMES" --sims "$SIMS" \
  --temp-plies "$TEMP_PLIES" --max-plies "$MAX_PLIES" --seed "$GEN_SEED" \
  --out-jsonl "$DATA" --device "$DEVICE"
POS=$(wc -l < "$DATA")
echo "-> $POS training positions written to $DATA"
echo

echo "------------------------------------------------------------"
echo " STAGE 2/3  train policy head on hard 512-sim targets"
echo "------------------------------------------------------------"
uv run python scripts/selfplay_az.py train \
  --checkpoint "$BASE" --data "$DATA" \
  --hard-targets --freeze-nonpolicy --kl-base "$BASE" --kl-beta "$KL_BETA" \
  --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH" \
  --out "$CAND" --device "$DEVICE"
echo "-> candidate checkpoint: $CAND"
echo

echo "------------------------------------------------------------"
echo " STAGE 3/3  EXTERNAL gate vs Maia/Leela-$MAIA_ELO @ $GATE_SIMS sims"
echo "------------------------------------------------------------"
uv run python scripts/maia_gauntlet.py \
  --checkpoint "$CAND" --maia-weights "$MAIA_WEIGHTS" --maia-elo "$MAIA_ELO" \
  --lc0-path "$LC0_PATH" --backend "$BACKEND" --maia-nodes 1 \
  --games "$GATE_GAMES" --simulations "$GATE_SIMS" --seed "$GATE_SEED" \
  --out-jsonl "$GATE_STEM.jsonl" --out-pgn "$GATE_STEM.pgn" --device "$DEVICE"

SCORE=$(python3 -c "
import json,sys
s=[json.loads(l)['score'] for l in open('$GATE_STEM.jsonl') if l.strip()]
print(f'{sum(s)/len(s):.3f}' if s else '0.000')
")
THRESH=$(python3 -c "print(f'{$BASELINE+$MARGIN:.3f}')")
echo
echo "============================================================"
echo " VERDICT"
echo "============================================================"
echo "candidate gate score:  $SCORE   ($GATE_GAMES games vs $MAIA_ELO @ $GATE_SIMS sims)"
echo "baseline (tactical R1): $BASELINE"
echo "pass threshold:        $THRESH  (baseline + $MARGIN)"
if python3 -c "import sys; sys.exit(0 if $SCORE >= $THRESH else 1)"; then
  echo "RESULT: PASS  -> 512-sim search DID distill into the weights. promote + consider iter 2."
else
  if python3 -c "import sys; sys.exit(0 if $SCORE < $BASELINE else 1)"; then
    echo "RESULT: REGRESSED (< baseline) -> distillation hurt. 10th self-play no-gain."
  else
    echo "RESULT: FLAT (within noise) -> 512-sim strength is inference-only, not distillable. 10th no-gain."
  fi
fi
echo "gate pgn/jsonl: $GATE_STEM.{pgn,jsonl}"
echo "SELFPLAY_EI_DONE -> $SCORE vs threshold $THRESH"
