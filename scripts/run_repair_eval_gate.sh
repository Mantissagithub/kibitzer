#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

GAMES="${GAMES:-80}"
SIMS="${SIMS:-128}"
SEED="${SEED:-17}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
MAIA_ELO="${MAIA_ELO:-2700}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
MAIA_NODES="${MAIA_NODES:-1}"
DEVICE="${DEVICE:-cuda}"

CANDIDATE_NAME="${CANDIDATE_NAME:-tactical_repair}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
CANDIDATE_REPORT_DIR="${CANDIDATE_REPORT_DIR:-reports/tactical_repair}"

POLICY_NAME="${POLICY_NAME:-policy_regret_repair}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-runs/regret/policy_regret_repair.pt}"
POLICY_REPORT_DIR="${POLICY_REPORT_DIR:-reports/regret}"

COMP_NAME="${COMP_NAME:-comp_base}"
COMP_CHECKPOINT="${COMP_CHECKPOINT:-runs/scaling_shaw_comp/S2_shaw_142M_comp.pt}"
COMP_REPORT_DIR="${COMP_REPORT_DIR:-reports/regret}"

for checkpoint in "$CANDIDATE_CHECKPOINT" "$POLICY_CHECKPOINT" "$COMP_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "error: checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
done
if [[ ! -s "$MAIA_WEIGHTS" ]]; then
  echo "error: Maia/Leela weights are missing: $MAIA_WEIGHTS" >&2
  exit 1
fi
if [[ ! -x "$LC0_PATH" ]]; then
  echo "error: lc0 executable is missing or not executable: $LC0_PATH" >&2
  exit 1
fi

mkdir -p "$CANDIDATE_REPORT_DIR" "$POLICY_REPORT_DIR" "$COMP_REPORT_DIR"

run_gate() {
  local name="$1"
  local checkpoint="$2"
  local report_dir="$3"
  local stem="${name}_vs${MAIA_ELO}_s${SIMS}_g${GAMES}_seed${SEED}"
  local jsonl="$report_dir/${stem}.jsonl"
  local pgn="$report_dir/${stem}.pgn"
  local log="$report_dir/${stem}.log"

  echo "============================================================"
  echo " REPAIR EVAL GATE: $name"
  echo "============================================================"
  echo "checkpoint:       $checkpoint"
  echo "games/sims/seed:  $GAMES / $SIMS / $SEED"
  echo "opponent:         Maia/Leela-$MAIA_ELO nodes=$MAIA_NODES backend=$BACKEND"
  echo "jsonl:            $jsonl"
  echo "pgn:              $pgn"
  echo "log:              $log"
  echo
  echo "live monitor:"
  echo "  uv run python scripts/monitor_match_jsonl.py --path $jsonl --expected-games $GAMES --poll-seconds 30"
  echo

  # same tag means same gate; interrupted reruns should replace partial files.
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$checkpoint" \
    --maia-weights "$MAIA_WEIGHTS" \
    --maia-elo "$MAIA_ELO" \
    --lc0-path "$LC0_PATH" \
    --backend "$BACKEND" \
    --maia-nodes "$MAIA_NODES" \
    --games "$GAMES" \
    --simulations "$SIMS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out-jsonl "$jsonl" \
    --out-pgn "$pgn" 2>&1 | tee "$log"

  echo
  echo "summary:"
  uv run python scripts/monitor_match_jsonl.py --path "$jsonl" --expected-games "$GAMES" --once
  echo
}

echo "============================================================"
echo " KIBITZER 3-WAY REPAIR EVAL"
echo "============================================================"
echo "candidate:        $CANDIDATE_NAME -> $CANDIDATE_CHECKPOINT"
echo "policy baseline:  $POLICY_NAME -> $POLICY_CHECKPOINT"
echo "comp base:        $COMP_NAME -> $COMP_CHECKPOINT"
echo "gate:             games=$GAMES sims=$SIMS seed=$SEED"
echo

run_gate "$CANDIDATE_NAME" "$CANDIDATE_CHECKPOINT" "$CANDIDATE_REPORT_DIR"
run_gate "$POLICY_NAME" "$POLICY_CHECKPOINT" "$POLICY_REPORT_DIR"
run_gate "$COMP_NAME" "$COMP_CHECKPOINT" "$COMP_REPORT_DIR"

echo "============================================================"
echo " 3-WAY GATE DONE"
echo "============================================================"
echo "next:"
echo "  uv run python scripts/plot_regret_policy.py"
