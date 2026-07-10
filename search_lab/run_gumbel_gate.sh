#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"
DEVICE="${DEVICE:-cuda}"
MAIA_ELO="${MAIA_ELO:-2700}"
MAIA_NODES="${MAIA_NODES:-1}"
GAMES="${GAMES:-40}"
SIMS="${SIMS:-128}"
MAX_PLIES="${MAX_PLIES:-200}"
SEED="${SEED:-23}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
GUMBEL_SCALE="${GUMBEL_SCALE:-0}"
RESULT_DIR="${RESULT_DIR:-search_lab/results/gumbel}"

for path in "$CHECKPOINT" "$MAIA_WEIGHTS"; do
  if [[ ! -s "$path" ]]; then
    echo "error: missing file: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$LC0_PATH" ]]; then
  echo "error: lc0 is missing or not executable: $LC0_PATH" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"

PUCT_STEM="puct_vs${MAIA_ELO}_s${SIMS}_g${GAMES}_seed${SEED}"
GUMBEL_STEM="gumbel_vs${MAIA_ELO}_s${SIMS}_g${GAMES}_seed${SEED}"
PUCT_JSONL="$RESULT_DIR/${PUCT_STEM}.jsonl"
GUMBEL_JSONL="$RESULT_DIR/${GUMBEL_STEM}.jsonl"

echo "============================================================"
echo " KIBITZER GUMBEL SEARCH A/B"
echo "============================================================"
echo "model:            $CHECKPOINT"
echo "opponent:         Leela-$MAIA_ELO nodes=$MAIA_NODES"
echo "paired gate:      $GAMES games each, $SIMS net evals, seed $SEED"
echo "gumbel:           max_actions=$MAX_ACTIONS scale=$GUMBEL_SCALE"
echo "results:          $RESULT_DIR"
echo
echo "this run does not train or modify the checkpoint."
echo

echo "[1/3] SEARCH UNIT TESTS"
uv run pytest -q tests/test_gumbel_search.py
echo

run_gate() {
  local search="$1"
  local stem="$2"
  local jsonl="$RESULT_DIR/${stem}.jsonl"
  local pgn="$RESULT_DIR/${stem}.pgn"
  local log="$RESULT_DIR/${stem}.log"

  uv run python -m search_lab.gumbel_gauntlet \
    --search "$search" \
    --checkpoint "$CHECKPOINT" \
    --maia-weights "$MAIA_WEIGHTS" \
    --maia-elo "$MAIA_ELO" \
    --lc0-path "$LC0_PATH" \
    --backend "$BACKEND" \
    --maia-nodes "$MAIA_NODES" \
    --games "$GAMES" \
    --simulations "$SIMS" \
    --max-plies "$MAX_PLIES" \
    --max-actions "$MAX_ACTIONS" \
    --gumbel-scale "$GUMBEL_SCALE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out-jsonl "$jsonl" \
    --out-pgn "$pgn" 2>&1 | tee "$log"
}

echo "[2/3] CONTROL: NORMAL PUCT"
run_gate puct "$PUCT_STEM"
echo

echo "[3/3] CANDIDATE: GUMBEL + SEQUENTIAL HALVING"
run_gate gumbel "$GUMBEL_STEM"
echo

uv run python -m search_lab.compare_gumbel_gate \
  --puct "$PUCT_JSONL" \
  --gumbel "$GUMBEL_JSONL" \
  --opponent-elo "$MAIA_ELO"

uv run python -m search_lab.plot_gumbel_gate \
  --puct "$PUCT_JSONL" \
  --gumbel "$GUMBEL_JSONL" \
  --opponent-elo "$MAIA_ELO" \
  --output-dir "$RESULT_DIR"

echo
echo "80-game confirmation, only if the verdict is PROMISING:"
echo "  GAMES=80 bash search_lab/run_gumbel_gate.sh"
