#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

BASE_CHECKPOINT="${BASE_CHECKPOINT:-runs/tactical/tactical_repair.pt}"
OUT_DIR="${OUT_DIR:-runs/oracle_process_rl}"
REPORT_DIR="${REPORT_DIR:-reports/oracle_process_rl}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
DEVICE="${DEVICE:-cuda}"

NUM_GROUPS="${NUM_GROUPS:-8}"
GROUP_SIZE="${GROUP_SIZE:-4}"
SIMS="${SIMS:-512}"
OPPONENT_ELO="${OPPONENT_ELO:-2300}"
MAX_PLIES="${MAX_PLIES:-160}"
ENGINE_TIME="${ENGINE_TIME:-0.02}"
TEMP="${TEMP:-0.8}"
TEMP_PLIES="${TEMP_PLIES:-20}"
TEMP_LATE="${TEMP_LATE:-0.0}"

TEACHER_NODES="${TEACHER_NODES:-10000}"
MULTIPV="${MULTIPV:-4}"
WORKERS="${WORKERS:-4}"
REWARD_CLIP="${REWARD_CLIP:-0.5}"
PROCESS_WEIGHT="${PROCESS_WEIGHT:-0.25}"
TERMINAL_WEIGHT="${TERMINAL_WEIGHT:-1.0}"
GAMMA="${GAMMA:-0.99}"
MIN_REGRET="${MIN_REGRET:-0.05}"
MIN_ABS_ADVANTAGE="${MIN_ABS_ADVANTAGE:-0.1}"

BETA="${BETA:-0.1}"
DELTA="${DELTA:-0.1}"
MAX_TV_BASE="${MAX_TV_BASE:-0.08}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-31}"

RUN_GATES="${RUN_GATES:-0}"
RESUME="${RESUME:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
GATE_GAMES="${GATE_GAMES:-80}"
GATE_SIMS_LIST="${GATE_SIMS_LIST:-128 512}"
MAIA_WEIGHTS="${MAIA_WEIGHTS:-data/leela/t1-256x10-distilled.pb.gz}"
LC0_PATH="${LC0_PATH:-data/leela/lc0}"
BACKEND="${BACKEND:-cuda}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT_DIR="/tmp/kibitzer_oracle_process_rl_smoke/run"
  REPORT_DIR="/tmp/kibitzer_oracle_process_rl_smoke/report"
  DEVICE=cpu
  NUM_GROUPS=2
  GROUP_SIZE=3
  SIMS=0
  OPPONENT_ELO=1600
  MAX_PLIES=8
  ENGINE_TIME=0.001
  TEMP=1.0
  TEMP_PLIES=99
  TEACHER_NODES=64
  MULTIPV=2
  WORKERS=1
  MIN_REGRET=0.0
  MIN_ABS_ADVANTAGE=0.0
  BATCH_SIZE=16
  EPOCHS=1
  RUN_GATES=0
fi

if [[ ! -s "$BASE_CHECKPOINT" ]]; then
  echo "error: base checkpoint is missing: $BASE_CHECKPOINT" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$REPORT_DIR"

if [[ "$TRAIN_ONLY" == "1" ]]; then
  LABELED_DATA="$OUT_DIR/oracle_labeled.jsonl"
  if [[ ! -s "$LABELED_DATA" ]]; then
    echo "error: labeled oracle buffer is missing: $LABELED_DATA" >&2
    exit 1
  fi
  echo "============================================================"
  echo " ORACLE RL TRAIN-ONLY RETRY"
  echo "============================================================"
  echo "data:              $LABELED_DATA"
  echo "base/anchor:       $BASE_CHECKPOINT"
  echo "selection:         held-out signed advantage log-probability"
  echo "epoch snapshots:   $OUT_DIR/oracle_process_rl_epoch{1,2}.pt"
  echo "final checkpoint:  $OUT_DIR/oracle_process_rl.pt"
  echo
  uv run python scripts/train_oracle_process_rl.py train \
    --data "$LABELED_DATA" \
    --checkpoint "$BASE_CHECKPOINT" \
    --anchor "$BASE_CHECKPOINT" \
    --min-regret "$MIN_REGRET" \
    --min-abs-advantage "$MIN_ABS_ADVANTAGE" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --delta "$DELTA" \
    --beta "$BETA" \
    --max-tv-base "$MAX_TV_BASE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out "$OUT_DIR/oracle_process_rl.pt" \
    --metrics "$REPORT_DIR/training_metrics_signed_selector.jsonl" 2>&1 | tee -a "$REPORT_DIR/train.log"
  echo "ORACLE_PROCESS_RL_RETRAIN_DONE"
  exit 0
fi

echo "============================================================"
echo " ORACLE-SHAPED ON-POLICY RL"
echo "============================================================"
echo "base:              $BASE_CHECKPOINT"
echo "rollout:           $((NUM_GROUPS * GROUP_SIZE)) games @ $SIMS sims vs SF-$OPPONENT_ELO"
echo "teacher labels:    Stockfish nodes=$TEACHER_NODES multipv=$MULTIPV workers=$WORKERS"
echo "reward:            process=$PROCESS_WEIGHT terminal=$TERMINAL_WEIGHT gamma=$GAMMA clip=$REWARD_CLIP"
echo "filter:            regret>=$MIN_REGRET and |advantage|>=$MIN_ABS_ADVANTAGE"
echo "update:            policy+norm only, value/trunk frozen"
echo "trust controls:    base-KL beta=$BETA DPPO-TV delta=$DELTA max-base-TV=$MAX_TV_BASE"
echo "optimizer:         lr=$LEARNING_RATE batch=$BATCH_SIZE epochs=$EPOCHS"
echo "outputs:           $OUT_DIR"
echo "reports:           $REPORT_DIR"
echo "external gates:    $RUN_GATES  sims=[$GATE_SIMS_LIST] games=$GATE_GAMES each"
echo "resume rollout:    $RESUME"
echo
echo "what the stages mean:"
echo "  1. play fresh games with the current searched policy"
echo "  2. ask full-strength Stockfish how much each sampled move lost"
echo "  3. combine move-level process rewards with the final {-1,0,+1} result"
echo "  4. update only sampled-action probabilities under a frozen-base KL anchor"
echo

if [[ "${REUSE_TRAINED:-0}" != "1" ]]; then
  RESUME_ARGS=()
  TEE_ARGS=()
  if [[ "$RESUME" == "1" ]]; then
    RESUME_ARGS+=(--reuse-rollout)
    TEE_ARGS+=(-a)
  fi
  uv run python scripts/train_oracle_process_rl.py run \
    --checkpoint "$BASE_CHECKPOINT" \
    --stockfish "$STOCKFISH_PATH" \
    --opponent-elo "$OPPONENT_ELO" \
    --groups "$NUM_GROUPS" \
    --group-size "$GROUP_SIZE" \
    --sims "$SIMS" \
    --temp "$TEMP" \
    --temp-plies "$TEMP_PLIES" \
    --temp-late "$TEMP_LATE" \
    --max-plies "$MAX_PLIES" \
    --engine-time "$ENGINE_TIME" \
    --teacher-nodes "$TEACHER_NODES" \
    --multipv "$MULTIPV" \
    --workers "$WORKERS" \
    --reward-clip "$REWARD_CLIP" \
    --process-weight "$PROCESS_WEIGHT" \
    --terminal-weight "$TERMINAL_WEIGHT" \
    --gamma "$GAMMA" \
    --min-regret "$MIN_REGRET" \
    --min-abs-advantage "$MIN_ABS_ADVANTAGE" \
    --beta "$BETA" \
    --delta "$DELTA" \
    --max-tv-base "$MAX_TV_BASE" \
    --lr "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    --report-dir "$REPORT_DIR" \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${TEE_ARGS[@]}" "$REPORT_DIR/train.log"
else
  echo "reusing trained checkpoint: $OUT_DIR/oracle_process_rl.pt"
fi

CANDIDATE="$OUT_DIR/oracle_process_rl.pt"
if [[ ! -s "$CANDIDATE" ]]; then
  echo "error: candidate checkpoint is missing: $CANDIDATE" >&2
  exit 1
fi
if [[ "$RUN_GATES" != "1" ]]; then
  echo
  echo "training finished. external gates were not started."
  echo "run both paired gates with:"
  echo "  RUN_GATES=1 REUSE_TRAINED=1 bash scripts/run_oracle_process_rl.sh"
  echo "ORACLE_PROCESS_RL_TRAIN_DONE"
  exit 0
fi

if [[ ! -s "$MAIA_WEIGHTS" || ! -x "$LC0_PATH" ]]; then
  echo "error: gate requested but lc0 weights or executable are missing" >&2
  exit 1
fi

run_gate() {
  local name="$1"
  local checkpoint="$2"
  local sims="$3"
  local stem="${name}_vs2700_s${sims}_g${GATE_GAMES}_seed${SEED}"
  echo
  echo "============================================================"
  echo " EXTERNAL GATE: $name @ $sims SIMS"
  echo "============================================================"
  uv run python scripts/maia_gauntlet.py \
    --checkpoint "$checkpoint" \
    --maia-weights "$MAIA_WEIGHTS" \
    --maia-elo 2700 \
    --lc0-path "$LC0_PATH" \
    --backend "$BACKEND" \
    --maia-nodes 1 \
    --games "$GATE_GAMES" \
    --simulations "$sims" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --out-jsonl "$REPORT_DIR/${stem}.jsonl" \
    --out-pgn "$REPORT_DIR/${stem}.pgn" 2>&1 | tee "$REPORT_DIR/${stem}.log"
}

for sims in $GATE_SIMS_LIST; do
  run_gate oracle_process_rl "$CANDIDATE" "$sims"
  run_gate tactical_base "$BASE_CHECKPOINT" "$sims"
done

echo
echo "ORACLE_PROCESS_RL_GATE_DONE"
