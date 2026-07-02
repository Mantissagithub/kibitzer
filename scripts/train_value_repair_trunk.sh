#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export STAGE_LABEL="STAGE C — LAST-BLOCK VALUE REPAIR"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-runs/value_repair/value_repair_best.pt}"
export UNFREEZE_TRUNK_BLOCKS="${UNFREEZE_TRUNK_BLOCKS:-1}"
export TRUNK_LEARNING_RATE="${TRUNK_LEARNING_RATE:-5e-6}"
export HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
export NORM_LEARNING_RATE="${NORM_LEARNING_RATE:-2e-5}"
export POLICY_KL_WEIGHT="${POLICY_KL_WEIGHT:-1.0}"
export MAX_POLICY_KL="${MAX_POLICY_KL:-0.01}"
export MIN_POLICY_TOP1_AGREEMENT="${MIN_POLICY_TOP1_AGREEMENT:-0.98}"
export EPOCHS="${EPOCHS:-3}"
export OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-runs/value_repair_trunk/value_repair_trunk_best.pt}"
export HF_REPO="${HF_REPO:-Pradheep1647/kibitzer-clean-value-repair-trunk1}"

echo "Final bounded supervised repair experiment."
echo "If no epoch beats epoch 0, stop this checkpoint lineage."
echo

exec bash scripts/train_value_repair_norm.sh
