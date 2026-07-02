#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export ACTION=validate
export CANDIDATE_NAME="${CANDIDATE_NAME:-value_repair}"
export CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-runs/value_repair/value_repair_best.pt}"
export VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-runs/diagnostics/value_repair_validation.json}"
export VALUE_SCALES="${VALUE_SCALES:-0,0.5,1}"

if [[ ! -s "$CANDIDATE_CHECKPOINT" ]]; then
  echo "error: repaired value checkpoint is missing: $CANDIDATE_CHECKPOINT" >&2
  exit 1
fi

echo "This reuses the common-oracle validation split."
echo "The locked test split will not be read or modified."
echo

exec bash scripts/run_search_diagnostics.sh
