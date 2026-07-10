#!/usr/bin/env bash

set -euo pipefail

# raw interp capture only. plots and videos come after we inspect the measurements.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
PGN="${PGN:-reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.pgn}"
JSONL="${JSONL:-reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.jsonl}"
OUT_DIR="${OUT_DIR:-interp/data}"
DEVICE="${DEVICE:-auto}"

for path in "$CHECKPOINT" "$PGN" "$JSONL"; do
    if [[ ! -f "$path" ]]; then
        echo "missing required input: $path" >&2
        exit 1
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
    DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/analysis_${RUN_ID}.log"
MANIFEST="$OUT_DIR/analysis_${RUN_ID}.txt"

{
    echo "run_id=$RUN_ID"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "checkpoint=$CHECKPOINT"
    echo "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
    echo "pgn=$PGN"
    echo "jsonl=$JSONL"
    echo "games=$(wc -l < "$JSONL")"
    echo "device=$DEVICE"
    echo "out_dir=$OUT_DIR"
} > "$MANIFEST"

echo "============================================================"
echo " KIBITZER INTERPRETABILITY CAPTURE"
echo "============================================================"
echo "checkpoint:  $CHECKPOINT"
echo "source games: $(wc -l < "$JSONL")"
echo "capture:      first win, draw, and loss exemplar"
echo "device:      $DEVICE"
echo "results:     $OUT_DIR"
echo "manifest:    $MANIFEST"
echo "log:         $LOG"
echo

set +e
uv run python interp/analyze.py \
    --checkpoint "$CHECKPOINT" \
    --pgn "$PGN" \
    --jsonl "$JSONL" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

{
    echo "exit_status=$STATUS"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$MANIFEST"

if [[ "$STATUS" -ne 0 ]]; then
    echo
    echo "capture failed with exit status $STATUS"
    echo "inspect: $LOG"
    exit "$STATUS"
fi

echo
echo "INTERP_CAPTURE_DONE"
echo "summary:  $OUT_DIR/summary.json"
echo "raw data: $OUT_DIR/{win,draw,loss}.npz"
echo "per-ply:  $OUT_DIR/{win,draw,loss}_plies.json"
