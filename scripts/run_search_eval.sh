#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:The pynvml package is deprecated:FutureWarning}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-runs/value/value_final.pt}"
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
GAMES="${GAMES:-10}"
SIMULATIONS="${SIMULATIONS:-64}"
STOCKFISH_ELO="${STOCKFISH_ELO:-1320}"
STOCKFISH_TIME="${STOCKFISH_TIME:-0.05}"
MAX_PLIES="${MAX_PLIES:-200}"
EVAL_OUT="${EVAL_OUT:-eval_pgns/search_vs_stockfish_${STOCKFISH_ELO}.pgn}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: project Python does not exist: $PYTHON" >&2
  exit 1
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "error: value checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi
if [[ -z "$STOCKFISH_PATH" || ! -x "$STOCKFISH_PATH" ]]; then
  echo "error: Stockfish is required; set STOCKFISH_PATH to its executable" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("error: CUDA is required for search evaluation")
print(f"search device: {torch.cuda.get_device_name(0)}")
PY

"$PYTHON" scripts/eval_search_vs_stockfish.py \
  --checkpoint "$CHECKPOINT" \
  --out "$EVAL_OUT" \
  --games "$GAMES" \
  --simulations "$SIMULATIONS" \
  --stockfish-path "$STOCKFISH_PATH" \
  --stockfish-elo "$STOCKFISH_ELO" \
  --stockfish-time "$STOCKFISH_TIME" \
  --max-plies "$MAX_PLIES" \
  --device cuda
