#!/bin/bash
# runs on the rented pod: install deps + stockfish, then launch a parallel
# gauntlet (levels x shards) against stockfish. results stream to ~/gaunt_out
# per game so partial data is usable if we stop at the time cap.
set -u
export DEBIAN_FRONTEND=noninteractive

echo "[setup] apt stockfish"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq stockfish >/dev/null 2>&1
echo "[setup] pip deps"
pip install -q python-chess numpy >/dev/null 2>&1
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1
python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p ~/gaunt_out
CKPT=~/S2_shaw_100M.pt
LEVELS="1900 2100 2300 2500"
SHARDS="0 1 2"
GAMES=24
SIMS=256

echo "[gauntlet] launching $(( $(echo $LEVELS|wc -w) * $(echo $SHARDS|wc -w) )) parallel shards"
seed=0
for lvl in $LEVELS; do
  for sh in $SHARDS; do
    seed=$((seed+1))
    PYTHONPATH=~ python ~/pod_gauntlet.py --checkpoint "$CKPT" --stockfish-elo "$lvl" \
      --games "$GAMES" --simulations "$SIMS" --seed "$seed" \
      --out-jsonl ~/gaunt_out/g_${lvl}_${sh}.jsonl --out-pgn ~/gaunt_out/g_${lvl}_${sh}.pgn \
      >~/gaunt_out/log_${lvl}_${sh}.txt 2>&1 &
  done
done
wait
echo "[gauntlet] ALL DONE"
