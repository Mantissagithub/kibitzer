set -e
cd /home/pradheep/Documents/kibitzer
NET=data/leela/t1-256x10-distilled.pb.gz
BASE=runs/scaling_shaw_comp/S2_shaw_142M_comp.pt
echo "starting AZ self-play from comp_base (200 games @ 200 sims)"
PREV=$BASE
GAMES=200; SIMS=200; ITERS=3
for i in 1 2 3; do
  echo "=== AZ ITER $i: SELF-PLAY (sims $SIMS, $GAMES games, dirichlet) from $(basename $PREV) ==="
  PYTHONUNBUFFERED=1 uv run python scripts/selfplay_az.py gen \
    --checkpoint $PREV --games $GAMES --sims $SIMS --temp-plies 20 \
    --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --seed $((20+i)) \
    --out-jsonl runs/az/az_data_$i.jsonl
  echo "=== AZ ITER $i: TRAIN (AZ loss: policy-CE to visits + value-MSE to outcome z, replay buffer) ==="
  PYTHONUNBUFFERED=1 uv run python scripts/selfplay_az.py train \
    --checkpoint $PREV --data "runs/az/az_data_*.jsonl" --value-weight 1.0 \
    --epochs 3 --lr 2e-5 --batch-size 128 --out runs/az/az_iter_$i.pt
  echo "=== AZ ITER $i: EVAL vs BASE ==="
  PYTHONUNBUFFERED=1 uv run python scripts/selfplay_az.py match \
    --model-a runs/az/az_iter_$i.pt --model-b $BASE --games 20 --sims 128 --seed 7 \
    --out-json reports/az/az_iter${i}_vs_base.json
  echo "=== AZ ITER $i: EVAL vs 2700 (lc0 nodes=1), base ref 0.225 ==="
  PYTHONUNBUFFERED=1 uv run python scripts/maia_gauntlet.py \
    --checkpoint runs/az/az_iter_$i.pt --maia-weights $NET --maia-elo 2700 \
    --lc0-path data/leela/lc0 --backend cuda --maia-nodes 1 --games 20 --simulations 64 --seed 7 \
    --out-jsonl reports/az/az_iter${i}_vs2700.jsonl --out-pgn reports/az/az_iter${i}_vs2700.pgn
  echo "AZ_ITER_${i}_DONE"
  PREV=runs/az/az_iter_$i.pt
done
echo "AZ_SELFPLAY_DONE"