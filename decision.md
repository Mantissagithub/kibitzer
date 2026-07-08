# AZ Self-Play Decision Log

## 2026-07-09: Kill slow self-play, rewrite for GPU parallelism + logging

### What was running

Az_run.sh started 2026-07-08, a 3-iteration AlphaZero loop from `S2_shaw_142M_comp.pt`:
- Iter 1: 80 games @ 400 sims → 7431 positions, 3 epochs training, match vs base (0.625), eval vs Maia 2700 (0.100 — worse than base ref 0.225)
- Iter 2: gen killed at ~80 games (2h44m elapsed), before training/eval

### Why we killed it

| Metric | Value | Issue |
|---|---|---|
| Gen time per iter | ~2.5 hours | 400 sims/move, single-threaded, no batching |
| GPU utilization | 42% avg | Model forward pass is batch-size=1, GPU mostly idle |
| CPU utilization | 100% on 1 core | MCTS tree ops on one thread, no parallelism |
| Positions per iter | ~7000-7500 | 80 games too few, narrow training distribution |
| Per-iter improvement | vs base +0.125, vs Maia -0.125 | Overfitting to self-play, regressing vs strong opponent |
| Gen logging | **zero output** | Gen subcommand is silent — no way to monitor progress |
| Model forward pass | ~5ms each | ~1.92M single-board forwards per gen iter (400 sims × 60 moves × 80 games) |
| VRAM usage | 252 MB / 8 GB | 96% VRAM wasted |

### What we're changing

1. **Lower sims, more games** — 200 sims (down from 400), 200-300 games (up from 80). 200 sims is enough for reasonable search quality; more games gives the trainer a wider position distribution.

2. **Clear logging** — gen now prints per-game progress: game number, plies, positions, cumulative total, ETA, result. Example:
   ```
   [gen 7/200] 54 plies  53 pos  total 371 pos  result 1-0  2.3m elapsed  ~63m left
   ```

3. **Batch inference in ModelEvaluator** — added `evaluate_batch()` for future search optimization (not yet wired into puct_search).

### Recommended new run config

```bash
# 200 games @ 200 sims: ~2× more positions than old 80@400, similar wall-clock
uv run python scripts/selfplay_az.py gen \
  --checkpoint runs/scaling_shaw_comp/S2_shaw_142M_comp.pt \
  --games 200 --sims 200 \
  --out-jsonl runs/az/az_data_v2.jsonl
```

Expected: ~15,000 positions (2× the old 7,500), similar total time since 200 sims is ~half the cost of 400.

### Future improvements

- **Parallel game generation** — multiprocessing across games for 4-8× gen speedup (deferred to keep current implementation simple).
- **Batch MCTS search** — wire `evaluate_batch` into `puct_search` so each simulation step evaluates all leaf nodes in one GPU call.
- **Streaming data pipeline** — stream positions to disk and training queue while generating.