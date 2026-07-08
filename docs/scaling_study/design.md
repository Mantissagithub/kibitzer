# scaling study , design

## fixed choices

- **architecture family (hard constraint): attention-first.** set `attention_every=1` so every trunk block is a
  `CausalAttentionBlock`; scale by width (`d_model`) and depth (`trunk_layers`). the 3-layer encoder and 8 heads
  stay, heads scale with width to keep head-dim ~constant.
- **metric: policy cross-entropy** on lichess-elite human moves (primary , cheap, smooth, low-variance). value MSE
  secondary. top-1 move-match as the interpretable capability axis. **not match-play elo** , D35 showed it is too
  noisy (SE ≈ 0.09 at 20 games) and does not track offline gains.
- **data:** the already-cached lichess-elite pool. vary positions seen `D`.
- **eval:** one fixed game-disjoint held-out position set for loss; a fixed move-match set for capability. same
  split reused across every rung so curves are comparable.

## model-size ladder

all rungs carry attention in every trunk block. policy head is `d_model × 4672`, so params are not pure width² ,
exact counts are measured by the harness, not assumed.

| tag | d_model | trunk_layers | n_heads | approx params | notes |
|-----|---------|--------------|---------|---------------|-------|
| S0  | 128     | 6            | 4       | ~5M           | LR-tuning anchor |
| S1  | 192     | 8            | 6       | ~10M          | |
| S2  | 256     | 10           | 8       | ~18M          | |
| S3  | 320     | 10           | 8       | ~32M          | current default config |
| S4  | 448     | 12           | 8       | ~60M          | grad-accum to fit 8gb |

## learning rate

confounded scaling curves come from mistuned LR per size. use **μP-style transfer**: sweep LR once at S0
(e.g. {1e-3, 3e-4, 1e-4}), pick best final policy loss, then set each larger rung's LR = `lr_S0 · width_S0 / width`.
confirm the transfer holds with a single check at S2. schedule: linear warmup (~2% of steps) + cosine decay , this
directly fixes the constant-LR epoch-3 divergence seen in the joint_scratch run (`reports/joint_scratch/fig1`).

## procedure

1. **cheapest first cut (params-limited):** train S0–S3 at a fixed data-rich `D` (~20M positions) to convergence.
   fit `L(N)` , a power law in params. this alone tells the exponent and whether more params still help before
   spending on a full grid. add S4 only if the curve is still improving.
2. **iso-flop grid (only if step 1 is promising):** pick 3–4 compute budgets `C` (flops ≈ `6ND`). for each `C`
   train several `(N, D)` pairs along the iso-flop line; the loss-minimizing `N` per `C` traces the compute-optimal
   frontier `N_opt(C)`, `D_opt(C)` (chinchilla).
3. **extrapolate + verdict:** from the fit, read off the `(N, D)` needed to reach a target policy loss / move-match,
   convert to flops, compare against the 8gb-4060 budget. output an honest reachable / not-reachable-on-laptop call.

## logging

each rung logs `(N_params, D_positions, policy_loss, value_mse, top1_move_match, wall_clock, peak_vram)` to
`reports/scaling_law/results.json`. plots (params-vs-loss, iso-flop frontier) go to `reports/scaling_law/`.

## entrypoint (to build)

does not exist yet , first build step:

```bash
# scripts/scaling_sweep.py drives train_bc.py per rung with attention_every=1 and mup LR transfer
python scripts/scaling_sweep.py --sizes S0,S1,S2,S3 --attention-every 1 \
  --max-positions 20000000 --lr-transfer mup --out reports/scaling_law/results.json
```
