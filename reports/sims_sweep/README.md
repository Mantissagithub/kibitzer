# sims_sweep report

D63 tests whether the current best checkpoint is actually compute-starved at inference time.
The model is fixed: `runs/tactical/tactical_repair.pt`. Only the PUCT simulation count changes.
The opponent is the same Leela/Maia-2700 proxy at nodes=1, so these numbers are a search-budget yardstick, not an intrinsic model Elo.

![score and wdl](fig_sims_sweep.png)

![implied elo](fig_sims_elo.png)

## results

| sims | games | W/D/L | score | Elo delta | implied proxy Elo | source |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 40 | 3/1/36 | 0.087 | -407 | 2293 | `kibitzer_vs2700_s64_g40_seed23.jsonl` |
| 128 | 40 | 6/11/23 | 0.287 | -158 | 2542 | `kibitzer_vs2700_s128_g40_seed23.jsonl` |
| 256 | 40 | 5/16/19 | 0.325 | -127 | 2573 | `kibitzer_vs2700_s256_g40_seed23.jsonl` |
| 512 | 40 | 29/8/3 | 0.825 | 269 | 2969 | `kibitzer_vs2700_s512_g40_seed23.jsonl` |

## read

- 128 sims is the control: 6W/11D/23L, score 0.287.
- 512 sims is the winner: 29W/8D/3L, score 0.825.
- The best point is +0.537 score rate over the 128-sim control.
- The jump is inference-time search, not learning. It says the checkpoint had latent strength that shallow search was failing to extract.
- Because the opponent is one-node Leela/Maia, the implied Elo should be read as a proxy scale only.

## next

Run a budgeted 1024/2048 confirmation on a rented GPU, ideally against 2700 plus stronger Leela checkpoints.
Keep the same paired openings and colors before claiming a new default search budget.
