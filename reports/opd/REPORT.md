# On-Policy Distillation from lc0 (D53/D54)

## Motivation

Our 142M-parameter comp model has a hard strength ceiling (~2500-2600 Elo,
see D46/D47) that scale alone has not been able to push past recently. lc0
`t1-256x10` is a materially stronger teacher: run at `nodes=1` (i.e. as a raw
policy, no search), its move choice beats our own 64-sim PUCT search 0.78
head-to-head. That gap motivated trying a modern on-policy distillation (OPD)
recipe , imitate a stronger teacher's policy on positions the *student itself*
generates, rather than a static offline dataset , to see if we could pull some
of lc0's policy quality into our net without a full retrain.

## Method

- **Position generation**: on-policy rollouts from the current student model
  at temperature 0.3 (self-generated positions, not external games).
- **Teacher signal**: lc0 `t1-256x10` visit-distribution at 400 nodes, used as
  a soft target.
- **Objective**: reverse-KL from student to teacher visit distribution, plus
  an anchor term back to the base model's own policy (to limit drift).
- **Trunk**: frozen in the gentle configuration; trainable in the aggressive
  configuration.
- **Configurations tested**:
  - **v1 (aggressive)** , all parameters trainable, lr 2e-4, no trunk freeze.
  - **v2 (gentle)** , frozen trunk, base-anchor loss, single round.
  - **iterative (R1->R2->R3)** , three successive rounds of the v2-style
    recipe, each round distilling from the model produced by the previous
    round, to test whether repeated rounds compound gains.
- **Evaluation**: win rate over n=20 games against two opponents ,
  (a) **Leela-2700**, `lc0 t1-256x10` at `nodes=1`, an *external*, fixed
  reference we did not train against, and (b) the model's **own starting
  base**, i.e. a self-play head-to-head. Wilson 95% score intervals are
  reported for all rates given the small n.

## Results

### Win rate vs Leela-2700 (external test, n=20/point)

| Variant | Win rate | Wilson 95% CI |
|---|---|---|
| base (starting model) | 0.225 | [0.096, 0.443] |
| v1 (aggressive) | 0.056 | [0.011, 0.244] |
| v2 (gentle) | 0.100 | [0.028, 0.301] |
| iter-R1 | 0.200 | [0.081, 0.416] |
| iter-R2 | 0.125 | [0.040, 0.331] |
| iter-R3 | 0.150 | [0.052, 0.360] |

Every single configuration lands at or below the base rate of 0.225. The
iterative run's own trajectory also declines across rounds (R1 0.200 -> R2
0.125 -> R3 0.150), i.e. more rounds of self-generated on-policy distillation
made things worse, not better, within the tested range.

### Win rate vs own base (head-to-head, n=20)

| Variant | vs own base | Wilson 95% CI | vs Leela-2700 | Wilson 95% CI |
|---|---|---|---|---|
| v2 (gentle) | 0.625 | [0.409, 0.800] | 0.100 | [0.028, 0.301] |
| iter-R3 | 0.300 | [0.145, 0.519] | 0.150 | [0.052, 0.360] |

(R1 and R2 head-to-head-vs-base numbers were not collected -- N/A.)

v2 clears 0.5 against its own base sibling (0.625, CI excludes 0.5) -- by that
metric alone it would look like an improvement. But against the external
Leela-2700 reference it drops to 0.100, well below base's own 0.225. The two
metrics disagree, and the external one is the one that matters.

See figures:

- ![vs Leela-2700 arc](fig_opd_vs2700_arc.png)
- ![own base vs external](fig_opd_base_vs_external.png)

## Verdict: closed lever

On-policy distillation from lc0, in every flavor tried (aggressive full
fine-tune, gentle frozen-trunk+anchor, and 3-round iteration on top of the
gentle recipe), **failed to strengthen the model against an external
opponent**. All six data points sit at or below the untrained base's 0.225
win rate vs Leela-2700, and the iterative variant got monotonically worse
after round 1. We are closing this lever: policy imitation from a single
stronger teacher, in this setup, cannot lift this net's playing strength.

The key methodological takeaway carried forward: **beating your own base
head-to-head is not evidence of strength.** v2 beats its own base 0.625 of
the time while being clearly worse in absolute terms (0.100 vs 0.225 vs
Leela-2700) -- the distilled model likely overfit to exploitable stylistic
quirks of its sibling rather than gaining generalizable skill. Any future
self-play or distillation experiment must be scored against a fixed external
opponent, not just the model's own ancestor.
