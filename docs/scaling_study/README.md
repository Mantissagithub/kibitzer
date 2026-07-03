# scaling study — why

the case for stopping point-experiments and measuring the scaling curve instead.

## the evidence that forced this

thirty-five point-experiments (D1–D35), none beat the supervised ~1320 baseline, every one run on a single fixed
~32M model. the failure is not any single trick — it is the method: guessing one lever at a time on one model size,
graded by a noisy expensive signal.

D35 made the confound explicit:

- **offline value metrics do not predict play.** joint-from-scratch won the held-out decisive-sign metric by
  +6.67pp (65.95% → 72.62%) and came out level-to-behind at the board (0.175 vs 0.225 score vs stockfish-1320,
  within the ±0.09 noise of a 20-game match). the whole D30–D35 value-repair campaign was optimizing a proxy that
  doesn't track the goal.
- **match play is a terrible optimization signal.** 20 games ⇒ score SE ≈ 0.09, so two configs 0.05 apart are
  indistinguishable. every gate we ran was under-powered.
- **the only lever that moved strength was search depth.** value_final went 0.100 → 0.225 from 64 → 256 sims, then
  plateaus. search is an inference-time crutch, not a better model.

full numbers: `reports/search_depth/results.json`, figures in `reports/joint_scratch/` and `reports/search_depth/`.

## what a scaling law buys us

instead of anecdotes, a prediction. fit policy cross-entropy `L(N, D)` over model size `N` and positions seen `D`,
then extrapolate to the one question that actually matters:

> what `N` and `D` reach a target move-match (an elo proxy), and is that reachable on the 8gb 4060 — or is this
> fundamentally a cloud-scale problem?

policy cross-entropy is the right anchor because it is cheap, smooth, and low-variance — the opposite of match play.
this is the chinchilla / kaplan method (arXiv:2001.08361, 2203.15556). the scale reference is deepmind's
searchless-chess (arXiv:2402.04494): 2895 lichess elo with 270M params on ~15b positions, no search. locating
ourselves against that curve tells us honestly whether the laptop can ever get there.

## the hard constraint

the backbone must be **attention**. the current model is mostly SSM (`attention_every=3`, blocks 0/3/6/9 attention,
the rest selective-SSM). the study sets `attention_every=1` so every trunk block is causal attention, and scales
width and depth from there. design and ladder: `design.md`.
