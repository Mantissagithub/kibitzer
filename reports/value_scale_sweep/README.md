# D62: value_scale sweep (rejected, value head is load-bearing)

The mechanistic-interp study showed the value head lands on the right final sign but is
noisy and late. Since PUCT backs that value up at full weight (`value_scale=1.0`) and the
Leela-2700 gate hardcoded it, the cheap non-architectural test was: **down-weight the
noisy value in search and see if external play holds or improves.** Zero training.

It did not. It regressed monotonically.

| value_scale | W/D/L | score | implied Elo |
|---|---|---|---|
| **1.0** (control) | 6 / 11 / 23 | **0.287** | 2542 |
| 0.75 | 3 / 7 / 30 | 0.163 | 2415 |
| 0.5 | 2 / 1 / 37 | 0.062 | 2230 |

(vs Leela/Maia-2700, 40 games @128 sims, seed 23. The 0.25 and 0.0 runs were killed once
the trend was obvious.)

![value_scale sweep](fig_value_scale_sweep.png)

## read

The 1.0 control reproduced ~0.29, so nothing else drifted. Every notch away from the
value head collapsed play harder. So the interp's "value is noisy and late" was true, but
the inference was wrong: the value backup is **heavily load-bearing**. Even a shaky value
signal is what turns the raw policy prior into ~2500 play, and starving it drops the
search back toward the raw prior, which Leela-2700 punishes hard.

## verdict

Down-weighting value is the wrong direction. Combined with D52 this fences the value head
from both sides:

- **D52** — make it bigger / better offline → play regressed.
- **D62** — trust it less in search → play regressed worse.

The value head is weak **and** essential, and cannot be cheaply routed around or upgraded.
The only remaining fix is better representations feeding it, i.e. the scale / deeper-encoder
+ real-pooling track (which the interp's mean-pool-bottleneck finding independently points
at). The `--value-scale` passthrough stays in `maia_gauntlet.py`, default 1.0 now confirmed
correct.

Full decision: `LOGBOOK.md` D62. Raw games: `kibitzer_vs2700_s128_g40_seed23_vs*.{jsonl,pgn}`.
