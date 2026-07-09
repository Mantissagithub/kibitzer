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

## 2026-07-09: Stop pure AZ as the next lever; add regret-guided teacher repair

### What changed

The 200-game / 200-sim AZ run made the core problem clearer: training on our
own visit distributions and self-play outcomes can beat a sibling checkpoint
without proving real strength. The older value-head repairs also already showed
the same pattern in another form: better offline value labels did not translate
to better play.

### Decision

Do not spend the next cycle on a stronger scalar value head or another pure AZ
iteration. Add a targeted repair loop instead:

1. collect positions from AZ JSONL / PGN where Kibitzer is likely wrong;
2. label those positions with Stockfish MultiPV top-k action values;
3. keep only high-regret positions, or AZ positions where final outcome and
   teacher value strongly disagree;
4. fine-tune from `S2_shaw_142M_comp.pt` with teacher top-k policy/value targets;
5. keep a KL anchor to the base policy so the repair does not become another
   sibling-overfit model.

Implemented entrypoint:

```bash
uv run python scripts/train_regret_repair.py label \
  --az-jsonl runs/az/az_data_1.jsonl \
  --checkpoint runs/scaling_shaw_comp/S2_shaw_142M_comp.pt \
  --out-jsonl runs/regret/az1_sf12.jsonl \
  --depth 12 --multipv 8 --min-regret 0.20 --min-outcome-gap 0.75

uv run python scripts/train_regret_repair.py train \
  --checkpoint runs/scaling_shaw_comp/S2_shaw_142M_comp.pt \
  --data "runs/regret/*.jsonl" \
  --out runs/regret/regret_repair.pt \
  --anchor-weight 0.5 --value-weight 0.25
```

### Gate

This is not accepted because the training loss improves. It only counts if it
beats the base on fixed external opponents: Leela/Maia-2700 style eval and the
existing Stockfish ladder. Head-to-head vs its own base stays diagnostic-only.

### First result

The first regret-repair checkpoint failed the external gate:

| eval | result |
|---|---|
| `regret_repair.pt` vs Leela/Maia-2700 proxy, 128 Kibitzer sims, 20 games | 1W / 3D / 16L, score 0.125 |
| base reference in prior runs | 0.225 |

The buffer explains the failure. `runs/regret/az1_sf12.jsonl` had 4,611 records,
but only 764 were actual policy-regret hits. 3,995 were kept because AZ's final
outcome target disagreed with Stockfish value. That turned the run back into a
value-repair experiment, and value repair is already a closed lever.

Next attempt should be policy-regret-only:

```bash
ACTION=all \
AZ_JSONLS=runs/az/az_data_1.jsonl,runs/az/az_data_2.jsonl \
REGRET_JSONL=runs/regret/az12_policy_regret_sf12.jsonl \
OUTPUT_CHECKPOINT=runs/regret/policy_regret_repair.pt \
MIN_REGRET=0.05 \
MIN_OUTCOME_GAP=9 \
VALUE_WEIGHT=0.0 \
ANCHOR_WEIGHT=0.75 \
EPOCHS=5 \
bash scripts/run_regret_repair.sh
```

Rationale: collect many more policy mistakes, ignore scalar outcome/value
disagreement for now, and keep the base-policy anchor stronger while moving the
policy head toward Stockfish top-k moves.

### Policy-regret-only result

The policy-regret-only run improved over the previous failed repairs, but still
needed a direct same-config base rerun:

| checkpoint | eval | result |
|---|---|---|
| `az_iter_1.pt` | Leela/Maia-2700 proxy, 64 sims, 20 games | 1W / 2D / 17L, score 0.100 |
| `regret_repair.pt` | Leela/Maia-2700 proxy, 128 sims, 20 games | 1W / 3D / 16L, score 0.125 |
| `policy_regret_repair.pt` | Leela/Maia-2700 proxy, 128 sims, 20 games | 2W / 3D / 15L, score 0.175 |
| `S2_shaw_142M_comp.pt` direct rerun | Leela/Maia-2700 proxy, 128 sims, 20 games | 0W / 3D / 17L, score 0.075 |
| base reference in older prior runs | Leela/Maia-2700 proxy | 0.225 |

This says the policy-regret filter is directionally better than outcome/value
repair. Against the same 128-sim seed/openings it also beat the comp base
directly, though n=20 is still small and the older base reference conflicts with
this rerun. Treat this as promising, not proven: next validate with a larger
same-config match or a Stockfish move-regret diagnostic before scaling the run.

### Bigger policy-regret result

Tried a broader/longer policy-regret run:

```bash
ACTION=all \
AZ_JSONLS=runs/az/az_data_1.jsonl,runs/az/az_data_2.jsonl \
REGRET_JSONL=runs/regret/az12_policy_regret_sf12_bigger.jsonl \
OUTPUT_CHECKPOINT=runs/regret/policy_regret_repair_bigger.pt \
MIN_REGRET=0.03 \
MIN_OUTCOME_GAP=9 \
VALUE_WEIGHT=0.0 \
ANCHOR_WEIGHT=0.75 \
EPOCHS=8 \
bash scripts/run_regret_repair.sh
```

External result:

| checkpoint | Leela/Maia-2700 proxy, 128 sims, 20 games |
|---|---|
| `policy_regret_repair_bigger.pt` | 0W / 6D / 14L, score 0.150 |
| `policy_regret_repair.pt` | 2W / 3D / 15L, score 0.175 |
| `S2_shaw_142M_comp.pt` direct rerun | 0W / 3D / 17L, score 0.075 |

Conclusion: lowering the regret threshold and training longer did not improve
over the first policy-regret run. The broader buffer may be too noisy/diffuse,
or 8 epochs may be overtraining the policy head toward Stockfish top-k quirks.

## 2026-07-09: Add regret-start mini self-play

Plain AZ from openings overfit sibling style, while policy-regret repair gave
the first direct same-config lift over comp base. The next implementation is
targeted self-play from high-regret positions, not another opening self-play
loop.

Implemented entrypoint:

```bash
bash scripts/run_regret_start_az.sh
```

Default shape:

- init checkpoint: `runs/regret/policy_regret_repair.pt`
- start buffer: `runs/regret/az12_policy_regret_sf12.jsonl`
- generation: 1,000 high-regret starts, 128 sims, 32 continuation plies
- value target: root search value, with value loss disabled by default
- training: policy visit CE + base/candidate KL anchor, heads + final norm only

Smoke verified on CPU with 1 start / 1 sim / 1 ply and 1 training epoch using
temporary `/tmp` outputs.

### Regret-start result

The targeted self-play checkpoint did not help externally:

| checkpoint | Leela/Maia-2700 proxy, 128 sims, 20 games |
|---|---|
| `regret_start_az.pt` | 1W / 1D / 18L, score 0.075 |
| `policy_regret_repair.pt` | 2W / 3D / 15L, score 0.175 |
| `policy_regret_repair_bigger.pt` | 0W / 6D / 14L, score 0.150 |
| `S2_shaw_142M_comp.pt` direct rerun | 0W / 3D / 17L, score 0.075 |

Conclusion: targeted self-play from regret positions still overfit to the
model's own search visits. The current best remains `policy_regret_repair.pt`;
self-play should stay closed unless an external teacher labels the continuation
states.

## 2026-07-09: Tactical supervised repair beats the cheap external gate

The competition/puzzle supervised repair finished from
`runs/regret/policy_regret_repair.pt`:

```text
base  : top1=0.4933 value_mse=0.6562
tactic: top1=0.4922 value_mse=0.6602
delta : top1=-0.0011 value_mse=+0.0040
GATE: PASS_HELDOUT_TOP1
```

Then the same cheap external gate used for the regret branch was run:

```bash
uv run python scripts/maia_gauntlet.py \
  --checkpoint runs/tactical/tactical_repair.pt \
  --maia-weights data/leela/t1-256x10-distilled.pb.gz \
  --maia-elo 2700 \
  --lc0-path data/leela/lc0 \
  --backend cuda \
  --maia-nodes 1 \
  --games 20 \
  --simulations 128 \
  --seed 7 \
  --out-jsonl reports/tactical_repair/tactical_repair_vs2700_s128.jsonl \
  --out-pgn reports/tactical_repair/tactical_repair_vs2700_s128.pgn
```

Result:

| checkpoint | Leela/Maia-2700 proxy, 128 sims, 20 games |
|---|---|
| `tactical_repair.pt` | 2W / 6D / 12L, score 0.250 |
| `policy_regret_repair.pt` | 2W / 3D / 15L, score 0.175 |
| `policy_regret_repair_bigger.pt` | 0W / 6D / 14L, score 0.150 |
| `regret_start_az.pt` | 1W / 1D / 18L, score 0.075 |
| `S2_shaw_142M_comp.pt` direct rerun | 0W / 3D / 17L, score 0.075 |

Conclusion: tactical supervised repair is now the best cheap-gate checkpoint,
but only by a 20-game probe. Promote it only after a larger same-config external
match confirms the lift. The report/graphs for this decision live under
`reports/repair_eval/`.

### Paired 80-game confirmation

Ran the 3-way paired gate with the same opponent/config/opening seed:

```bash
bash scripts/run_repair_eval_gate.sh
```

Gate config:

- opponent: Leela/Maia-2700 proxy, nodes=1
- Kibitzer search: 128 sims
- games: 80 each
- seed: 17

Result, using the standard Elo transform
`elo_delta = 400 * log10(score_rate / (1 - score_rate))` against the 2700
opponent label:

| checkpoint | W/D/L | score rate | Elo delta | implied Elo |
|---|---:|---:|---:|---:|
| `tactical_repair.pt` | 7 / 25 / 48 | 0.244 | -197 | 2503 |
| `policy_regret_repair.pt` | 9 / 18 / 53 | 0.225 | -215 | 2485 |
| `S2_shaw_142M_comp.pt` | 6 / 17 / 57 | 0.181 | -262 | 2438 |

Conclusion: the tactical repair lift survived the larger paired gate, but the
margin over policy-regret is small: about +0.019 score rate, or +18 Elo by this
proxy. Treat `runs/tactical/tactical_repair.pt` as the current best base, but do
not read it as a major breakthrough. The next training branch should continue
from tactical repair only if it keeps the same external gate discipline.

### Tactical R2 result

Tried a conservative tactical continuation from `runs/tactical/tactical_repair.pt`:

```bash
INIT_CHECKPOINT=runs/tactical/tactical_repair.pt \
OUTPUT_CHECKPOINT=runs/tactical/tactical_repair_r2.pt \
MAX_POSITIONS=300000 \
MIX_RATIO=0.30 \
RATING_MIN=1800 \
RATING_MAX=2800 \
LEARNING_RATE=1e-5 \
VALUE_WEIGHT=0.0 \
EVAL_POSITIONS=50000 \
bash scripts/run_tactical_repair.sh
```

Offline held-out gate passed:

```text
base  : top1=0.4922 value_mse=0.6602
tactic: top1=0.4916 value_mse=0.6595
delta : top1=-0.0007 value_mse=-0.0007
GATE: PASS_HELDOUT_TOP1
```

External paired gate, seed 23:

| checkpoint | W/D/L | score rate | Elo delta | implied Elo |
|---|---:|---:|---:|---:|
| `tactical_repair.pt` | 12 / 23 / 45 | 0.294 | -152 | 2548 |
| `tactical_repair_r2.pt` | 9 / 18 / 53 | 0.225 | -215 | 2485 |
| `policy_regret_repair.pt` | 8 / 16 / 56 | 0.200 | -241 | 2459 |

Conclusion: reject R2. The held-out tactical/game top-1 gate was not enough;
external play regressed by about -0.069 score rate / -63 Elo relative to R1 on
the same seed. Keep `runs/tactical/tactical_repair.pt` as the current best local
base and stop simply scaling tactical mid-training in this form.
