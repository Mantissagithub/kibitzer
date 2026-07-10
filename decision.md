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

## 2026-07-10: Add teacher-preference repair branch

The next RL-style branch should not be PPO, pure AZ, or another scalar value
repair. The recent evidence points the other way:

- plain AZ beat its sibling but regressed vs the Leela/Maia external yardstick
- regret-start self-play also fell back to the comp-base score band
- tactical R1 is the current best checkpoint
- tactical R2 passed held-out top-1 but failed the external match

Implemented a DPO/AWAC-style preference repair path:

```bash
bash scripts/run_preference_repair.sh
```

Default shape:

- reference/init checkpoint: `runs/tactical/tactical_repair.pt`
- preference buffer: `runs/preference/r1_teacher_pairs_sf12.jsonl`
- output checkpoint: `runs/preference/preference_repair.pt`
- teacher: Stockfish depth 12, MultiPV 8
- sources: policy-regret buffers plus the held-out elite PGN slice
- objective: DPO pair loss + small teacher CE + frozen-reference KL
- trainable scope: policy head + final norm by default
- no value loss and no learned Q/value critic in the first pass

The labels are intentionally model-mistake pairs: for each position, keep the
teacher's best move as `good_move` and the current policy's highest-probability
clearly-worse move as `bad_move`. If the policy mistake is outside MultiPV, use
the teacher's worst labeled score as a floor so the pair still trains away from
the tempting move.

Promotion rule stays external. Offline pair accuracy and CE are sanity checks
only. Promote `preference_repair.pt` only if the paired Leela/Maia-2700 gate at
128 sims beats tactical R1 by at least +0.03 score rate or about +25 Elo:

```bash
CANDIDATE_NAME=preference_repair \
CANDIDATE_CHECKPOINT=runs/preference/preference_repair.pt \
CANDIDATE_REPORT_DIR=reports/preference_repair \
SEED=31 \
bash scripts/run_repair_eval_gate.sh
```

### Preference repair result

The first preference repair checkpoint failed externally and should not be
promoted:

| checkpoint | gate | W/D/L | score rate | implied Elo |
|---|---:|---:|---:|---:|
| `preference_repair.pt` | 62/80 stopped | 3 / 13 / 46 | 0.153 | 2403 |
| `tactical_repair.pt` reference | 80/80 seed 23 | 12 / 23 / 45 | 0.294 | 2548 |

Offline checkpoint metrics looked superficially acceptable:

```text
dpo_loss=0.6934 ce_loss=2.2871 anchor_kl=0.0009
pair_acc=0.6070 pair_margin=0.7716
```

But the external curve collapsed after the first few games. The preference
buffer had 52,250 pairs, mean teacher margin 0.236, and 21,217 pairs where the
bad move used the MultiPV floor. That floor-heavy label shape is probably too
noisy/aggressive for this small policy.

Conclusion: reject this checkpoint. If this branch is retried, use a much lower
step size, much stronger anchor, and a single epoch. Do not run another full
gate unless the cheap early gate is clearly above tactical R1's band. The plots
and report live in `reports/preference_repair/`.

### Conservative preference retry result

Tried the conservative anchor-heavy salvage pass:

```bash
ACTION=train \
PREFERENCE_JSONL=runs/preference/r1_teacher_pairs_sf12.jsonl \
OUTPUT_CHECKPOINT=runs/preference/preference_repair_anchor_r1.pt \
LEARNING_RATE=3e-6 \
EPOCHS=1 \
BETA=0.03 \
CE_WEIGHT=0.5 \
ANCHOR_WEIGHT=0.5 \
bash scripts/run_preference_repair.sh
```

Then started the same Leela/Maia-2700 external gate and stopped it early:

| checkpoint | gate | W/D/L | score rate | implied Elo |
|---|---:|---:|---:|---:|
| `preference_repair_anchor_r1.pt` | 62/80 stopped | 4 / 17 / 41 | 0.202 | 2461 |
| `tactical_repair.pt` reference | 80/80 seed 23 | 12 / 23 / 45 | 0.294 | 2548 |

Conclusion: reject the conservative retry too. The stronger anchor reduced the
damage compared with the first preference checkpoint, but it still sits around
policy-regret/comp-base territory and does not threaten tactical R1. Close the
teacher-preference branch for now. Any future version needs cleaner labels
before training, not more tuning of this same buffer.

Saved stopped-run evidence:

- `reports/preference_repair/preference_repair_anchor_r1_vs2700_s128_g80_seed31_stopped62.jsonl`
- `reports/preference_repair/preference_repair_anchor_r1_vs2700_s128_g80_seed31_stopped62.log`
- `reports/preference_repair/preference_repair_anchor_r1_vs2700_s128_g80_seed31_stopped62.pgn`

## 2026-07-10: Genuine RL branch — GRPO + exact-divergence DPPO on external reward

Every repair branch above (regret, regret-start self-play, tactical R2, DPO/AWAC
preference) shares the same failure signature: it trains on the model's own
outputs and/or drifts off the base, beats a sibling, and stalls or regresses
against the external Leela/Maia-2700 yardstick. tactical R1
(`runs/tactical/tactical_repair.pt`, 0.294 @128 sims / ~2548 proxy-Elo) is still
the only genuine external-gate win, and it is supervised, not RL.

### Decision

Try genuine RL — the one untried lever — but design it to structurally avoid
both failure ingredients. Planned with Fable 5 against the user's research folio
methods (GRPO, DPPO arXiv 2602.04879, MaxRL arXiv 2602.02710) and domain papers
(GRPO-for-chess 2507.00726, policy-gradient-search 1904.03646). The bet:

- **Critic-free GRPO** — group-relative z-score advantage replaces the value head
  (the proven-dead lever, D52), so the frozen value head keeps 128-sim gate
  search anchored.
- **External verifiable reward** — game outcome vs a strength-capped Stockfish
  ladder, never self-play visit targets. No sibling in the loop = nothing to
  style-exploit.
- **Exact-divergence DPPO trust region** — chess's ~30 legal moves let us compute
  the full-distribution TV divergence exactly (no binary/top-k approximation),
  asymmetric mask keyed to δ=0.2, anchored to the rollout policy, plus a weak
  KL(π‖base) β=0.05 to the frozen tactical base as a global anti-drift anchor.
- **Search-based rollouts** — moves are selected by PUCT (default 64 sims, Dirichlet
  root noise + an opening temperature schedule for group diversity), so the model
  plays at its real searched strength; mu recorded is still the RAW prior so the
  DPPO trust region fences the raw policy toward the base while GRPO pushes it
  toward search-validated winners (expert-iteration flavor on external reward).
- **Scope** policy head + final norm only, lr 1e-5, one epoch per fresh buffer.

### Correction (searchless was wrong)

First cut used raw-policy rollouts at temp 1.0 for speed. A diagnostic on
`tactical_repair.pt` vs SF-1600 (no search) exposed why that is fatal: the policy
is extremely sharp, so play collapses with temperature —

| move selection | score vs SF-1600 |
|---|---|
| 128-sim search (the gate) | 1.000 |
| greedy raw (temp 0) | 0.875 |
| raw temp 0.3 | ~0.5 |
| raw temp 0.5 | 0.167 |
| raw temp 1.0 | 0.000 |

At temp 1.0 the ~2500 model hangs pieces into a 1600 and the outcome reward
becomes blunder-noise, not a measure of decision quality. Fix: rollouts now use
PUCT search for move selection (strong play) with a temperature schedule only for
opening spread. The searchless raw path is retained (sims=0) for the smoke/ablation
only.

### The objective and algorithm (exact math)

**Notation.** State $s$ (a position where it is the model's turn) with legal moves
$L(s)$; policy $\pi_\theta(a \mid s) = \mathrm{softmax}$ over the masked policy-head
logits (mass only on $L(s)$). The rollout checkpoint's raw policy is
$\mu = \pi_{\theta_{\text{old}}}$; the frozen tactical base is $\pi_{\text{base}}$
(`tactical_repair.pt`, fixed for the whole run).

**Rollout / behavior.** Games are played in groups; a group is $G$ games sharing one
opening, color, and opponent Elo. Moves are chosen by PUCT (128 sims) with Dirichlet
root noise and a temperature schedule on the visit counts. At every model ply we
store $\big(s_t,\, a_t,\, \mu(\cdot \mid s_t),\, \text{group},\, \text{game}\big)$,
where $a_t$ is the played move.

**Reward (external, verifiable).** Terminal outcome of game $i$ from the model's
point of view — no per-move value target, no learned critic:

$$
R_i = \begin{cases} 1 & \text{win} \\ \tfrac{1}{2} & \text{draw} \\ 0 & \text{loss} \end{cases}
$$

**Advantage (GRPO, critic-free).** z-score the game rewards inside each group $g$ and
broadcast to every model ply $t$ of that game:

$$
\hat{A}_i = \frac{R_i - \frac{1}{|g|}\sum_{j \in g} R_j}
{\sqrt{\frac{1}{|g|}\sum_{j \in g}\big(R_j - \bar{R}_g\big)^2} + \varepsilon},
\qquad \hat{A}_t = \hat{A}_{\mathrm{game}(t)}, \qquad \varepsilon = 10^{-4}
$$

A group with a single distinct result has $\mathrm{std} = 0 \Rightarrow \hat{A} = 0$
(no gradient — uninformative).

**Importance ratio** (per played move):

$$
r_t = \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}
$$

**Exact DPPO trust region.** Total-variation distance over the legal moves only —
exact, since $|L(s)| \approx 30$ (no binary / top-$k$ approximation needed):

$$
D_t = D_{\mathrm{TV}}\!\big(\pi_\theta(\cdot \mid s_t),\, \mu(\cdot \mid s_t)\big)
= \frac{1}{2}\sum_{a \in L(s_t)} \big|\, \pi_\theta(a \mid s_t) - \mu(a \mid s_t) \,\big|
$$

**DPPO asymmetric mask** ($\delta = 0.2$) — block an update only when it pushes
further in the reward-relevant direction past the trust radius; moves back toward
$\mu$ are always allowed; no ratio clipping:

$$
m_t = \begin{cases}
0, & \big(\hat{A}_t > 0 \,\wedge\, r_t > 1 \,\wedge\, D_t > \delta\big)
   \;\vee\; \big(\hat{A}_t < 0 \,\wedge\, r_t < 1 \,\wedge\, D_t > \delta\big) \\[4pt]
1, & \text{otherwise}
\end{cases}
$$

**Base-policy anchor (anti-drift)** — forward KL to the frozen tactical base over
legal moves:

$$
\mathrm{KL}_t = \sum_{a \in L(s_t)} \pi_\theta(a \mid s_t)
\Big[\log \pi_\theta(a \mid s_t) - \log \pi_{\text{base}}(a \mid s_t)\Big]
$$

**Loss** (minimized over $T$ plies; $\beta = 0.05$):

$$
\mathcal{L}(\theta) = -\,\frac{\sum_t m_t\, r_t\, \hat{A}_t}{\sum_t m_t}
\;+\; \beta \cdot \frac{1}{T}\sum_t \mathrm{KL}_t
$$

The gradient flows to the policy head and final RMSNorm only; the trunk and value
head are frozen, so PUCT's value backup at gate time is unchanged.

**Per-iteration algorithm** ($k = 1 \dots 30$):

1. **rollout** — from $\theta_{k-1}$, play $G$-grouped games vs Stockfish at ladder
   Elo $e_k$ (PUCT-128 move selection, Dirichlet + temperature schedule); record
   $\mu$, $a_t$, group.
2. **reward** — $R_i =$ game outcome (model POV).
3. **advantage** — $\hat{A} =$ per-group z-score of $R$, broadcast to plies.
4. **update** — $\theta_k \leftarrow$ one AdamW epoch (lr $10^{-5}$) on
   $\mathcal{L}(\theta)$, warm-started from $\theta_{k-1}$, with $\mu$ frozen as
   $\theta_{k-1}$'s raw policy (the rollout anchor).
5. **ladder** — $e_{k+1} = \mathrm{clip}\big(e_k \pm 100 \text{ toward a } 50\% \text{ score},\ [1600, 2600]\big)$.
6. **every 5** — probe $\theta_k$ vs a fixed held-out Stockfish-2000 (greedy, 128 sims).

**Promotion gate** (external, terminal, unchanged): the best $\theta_k$ vs
Leela/Maia-2700 at 128 sims must beat `tactical_repair.pt`'s $0.294$ by
$\geq +0.03$ score rate. Head-to-head vs the base is deliberately absent — it is the
discredited signal (D48–D54).

**Stage-3 MaxRL variant** (reward transform only, run later on a harder rung):
binarize $R$ (win $=1$, else $0$); within a group drop the non-winning trajectories
from the policy term (success-only averaging) and weight the winners by the harmonic
$\text{pass@}k$ mixture

$$
\nabla_\theta J_{\text{MaxRL}} = \sum_{k=1}^{T} \frac{1}{k}\, \nabla_\theta\, \text{pass@}k,
$$

so low-pass-rate (hard) openings dominate the gradient. The DPPO mask and base-KL
anchor are unchanged.

### Implemented

Greenfield (train_rl.py/train_ppo.py from AGENTS.md don't exist on this branch):

- `kibitzer/grpo.py` — pure math: `group_zscore`, `exact_tv`, `dppo_mask`.
- `kibitzer/rollout.py` — batched parallel game runner vs a Stockfish `UCI_Elo`
  opponent, temperature sampling, 200-ply adjudication (cap = draw), per-ply μ
  records + outcome reward.
- `scripts/train_grpo.py` — `gen`/`train`/`probe`/`loop` subcommands with the
  adaptive ladder (nudge opponent toward ~50% win rate, bounds 1600–2600) and a
  per-iteration metrics jsonl.
- `scripts/run_grpo.sh` — env-var driver with a `SMOKE=1` cpu path.
- `tests/test_grpo.py` — 6 unit tests (z-score edge cases, DPPO mask truth
  table, exact-TV vs hand computation); all pass.

### Gate (unchanged discipline)

Promotion only if the paired 80-game Leela/Maia-2700 gate at 128 sims
(`run_repair_eval_gate.sh`, seed 23) beats tactical R1's 0.294 by ≥ +0.03 score
rate (~+25 Elo). Head-to-head vs own base stays diagnostic-only. If this — the
first RL with a truly external reward — also fails to clear the gate, scale
remains the only lever with a positive slope.
