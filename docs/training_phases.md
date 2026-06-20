# Training Phases

This is the map of what I am training, why each phase existed, what signal it
used, and what the result means. The main rule is simple: a phase only matters
if it survives a direct chess gate. Loss curves are useful for debugging, but
the checkpoint has to play better chess.

## Goal

Train `kibitzer`, a small history-dependent chess transformer, into a stronger
raw chess policy first. Search and self-play should come after that, not before.

The working path is:

1. build a raw model that beats the old SFT baseline;
2. gate it against rated Stockfish;
3. only then try search or self-play;
4. stop any phase that improves a metric while making the raw policy worse.

The model predicts:

- a policy over AZ-style move ids;
- a scalar value head;
- moves using short board history, so realistic histories matter.

Architecture diagram: [kibitzer_transformer_architecture.png](kibitzer_transformer_architecture.png).
Editable source: [kibitzer_transformer_architecture.svg](kibitzer_transformer_architecture.svg).

## Core Training Notation

For a position/history `x`:

- student logits: `z_theta(x)`;
- student legal-move policy: `p_theta(a | x)`;
- teacher policy: `q(a | x)`;
- student value: `v_theta(x)`;
- teacher or game value target: `y`.

Legal masking means illegal moves get zero probability before the loss:

```text
p_theta(a | x) = softmax(mask_legal(z_theta(x)))_a
```

The dense supervised policy loss is:

```text
L_policy(x) = - sum_a q(a | x) log p_theta(a | x)
```

The scalar value loss used in the current base run is:

```text
L_value(x) = (tanh(v_theta(x)) - y)^2
```

The current total supervised loss is:

```text
L_base = L_policy + lambda_value L_value
```

In the current trainer, `lambda_value = 1.0`.

## Phase 0: SFT Baseline

Purpose: get a legal, usable starting chess model.

The repo has `checkpoints/sft_best.pt`. This is the old baseline. It matters
because every later checkpoint needs to beat it under the same gate. It should
not be treated as sacred, because several later warm-started runs showed that
the SFT basin is fragile.

What this phase gave:

- legal move behavior;
- a fixed baseline checkpoint;
- enough chess knowledge to test search;
- a value head that was not reliable enough.

The basic SFT objective is normal next-move cross entropy:

```text
L_sft = - log p_theta(a_game | x)
```

This is weaker than dense policy training because it says only "the played move
was this". It does not tell the model which other legal moves were also good,
which were losing, or how close the alternatives were.

Result:

- useful baseline;
- not strong enough as the final model;
- fragile when used as the seed for more aggressive training.

Current use: keep as `sft` in gate comparisons.

## Phase 1: Stockfish Best-Move Distillation

Purpose: cheaply imitate an engine.

The idea was: ask Stockfish for strong moves, train the student to copy those
moves, then check whether the student climbs the Stockfish ladder.

The simplified objective was still close to hard-label imitation:

```text
L_sf_best = - log p_theta(a_stockfish | x)
```

where `a_stockfish` is the engine's chosen move.

Why this looked reasonable:

- Stockfish is far stronger than the student;
- labeling is conceptually simple;
- a small 100k-position dataset is cheap enough to generate and train on.

Why it did not work:

- the dataset was too small for chess policy learning;
- some histories were unrealistic for a history-dependent net;
- a single best move is a thin target;
- the value head still did not get enough reliable structure;
- the gate did not show real improvement.

Result:

- the run did not beat the SFT baseline;
- later evals showed collapse rather than useful strength;
- this phase is retired.

Lesson: for this model, "copy Stockfish's top move on 100k positions" is not the
same as learning a chess policy. The model needs many real positions and a dense
distribution over legal moves.

## Phase 2: ChessBot On-Policy Distillation

Purpose: use a stronger teacher on positions the student actually reaches.

`Maxlegrec/ChessBot` was verified as a strong teacher, roughly in the 2500-2600
range in quick local brackets. This made it a useful teacher for a small student.

The OPD loop was:

1. the student plays/self-plays and creates positions;
2. ChessBot labels those positions;
3. the student matches ChessBot while trying not to drift too far from the
   reference policy.

The policy matching used a distributional loss, closer to:

```text
L_teacher = CE(q_teacher, p_theta)
          = - sum_a q_teacher(a | x) log p_theta(a | x)
```

For OPD, a reference term was also important:

```text
L_ref = KL(p_theta(. | x) || p_ref(. | x))
```

A combined version looks like:

```text
L_opd = L_teacher + beta L_ref + lambda_value L_value
```

Some code also used generalized JSD-style policy matching. In plain terms, that
penalizes disagreement between the teacher distribution and student distribution
without treating the teacher as a one-hot move.

Why this phase was better motivated than Phase 1:

- positions came from the student, so they were on-policy;
- ChessBot gave dense legal-move probabilities;
- the teacher was strong enough to correct weak play;
- reference pressure was meant to prevent sudden collapse.

What happened:

- aggressive OPD collapsed;
- gentler OPD did not clearly beat SFT;
- the SFT seed still looked like the weak point.

Gate evidence from the Prime run:

| checkpoint / mode | opponent | games | result | score |
|---|---:|---:|---|---:|
| `checkpoints/sft_best.pt` raw | SF-1350 | 8 | 1W / 5L / 2D | 2.0/8 |
| `runs_rl/opd_gentle.pt` raw | SF-1350 | 8 | 2W / 6L / 0D | 2.0/8 |

Result: OPD did not give monotonic improvement.

Lesson: on-policy distillation is not automatically safe. If the student seed is
weak, the on-policy distribution can still be low quality, and gradient updates
can erase useful raw behavior.

## Phase 3: AZ / Search Probes

Purpose: test whether search can make the model stronger, and whether
search-guided training can improve the policy.

The AlphaZero-style idea is:

- use the current network to guide search;
- let search produce a stronger policy target `pi`;
- train the network toward `pi`;
- train the value head toward outcome or evaluation targets.

The classic search-policy loss shape is:

```text
L_az_policy = - sum_a pi(a | x) log p_theta(a | x)
```

The value side is:

```text
L_az_value = (tanh(v_theta(x)) - y_search_or_game)^2
```

So the training loss is roughly:

```text
L_az = L_az_policy + lambda_value L_az_value + beta L_ref
```

In this repo, the reference term matters because previous runs showed raw-policy
damage. The purpose of `L_ref` is to keep the model from moving too far away
from a known usable policy:

```text
L_ref = KL(p_theta(. | x) || p_ref(. | x))
```

What worked:

- inference-time search helped;
- one short AZ/search probe improved search-mode score in a tiny gate.

Gate evidence:

| checkpoint / mode | opponent | games | score |
|---|---:|---:|---:|
| SFT raw | SF-1350 | 8 | 2.0/8 |
| SFT + MCTS | SF-1350 | 4 | 2.5/4 |
| AZ iter 1 raw | SF-1350 | 8 | 2.0/8 |
| AZ iter 1 + MCTS | SF-1350 | 4 | 3.0/4 |
| AZ continuation raw | SF-1350 | 8 | 0.0/8 |

What failed:

- continuation damaged the raw policy;
- search score alone was not enough;
- a checkpoint that needs search to look good is not a strong new base.

Result: keep the first AZ probe as an experimental search-compatible checkpoint,
but do not promote AZ continuation.

Lesson: search is useful, but training from weak policies is dangerous. Raw
strength preservation is the invariant.

## Phase 4: Current Searchless Dense-Label Base

Purpose: train a fresh base from real chess positions, with dense teacher
targets, without self-play and without search.

This is the current main phase. It is the cleanest response to the failures
above:

- do not warm-start from the fragile SFT checkpoint;
- do not use tiny synthetic/history-broken data;
- do not rely on self-play before the model is strong;
- train on many real game positions;
- use dense legal-move teacher targets.

The current data recipe:

1. read real Lichess PGNs from `data/raw`;
2. walk real game histories;
3. label positions using `Maxlegrec/ChessBot`;
4. store sparse sharded labels under `data/chessbot_labeled`;
5. stream shards during training so RAM stays bounded.

Current labeled dataset:

| field | value |
|---|---:|
| games | 272,548 |
| positions | 11,788,146 |
| shards | 546 |
| min Elo | 2200 |
| position stride | 2 |
| teacher | `Maxlegrec/ChessBot` |

The current base objective is:

```text
L_base = - sum_a q_chessbot(a | x) log p_theta(a | x)
       + (tanh(v_theta(x)) - y_chessbot)^2
```

This is different from SFT:

- SFT says: imitate the one move that was played;
- dense labels say: learn the teacher's full legal-move preference shape;
- value labels say: also learn whether the position is good or bad.

This is also different from self-play:

- self-play depends on the current model's own distribution;
- this phase depends on real human game histories plus a stronger teacher;
- the model is not asked to bootstrap from itself yet.

Current training command shape:

```bash
.venv/bin/python scripts/train_shards.py \
  --data-dir data/chessbot_labeled \
  --output-dir runs_base \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 64 \
  --grad-accum-steps 4 \
  --epochs 2 \
  --num-workers 6 \
  --prefetch-factor 4
```

Important implementation details:

- `batch_size` is the micro-batch;
- `grad_accum_steps` gives the effective batch;
- current effective batch is `64 * 4 = 256`;
- shards are streamed instead of loaded all at once;
- CPU workers pre-encode batches so the GPU does not starve;
- checkpoints are written every 1000 optimizer steps.

Current run snapshot from `runs_base/train.log` on 2026-06-20:

| metric | latest seen value |
|---|---:|
| step | 44,700 / 92,096 |
| loss | 1.8707 |
| policy loss | 1.8103 |
| value loss | 0.0605 |
| entropy | 1.935 |
| learning rate | 1.64e-04 |
| latest saved checkpoint | `runs_base/base_step_0044000.pt` |

This snapshot is just progress. The real question is whether the final or best
checkpoint beats SFT in direct play.

## Phase 5: Gate Before Belief

Purpose: measure chess strength directly.

The gate exists because earlier phases looked plausible while training and still
failed when they played games. A gate is the real promotion test.

For `N` games, the score is:

```text
score = wins + 0.5 * draws
win_rate = score / N
```

The key comparison is not just "did it win games?" It is:

```text
Delta = score(candidate) - score(baseline)
```

under the same opponent, time controls, device settings, and search settings.

Minimum raw gate shape:

```bash
uv run python scripts/gate_stockfish.py \
  --checkpoint sft=checkpoints/sft_best.pt \
  --checkpoint base=runs_base/base_final.pt \
  --elo 1350 \
  --n-games 16
```

What to check:

- raw model first;
- search model second;
- candidate versus SFT under identical settings;
- no promotion if raw strength regresses;
- no belief from loss alone.

If raw improves, then search can be tested. If raw fails, search is a side tool,
not a new training seed.

## Phase 6: Later Self-Play / Anchored AZ

Purpose: improve a strong base, not rescue a weak one.

Self-play should start only after the fresh supervised base passes the gate. If
the base is weak, self-play mostly samples weak positions and trains the model
on its own bad distribution.

The later anchored AZ objective should look like:

```text
L_anchored_az =
    CE(pi_search, p_theta)
  + lambda_value L_value
  + beta KL(p_theta || p_ref)
  + gamma CE(q_anchor, p_theta)
```

where:

- `pi_search` is the MCTS/search policy;
- `p_ref` is a frozen reference policy;
- `q_anchor` is supervised anchor data from the strong base/data mixture;
- `beta` controls how hard the model is kept near the reference;
- `gamma` controls how much supervised anchor replay remains.

This phase should be guarded:

- start from the best gated supervised base;
- keep anchor replay;
- use small learning rates;
- gate raw and search modes separately;
- stop immediately if raw strength drops.

Status: later, not current.

## Papers This Is Based On

- [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494)
  is the main reason for the current pivot. It supports the idea that strong
  chess behavior can come from supervised training on real positions with dense
  engine-style labels, without explicit search at inference time.
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950)
  is the value-head reference. It motivates HL-Gauss / classification-style
  value training. I am not using it in the current run yet; scalar MSE gets a
  fair test first because the data scale is now much larger.
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
  is the AlphaZero-style reference for the later search/self-play phase. In this
  repo, that idea is delayed and gated because earlier AZ continuation hurt raw
  strength.

## Current Decision

The active phase is Phase 4: train a fresh supervised base on real games with
dense ChessBot labels.

Do now:

- finish the current base training;
- gate against `checkpoints/sft_best.pt`;
- compare raw strength first;
- only use search/self-play after a raw improvement exists.

Do not do now:

- restart old Stockfish best-move distillation;
- continue OPD from the old fragile seed;
- promote a search-only checkpoint;
- start self-play before the fresh base passes a direct gate.
