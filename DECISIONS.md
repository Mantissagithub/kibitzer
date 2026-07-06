# Kibitzer — Training Decision Log

Goal: train a **3000+ Elo** chess model that can compete with Stockfish while keeping
the training path runnable on an **RTX 4060 Laptop GPU with 8 GB VRAM**.

Only decisions that change the model, data, target, loss, optimization, curriculum,
or checkpoint-promotion rule belong in this file.

## D1 — Reject offline best-move distillation on generated histories

The original history-dependent 28M model was distilled on **100,323** Stockfish-14.1
positions labeled at depth 12. Positions came from generated games with three random
opening plies and opponent levels from 1320 to 2850. Training used batch 32 and peak
LR `3e-4`.

Epoch 1 scored **0% vs SF-1320** and **0% vs SF-1800**, compared with the supervised
baseline at 42% and 17%. This reproduced the earlier collapse: generated histories
did not match the histories encountered during real play, and additional teacher
forcing destroyed the fragile supervised policy.

Decision: stop after epoch 1. Do not use offline single-best-move distillation on
generated histories again.

## D2 — Reject ChessBot on-policy distillation

`Maxlegrec/ChessBot` was strong enough to serve as a teacher: in a small local bracket
it scored 100% vs SF-1500, 100% vs SF-2000, and 88% vs SF-2500. The student generated
its own positions; ChessBot supplied a dense legal-move distribution and scalar value.
The loss was generalized JSD on policy logits, value MSE, and KL to the frozen initial
student. JSD inputs used shape `(B, 1, 4672)` so TRL's `batchmean` reduction did not
divide the policy term by the vocabulary size.

The aggressive run (`lr=1e-4`, two epochs) scored **0% vs SF-1320**. A trust-region
retry (`lr=2e-5`, reference KL 0.5, student temperature 0.4, one epoch) only matched
the baseline in the paired gate: both scored **2/8 vs SF-1350**.

Decision: stop ChessBot policy distillation. On-policy sampling fixed the history
distribution but did not prevent catastrophic forgetting.

## D3 — Bound AlphaZero-style training by raw-policy preservation

A search-coupled probe started from the supervised checkpoint and used SF-1350,
32 PUCT simulations, material weight 0.85, Stockfish depth-4 value targets,
`lr=5e-5`, reference KL 0.2, four games, and 20 updates.

The first iteration preserved raw play at **2/8 vs SF-1350** and improved the small
search gate to **3/4**, versus the supervised checkpoint at 2.5/4. Continuing for two
iterations at `lr=3e-5` collapsed raw play to **0/8**.

Decision: a search improvement cannot promote a checkpoint if raw policy strength
regresses. Additional AZ iterations from a degraded checkpoint are forbidden; raw
play is the seed-quality constraint.

## D4 — Train dense targets only on real positions

To remove generated-history mismatch, the next base was trained from scratch on
**11,788,146 positions** from 272,548 Lichess Elite games. ChessBot supplied dense
policy targets and values. Training used context window 8, effective batch 256, cosine
LR, and streamed shards. At step 68,000, policy loss had fallen from 3.35 to about
1.77 and value MSE from 0.28 to 0.056 without optimization collapse.

Gameplay still failed. Against SF-1350, the model scored **3/40 raw** and **0.5/8 with
search**, versus the supervised baseline at 11/40 raw and 4/8 with search.

Decision: real positions are mandatory, but ChessBot's dense policy head is not a
sufficient target. Its playing strength comes from value-based move selection, not
from the policy distribution being distilled.

## D5 — Preserve action-value argmax information in the target

The same real-position pipeline was relabeled with ChessBot values after every legal
move. The first 500k-position run converted values to
`softmax(move_value / 0.1)`. It scored **3% vs SF-1350** at both step 2,000 and step
5,000 and produced near-random policies with entropy about 2.79.

The target was defective: legal-move values were tightly clustered, so temperature
0.1 assigned only 0.14–0.21 probability to the best move. Temperature 0.03 produced
0.34–0.65 and temperature 0.01 produced 0.65–0.93.

Decision: never discard the teacher's decisive argmax through an over-soft target.
Store raw action values and choose temperature at train time, or use a one-hot target
on the value-best move. The failed 500k run does not falsify action-value supervision.

## D6 — Remove game-history dependence

Chess is Markovian for move selection: the current board, side to move, castling
rights, en-passant state, and clocks contain the relevant state. The causal history
trunk created a distribution-matching burden without adding useful move-quality
information and repeatedly overfit to game paths.

Decision: rebuild from scratch as a **single-position model**. Keep the 64-square
position encoder and set the temporal context to one. Initial policy supervision is
one-hot behavioral cloning of moves played in 2300/2500+ Lichess Elite games. Initial
value supervision is separated from policy training so a weak value target cannot
damage the policy representation.

## D7 — Use staged policy training followed by a frozen-representation value fit

The clean rebuild trained policy on **5M positions per epoch for three epochs** with
batch 128. The value head was then trained alone on **250k game-disjoint positions**
labeled by Stockfish depth 14; 224,990 positions were used for training and 25,010 for
evaluation.

Value performance peaked at epoch 4: MSE **0.0637**, Pearson **0.5235**, sign accuracy
66.28%, and R2 **0.2736**. Epoch 5 improved sign accuracy only to 66.58% while MSE,
Pearson, and R2 regressed.

At 64 PUCT simulations the model scored 10% vs SF-1320; at 256 simulations it scored
5%. Deeper search reduced capped ACPL from 126.7 to 118.7 cp and major blunders from
51 to 42, but did not improve match score.

Decision: retain the clean single-position policy base. Stop the value-only stage at
epoch 4 and do not treat more simulations as a substitute for a better value model.

## D8 — Joint Stockfish distillation must preserve every epoch and use constrained selection

Joint distillation started from the clean value checkpoint. Stockfish depth-14
MultiPV-8 labeled 250k cached positions. Both heads, final norm, and the last three
trunk blocks were trainable; earlier layers stayed frozen. Training used batch 128,
five epochs, `lr=1e-4`, and equal policy/value loss weights.

Policy CE improved from 2.6014 to **2.5801** by epoch 3, but value quality peaked at
epoch 1: MSE **0.0717**, sign accuracy **66.74%**, and R2 **0.2872**. The scalar selector
`policy_CE + value_MSE` incorrectly selected epoch 3 because CE changes dominated MSE
changes. That checkpoint scored 20% at 64 simulations and 5% at 256 simulations.

Decision: save every epoch. Multi-head selection must apply value and policy floors
before ranking candidates; unlike metrics must not be added without normalization.
The joint checkpoint is rejected because deeper search amplified its value errors.

## D9 — Gate training changes on a locked common oracle

Training decisions use real positions from unseen PGN months. A depth-20 Stockfish
oracle contains separate game-disjoint validation and test splits, each with 200
positions in four absolute-score bins: `<0.5`, `0.5–2`, `2–5`, and `>5` pawns.
All checkpoints use `clip(cp / 1000, -1, 1)` as the value transform.

Policy evaluation reports exact-best accuracy, within-50cp accuracy, mean/p90/p95
regret, and paired bootstrap intervals. Value evaluation reports MAE and sign accuracy
per score bin. PUCT configuration is selected on validation; the test split is consumed
only after a candidate passes validation.

Decision: match WDL and aggregate MSE alone are too noisy to direct laptop-scale
training. A checkpoint must show paired improvement and preserve tactical sanity before
it can replace the current base.

## D10 — Repair value targets only after auditing teacher-label headroom

A deterministic audit re-evaluated 3,000 cached depth-14 positions at depth 20.
Bounded-value MAE was **0.0217**, sign disagreement 2.90%, and decisive/won-bin sign
disagreement 0%. Teacher depth was therefore far below the model's 0.165 validation
MAE and was not the bottleneck.

The first repair trained only the value head with inverse-frequency score-bin sampling
(exponent 0.5, maximum weight 1.894), three epochs, and `lr=1e-4`. Epoch 1 improved
decisive sign from 66.5% to 68.5% and natural-weighted MAE from 0.16527 to 0.16341.
At 64 simulations it improved mean regret by 8.52 cp over the previous checkpoint, but
the paired 95% interval was `[-2.38, 20.68]`; near-best improvement was also
indistinguishable from zero.

Adding final RMSNorm capacity with policy KL anchoring regressed held-out value metrics
despite 99.84% policy top-1 agreement. The bounded last-block repair also failed to
produce a promotable checkpoint.

Decision: stop this value-repair lineage. Label quality had sufficient headroom;
generalization, not Stockfish depth or policy drift, was the failure.

## D11 — Reject offline value metrics as a strength proxy

A full model trained from random initialization with joint policy and game-result value
targets improved decisive sign from 65.95% to **72.62%** and won-position sign from
81.32% to **86.73%**. It simultaneously reduced overall sign accuracy from 66.58% to
63.50% and Pearson from 0.5226 to 0.4796.

The offline gains did not transfer to play. Against SF-1320, the existing cp-value
model and the joint-scratch model both scored 0.100 at 64 simulations; at 256 they
scored 0.225 and 0.175. Increasing the cp-value model to 512 simulations produced
0.200, within noise of the 256-simulation result.

Decision: do not optimize value-head metrics in isolation. Strength must be validated
in paired play or move-regret evaluation, and search beyond 256 simulations is not a
training path by itself.

## D12 — Replace point experiments with a controlled scaling study

All subsequent base-model work uses an attention-first family and measures policy
scaling at fixed data and data scaling at fixed model size. The primary metric is
held-out policy cross-entropy on game-disjoint Lichess Elite moves; top-1 move match is
the capability metric and value MSE is secondary. LR transfers by width and uses warmup
plus cosine decay.

The measured model ladder is:

| tag | width | trunk blocks | parameters |
|---|---:|---:|---:|
| S0 | 128 | 6 | 2.98M |
| S1 | 192 | 8 | 7.43M |
| S2 | 256 | 10 | 14.89M |
| S3 | 320 | 10 | 22.89M |

Decision: change one scaling variable at a time. Do not infer architectural limits
from one fixed 32M-class model or from noisy short matches.

## D13 — Data, not parameter count, is the dominant scaling lever

At a fixed 5M positions, the parameter sweep produced:

| model | policy CE | top-1 | value MSE |
|---|---:|---:|---:|
| S0 | 2.3570 | 30.20% | 0.7118 |
| S1 | 2.3339 | 30.78% | 0.7213 |
| S2 | 2.3288 | 30.92% | 0.7177 |
| S3 | 2.3097 | 31.61% | 0.7211 |

A 7.7x parameter increase reduced CE by only 0.047 and improved top-1 by 1.41 points.
Value MSE was flat. The policy curve had not saturated, but every rung was severely
undertrained.

Holding S2 and `lr=1.5e-4` fixed, increasing data from 5M to 20M positions reduced CE
from 2.3288 to **2.0437**, increased top-1 from 30.92% to **37.65%**, and reduced value
MSE from 0.7177 to 0.6999. CE improved about **0.1425 per data doubling**, versus
0.0147 per parameter doubling.

Decision: prioritize more unique supervised positions before increasing width. At the
measured scale, data is approximately 10x the policy-loss lever of parameters.

## D14 — Make chess-relative Shaw attention the encoder default

The original encoder used only a learned absolute square embedding. A matched S2,
20M-position A/B test replaced encoder attention with Shaw-style relative terms indexed
by `(file_delta, rank_delta)` over 225 buckets. The temporal trunk remained unchanged
because its sequence length is one.

| metric | absolute | Shaw |
|---|---:|---:|
| parameters | 14.89M | 15.22M |
| policy CE | 2.0437 | **2.0382** |
| top-1 | 37.65% | **38.56%** |
| value MSE | 0.6999 | **0.6987** |

The 0.91-point top-1 gain was about four standard errors on the 50k-position evaluation
slice and cost about 2% more parameters. At 40M positions, Shaw S2 reached **44.52%**
top-1.

Decision: Shaw relative attention is the default for every newly trained model.
Checkpoint-stored configuration preserves compatibility with older absolute models.

## D15 — Reject TDLeaf value training

TDLeaf trained only the value head and final RMSNorm of the 40M-position Shaw S2 base.
The encoder, trunk, and policy head remained frozen. Training alternated four games
against a Stockfish curriculum with TD-lambda updates from PUCT root values
(`lambda=0.7`, 64 simulations, `lr=1e-4`) for approximately 200 games.

A controlled 256-simulation gate used identical openings:

| model | SF-1320 | SF-1900 |
|---|---:|---:|
| untrained Shaw base | 0.975 | 0.738 |
| TDLeaf | 0.975 | 0.762 |

The SF-1900 delta was +0.025 with standard error about 0.085. TDLeaf changed losses
into draws but added no measurable score.

Decision: stop value-head and self-play repair. The strength jump came from supervised
data, Shaw encoding, and inference search, not TDLeaf.

## D16 — Scale Shaw S2 to 100M supervised positions with in-loop play gates

The current strongest base trained Shaw S2 on **100M Lichess Elite positions** in fp32
with constant `lr=1.5e-4`. The run took 15.3 hours on the RTX 4060 Laptop GPU. Held-out
evaluation used Lichess Elite 2025-11. Every 20M positions, the current model played 20
games against SF-1900 using 64 PUCT simulations.

| positions | top-1 | SF-1900 score at 64 sims |
|---:|---:|---:|
| 20M | 38.56% | 0.250 |
| 40M | 44.52% | 0.350 |
| 60M | — | 0.650 |
| 80M | — | 0.775 |
| 100M | **49.45%** | **0.825** |

Policy CE reached **1.57255** and value MSE reached **0.64971**. No offline or play
plateau appeared by 100M positions.
At 256 simulations, the 100M checkpoint scored 0.900 vs SF-1900, 0.700 vs SF-2100,
and 0.833 vs SF-2300. The non-monotonic opponent results make the fitted 2470 estimate
noise-inflated; the defensible current range is approximately **2280–2470 Elo**.

Decision: the 100M Shaw S2 checkpoint is the current training champion. Supervised
data scaling remains the only training lever with monotonic offline and play evidence.

## D17 — Next experiment: test capacity at matched 100M data

The next run is a controlled **S3 Shaw at 100M positions** experiment. It must use the
same training PGN pool, held-out month, target construction, optimizer schedule, and
effective batch as D16. Use width-transferred LR `1.2e-4`; use gradient accumulation
as needed to stay within 8 GB VRAM. Save checkpoints and evaluate offline every 20M
positions so the S3 curve is directly paired with S2.

Promotion criteria:

1. On the same 50k-position held-out slice, S3 must reach policy CE below **1.57255**
   and top-1 of at least **49.95%**. The 0.50-point top-1 margin is more than twice the
   approximately 0.22-point standard error at this sample size.
2. Compare S3 directly with S2 over identical openings at 256 simulations against
   SF-1900, SF-2100, and SF-2300, with at least 40 games per level. The pooled paired
   bootstrap interval for score difference must exclude zero, and score must decrease
   monotonically as opponent Elo increases.
3. If S3 misses either criterion at 100M, reject the capacity increase and run the next
   controlled point as S2 Shaw at 200M positions.

No value repair, TDLeaf, OPD, or offline teacher distillation is allowed in this run.
This experiment isolates whether additional capacity becomes useful only after the
data starvation identified in D13 has been reduced.

---

## D44 — measured Elo of the 100M shaw model (calibrated, 160 games)

Proper Elo measurement of runs/scaling_shaw_data/checkpoints/S2_shaw_100M.pt via PUCT (256 sims) vs
Stockfish UCI_Elo, 40 games/level over 4 levels (real 20-line opening book, alternating colors):

| opponent | score | W/D/L |
|---|---|---|
| SF-1900 | 0.938 | 37/1/2 |
| SF-2100 | 0.850 | 31/6/3 |
| SF-2300 | 0.700 | 24/8/8 |
| SF-2500 | 0.525 | 13/16/11 |

Overall 0.753 across 160 games. **Final iterative performance Elo = 2483 (±~32)** — computed with the
proper sequential update R += K(S−E), E = 1/(1+10^((R_opp−R)/400)), K=16, iterated to convergence from
R0=2280. Monotonic ladder; the model is ~even (0.525) with SF-2500, sweeps SF-1900.

Supersedes the earlier single-anchor ~2280 (D43) with a multi-level calibrated fit. Note: the prior cloud
attempt (random 8-ply openings) gave a spurious ~1697 — random openings drop the model into untrained
positions and are not comparable; the fixed opening book is essential. Best 37 wins vs SF-2300/2500 saved to
reports/scaling_law/elo_local/best_games.pgn. Method uses the repo's eval_search_vs_stockfish.py (opening book
extended 5→20 lines). Cloud pod was terminated (~$0.30 spent).
