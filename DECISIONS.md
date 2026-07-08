# Kibitzer , Training Decision Log

Goal: train a **3000+ Elo** chess model that can compete with Stockfish while keeping
the training path runnable on an **RTX 4060 Laptop GPU with 8 GB VRAM**.

Only decisions that change the model, data, target, loss, optimization, curriculum,
or checkpoint-promotion rule belong in this file.

## D1 , Reject offline best-move distillation on generated histories

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

## D2 , Reject ChessBot on-policy distillation

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

## D3 , Bound AlphaZero-style training by raw-policy preservation

A search-coupled probe started from the supervised checkpoint and used SF-1350,
32 PUCT simulations, material weight 0.85, Stockfish depth-4 value targets,
`lr=5e-5`, reference KL 0.2, four games, and 20 updates.

The first iteration preserved raw play at **2/8 vs SF-1350** and improved the small
search gate to **3/4**, versus the supervised checkpoint at 2.5/4. Continuing for two
iterations at `lr=3e-5` collapsed raw play to **0/8**.

Decision: a search improvement cannot promote a checkpoint if raw policy strength
regresses. Additional AZ iterations from a degraded checkpoint are forbidden; raw
play is the seed-quality constraint.

## D4 , Train dense targets only on real positions

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

## D5 , Preserve action-value argmax information in the target

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

## D6 , Remove game-history dependence

Chess is Markovian for move selection: the current board, side to move, castling
rights, en-passant state, and clocks contain the relevant state. The causal history
trunk created a distribution-matching burden without adding useful move-quality
information and repeatedly overfit to game paths.

Decision: rebuild from scratch as a **single-position model**. Keep the 64-square
position encoder and set the temporal context to one. Initial policy supervision is
one-hot behavioral cloning of moves played in 2300/2500+ Lichess Elite games. Initial
value supervision is separated from policy training so a weak value target cannot
damage the policy representation.

## D7 , Use staged policy training followed by a frozen-representation value fit

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

## D8 , Joint Stockfish distillation must preserve every epoch and use constrained selection

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

## D9 , Gate training changes on a locked common oracle

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

## D10 , Repair value targets only after auditing teacher-label headroom

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

## D11 , Reject offline value metrics as a strength proxy

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

## D12 , Replace point experiments with a controlled scaling study

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

## D13 , Data, not parameter count, is the dominant scaling lever

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

## D14 , Make chess-relative Shaw attention the encoder default

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

## D15 , Reject TDLeaf value training

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

## D16 , Scale Shaw S2 to 100M supervised positions with in-loop play gates

The current strongest base trained Shaw S2 on **100M Lichess Elite positions** in fp32
with constant `lr=1.5e-4`. The run took 15.3 hours on the RTX 4060 Laptop GPU. Held-out
evaluation used Lichess Elite 2025-11. Every 20M positions, the current model played 20
games against SF-1900 using 64 PUCT simulations.

| positions | top-1 | SF-1900 score at 64 sims |
|---:|---:|---:|
| 20M | 38.56% | 0.250 |
| 40M | 44.52% | 0.350 |
| 60M | , | 0.650 |
| 80M | , | 0.775 |
| 100M | **49.45%** | **0.825** |

Policy CE reached **1.57255** and value MSE reached **0.64971**. No offline or play
plateau appeared by 100M positions.
At 256 simulations, the 100M checkpoint scored 0.900 vs SF-1900, 0.700 vs SF-2100,
and 0.833 vs SF-2300. The non-monotonic opponent results make the fitted 2470 estimate
noise-inflated; the defensible current range is approximately **2280–2470 Elo**.

Decision: the 100M Shaw S2 checkpoint is the current training champion. Supervised
data scaling remains the only training lever with monotonic offline and play evidence.

## D17 , Next experiment: test capacity at matched 100M data

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

## D44 , measured Elo of the 100M shaw model (calibrated, 160 games)

Proper Elo measurement of runs/scaling_shaw_data/checkpoints/S2_shaw_100M.pt via PUCT (256 sims) vs
Stockfish UCI_Elo, 40 games/level over 4 levels (real 20-line opening book, alternating colors):

| opponent | score | W/D/L |
|---|---|---|
| SF-1900 | 0.938 | 37/1/2 |
| SF-2100 | 0.850 | 31/6/3 |
| SF-2300 | 0.700 | 24/8/8 |
| SF-2500 | 0.525 | 13/16/11 |

Overall 0.753 across 160 games. **Final iterative performance Elo = 2483 (±~32)** , computed with the
proper sequential update R += K(S−E), E = 1/(1+10^((R_opp−R)/400)), K=16, iterated to convergence from
R0=2280. Monotonic ladder; the model is ~even (0.525) with SF-2500, sweeps SF-1900.

Supersedes the earlier single-anchor ~2280 (D43) with a multi-level calibrated fit. Note: the prior cloud
attempt (random 8-ply openings) gave a spurious ~1697 , random openings drop the model into untrained
positions and are not comparable; the fixed opening book is essential. Best 37 wins vs SF-2300/2500 saved to
reports/scaling_law/elo_local/best_games.pgn. Method uses the repo's eval_search_vs_stockfish.py (opening book
extended 5→20 lines). Cloud pod was terminated (~$0.30 spent).

---

## D45 , tactical mid-training is NEUTRAL for play strength

Tested whether curated tactical mid-training lifts the 100M shaw model's measured Elo. Recipe: continue from
S2_shaw_100M.pt on 20M positions, 30% Lichess puzzle-DB positions (solver moves, rating 1200-2400) / 70% elite
game positions, LR 5e-5, **value_weight=0** (a 25k smoke showed the puzzle value=+1 targets spike value MSE
0.65->0.96; value_weight=0 fixed it , final value MSE +0.012, top-1 -0.4pp). 150.8 min. Script:
scripts/train_midtrain.py.

Re-measured with the D44 protocol (160 games, 256 sims, SF-1900/2100/2300/2500 @ 40 each, iterative Elo):

| opponent | tactical | base |
|---|---|---|
| SF-1900 | 0.887 | 0.938 |
| SF-2100 | 0.875 | 0.850 |
| SF-2300 | 0.600 | 0.700 |
| SF-2500 | 0.637 | 0.525 |

**Tactical Elo = 2464 (±~32) vs base 2483 (±~32) → Δ −19, NEUTRAL** (CIs overlap heavily). The model traded a
little at SF-2300 for a little at SF-2500; net no change. Verdict: puzzle-based tactical mid-training does not
move search play strength at this scale , consistent with D40/D35 (policy/value refinements don't shift play;
the live levers remain arch + data + search depth). Best tactical wins vs SF-2300/2500 saved to
reports/scaling_law/elo_tactical/best_games.pgn. Tactical checkpoint NOT pushed to HF (no improvement).

---

## D46 , honest calibrated strength: Maia sweep + the 2483 caveat

The measured Elo (D44, 2483 vs Stockfish UCI_Elo) is on Stockfish's handicapped UCI_LimitStrength scale, which
randomizes/weakens SF and is widely known to be softer than real Elo , so 2483 is INFLATED and relative-only,
useful for ranking our own checkpoints but not a real-world rating.

Re-measured vs Maia (lc0 + maia weights at nodes=1, calibrated to real Lichess blitz Elo, caps at 1900), same
protocol (256 sims, 40 games/level, real opening book): **159W/1D/0L over 160 games (score 0.997)** ,
Maia-1300 1.000, Maia-1500 0.988, Maia-1700 1.000, Maia-1900 1.000. Total domination => the 100M shaw model is
comfortably **>1900 Lichess blitz** (a real, honest lower bound); Maia cannot measure how far above because it
caps at 1900. Next: Leela ceiling test (lc0 + t1-256x10-distilled net at 1/8/32 nodes ~2700/2850/2950) to
bracket the true value. lc0 built from source (eigen CPU backend), persistent binary at data/leela/lc0.
Methodology lesson: UCI_Elo Stockfish is NOT a real-Elo yardstick; use human-calibrated Maia / real engines.

---

## D47 , true strength bracketed at ~2500-2600 Elo (Leela ceiling test)

Ran the 100M shaw model (256 sims) vs lc0 + Leela net (t1-256x10-distilled, eigen CPU) at 1/8/32 nodes to
find the ceiling (Maia caps at 1900, couldn't measure it). 24 games/setting:

| opponent | approx Elo | score | W/D/L |
|---|---|---|---|
| Leela @ 1 node | ~2700 | 0.354 | 3/11/10 |
| Leela @ 8 nodes | ~2850 | 0.104 | 2/1/21 |
| Leela @ 32 nodes | ~2950 | 0.042 | 0/2/22 |

Implied Elo from the Leela@1node score: **~2596**. This CONVERGES with the Stockfish UCI_Elo number (2483) =>
**honest strength ~2500-2600 Elo**. Maia's ">1900" (D46) was a loose floor; the UCI 2483 (D44) was NOT
badly inflated after all (within ~100 of the Leela estimate). Hard ceiling <2700: the model loses almost
every game once Leela gets even 8 nodes of search. Different rating pools (Lichess/UCI/Leela) won't agree
exactly, but three independent measurements bracket ~2500-2600. Verdict: a genuinely strong ~2550 engine
(expert/CM-ish), well short of the 3000+/beat-Stockfish goal. Best Kibitzer wins vs Leela saved to
reports/scaling_law/elo_leela/best_games.pgn. 512-sim Maia-1900 retest: 29W/1D/0L (0.983), sweep robust to sims.

---

## D48 , self-play smoke: 1-iteration AZ-lite REGRESSED the model (negative)

Tested the one untried lever , self-play from the strong 100M base (past self-play/RL failures were all on the
weak base). AZ-lite smoke via scripts/selfplay_smoke.py: ~150 self-play games @64 sims from a varied opening
book with temperature exploration (first 16 plies), recording (position, search-argmax-best move, game-outcome
value z) => 14,089 positions. Behavioral-cloned the base toward those (policy CE to the search-best move +
value MSE to z), lr 2e-5, 3 epochs (loss 0.98->0.85->0.76, stable, no collapse). Head-to-head vs the base,
40 games @64 sims: **selfplay_v1 = 10W/11D/19L, score 0.388** => v1 LOST to its own parent.

Verdict: **negative** , 1-iteration AZ-lite self-play did not improve and mildly regressed the model. It did NOT
collapse (unlike the old weak-base attempts that hit 0), and this is a crude approximation of real AZ (hard
argmax target not the visit distribution; only 14k positions; 1 iteration; self-play @64 sims produces
~2500-level games that aren't clearly better than the 2500+ human-elite training data). So it doesn't rule out
a proper multi-iteration AZ with soft policy targets + far more games/sims, but as a cheap smoke it says
self-play is NOT an easy win here.

This is the 4th fine-tuning/self-play negative (D35 value-repair, D40 TDLeaf, D45 tactical, D48 self-play).
Consistent theme: the supervised base is a strong local optimum that cheap post-hoc methods degrade. The lever
with actual evidence remains SCALE (params S3+ / more data), per D43. Artifacts: scripts/selfplay_smoke.py
(gen/train/match modes), runs/selfplay_smoke/selfplay_v1.pt, reports/scaling_law/selfplay_smoke/.

---

## D49 , proper AlphaZero (1 iteration) regressed MORE than simple BC (negative)

Followed up D48's crude BC smoke with a PROPER AZ setup: added optional Dirichlet root noise to
kibitzer/search.py (puct_search, defaults off), and scripts/selfplay_az.py trains the policy toward the full
MCTS VISIT DISTRIBUTION (soft cross-entropy over the 4672 actions) + value toward the game outcome. One
iteration: ~120 self-play games @128 sims with Dirichlet noise (alpha 0.3, eps 0.25), 10,730 positions, lr 2e-5,
4 epochs (policy 1.83->1.72, value 0.29->0.14, clean/no collapse). Head-to-head vs the 100M base, 40 games
@128 sims: **az_v1 = 1W/13D/26L, A_score 0.1875** => regressed WORSE than the simple-BC smoke (0.388, D48).

Likely mechanism: at 128 sims the visit distribution is DIFFUSE (not sharply peaked), so the soft target
DILUTES the base's decisiveness , the base was trained on sharp one-hot human moves, and regressing toward a
fuzzy self-generated distribution makes it play passively (13 draws) and weaker. The "more correct" AZ objective
was more harmful than hard-argmax BC in this low-sim, single-iteration regime.

Verdict: **negative** , 1-iteration self-play (both hard-BC and proper soft-target AZ) does not improve and
regresses this strong supervised base. This is the 5th fine-tuning/self-play negative (D35, D40, D45, D48, D49).
CAVEAT (fair): a single iteration is NOT how AZ works , real AZ needs MANY iterations with a growing replay
buffer and progressively stronger self-play, plus much higher sims (800+) for sharp targets. That full campaign
is untested here and a large, uncertain-payoff undertaking; and the ~2500-level self-play data (D47) caps how
much new signal exists. Conclusion stands: the only lever with positive evidence is SCALE (params S3+/more data,
D43); self-play/fine-tuning is not a quick win on this base. Artifacts: scripts/selfplay_az.py (gen/train/match),
kibitzer/search.py dirichlet params, runs/selfplay_az/az_v1.pt, reports/scaling_law/selfplay_az/.

---

## D50 , search-axis lab: PUCT tuning neutral, classical alpha-beta collapses (value head is the bottleneck)

Pivoted off the fine-tuning axis to the SEARCH axis: does a different search algorithm over the SAME 100M shaw
net raise its cap? Built search_lab/ (variants.py + compare.py). Each variant is a picker(board, ev, budget)->
move; compared at EQUAL net-eval budget (CountingEvaluator counts evaluate() calls, so mcts sims and alpha-beta
nodes spend the same compute). Variants vs baseline_puct, ~20 games from a 12-line opening book, alternating
colors, budget 128:

CRITICAL calibration , baseline_puct vs an IDENTICAL baseline_puct scored **0.600** (7W/10D/3L). Same
deterministic search on both sides => that 0.60 is the A-side/sample NOISE FLOOR (SE ~0.11 at n=20), NOT a real
edge. Variants must clear ~0.65 to count as real, not 0.5.

| variant                | budget | score  | W/D/L    | avg evals | read                         |
|------------------------|--------|--------|----------|-----------|------------------------------|
| baseline_puct (self)   | 128    | 0.600  | 7/10/3   | 123       | NOISE FLOOR (identical algo) |
| puct_fpu               | 128    | 0.625  | 9/7/4    | 122       | within noise                 |
| puct_prune             | 128    | 0.600  | 6/12/2   | 121       | within noise                 |
| puct_stacked           | 128    | 0.5625 | 9/9/6    | 123       | within noise (no compounding)|
| puct_stacked           | 256    | 0.525  | 3/15/2   | 247       | within noise, 75% draws      |
| alphabeta              | 128    | 0.075  | 0/3/17   | 146       | COLLAPSE                     |
| alphabeta_quiescence   | 128    | 0.025  | 0/1/19   | 139       | COLLAPSE                     |


Verdict: **PUCT tuning is neutral-within-noise.** FPU (-0.2), prior-threshold pruning (0.15), cpuct(s) visit-
scaling, and all three STACKED all land within one SE of the 0.60 self-match floor , no tuning knob adds real
strength, stacking does not compound, and 256 budget just adds draws (no scaling with compute). The ONLY signal
far outside noise is that **classical alpha-beta CATASTROPHICALLY collapses** (0.075 / 0.025), and quiescence
made it WORSE. Mechanism: alpha-beta trusts the value head as a minimax LEAF evaluator, while MCTS averages many
policy-guided rollouts and leans on the strong policy. So the collapse localizes the bottleneck to the
**weak/miscalibrated value head** (D35), not the search. Search is already near its cap for this net.

This is the 6th negative on a non-scale lever (D35 value-repair, D40 TDLeaf, D45 tactical, D48 BC self-play,
D49 proper AZ, D50 search). The search axis is now fenced off like the fine-tuning axis. Two levers survive
with evidence: (1) SCALE (params/data, D43) , the through-line to the project goal; (2) a genuinely better
VALUE HEAD, which would ALSO retroactively unlock alpha-beta/stacked-PUCT. Decision with user: scale on DATA
more next. Artifacts: search_lab/variants.py, search_lab/compare.py, search_lab/results/*.json,
kibitzer/search.py (optional dirichlet params, defaults off).

---

## D51 , competition-data continuation (TWIC 2400+) HELD vs the online base (neutral)

Tested a DATA-QUALITY bet: does swapping training fuel from online-elite (lichess) to elite COMPETITION games
(OTB tournaments) lift strength? Pulled TWIC issues 920-1652 (732 files, 3.49M games), filtered to both players
2400+ (570,268 games = 16.3%, ~49M positions, 86.5 avg plies) into 12 shards. Continued the online 100M base
(warm-start via new scripts/scaling_sweep.py --init-checkpoint) on 11 shards / eval on the 12th, base-lr 8e-5,
eval-every 20M vs SF-1900. Data fix: TWIC mainlines contain null moves ("--") that move_to_index can't encode and
crashed the DataLoader at 39%; added a null-move skip to kibitzer/data.py iter_pgn_samples (push to keep board
states, don't yield as a target) , isolated to the pgn reader, online-elite data was unaffected. The 11 fresh
shards held ~41.6M positions, so training exhausted them at 41.6M (clean cap, NO repetition) => ~142M cumulative
(100M online + 41.6M competition). Snapshot runs/scaling_shaw_comp/S2_shaw_142M_comp.pt. Wall clock 6.3h local.

Results (all vs the online 100M base):
- HEAD-TO-HEAD (comp vs base, puct @128 sims), 30 games: 11W/14D/5L = **0.600**. Net +6 wins but ~1.1 sigma over
  30 games => a mild, NON-significant lean toward the competition model.
- vs SF-1900 @64 sims, SAME eval harness (apples-to-apples): base 14W/3D/3L = **0.775**, comp 21W/5D/4L (30g) =
  **0.783** => DEAD EVEN. (An earlier comp number of 0.900 vs the base's in-loop 0.825 was a harness mismatch ,
  different opening books; matched, both are ~0.78.)
- In-loop SF-1900 curve during the continuation: 20M=0.700, 40M=0.775 , it DIPPED below the base early
  (distribution shift) and recovered by the end. A mid-training checkpoint (~30M) lost the head-to-head 0W/6D/4L
  (0.300); the FINISHED model recovered fully to 0.600 => the dip was transient adaptation, not damage.
- Offline (competition-shard eval, NOT comparable to the online arm's lichess eval): eval_top1 0.4984,
  eval_value_mse 0.402.

Verdict: **HELD (neutral)** , the competition-data continuation produced a model statistically indistinguishable
from the online base: a mild non-significant h2h lean (0.600, 11W-5L/30) and dead-even vs Stockfish
(0.783 vs 0.775). NOT the 7th negative , unlike the six prior fine-tuning attempts (D35/D40/D45/D48/D49/D50) that
REGRESSED or were neutral-worse, competition data did NOT degrade the strong base despite being a distribution
shift AND scarcer data (~42M vs the online 100M). Mildly informative: elite-competition data is at least as good
as online-elite fuel and the base tolerated the swap without collapse. But it is not a breakthrough , no clear
strength gain, and pure competition data caps at ~42M positions (14 years of TWIC), too little to scale further
on its own. SCALE (more data of the same distribution, or bigger params) remains the only lever with a positive
slope (D43). Artifacts: reports/scaling_law_2/ (comp_arm.json, comp_train.log, h2h_vs_base*.json,
sf1900_*.json), runs/scaling_shaw_comp/S2_shaw_142M_comp.pt, scripts/scaling_sweep.py --init-checkpoint,
kibitzer/data.py null-move skip.

---

## D52 , enlarged value head: big offline win, consistently HURTS play (negative)

Tested the one untested structural lever: the value head was only 33,025 params (2-layer d_model//2 MLP) and is
the diagnosed weak link (D50 alpha-beta collapse). Made it config-driven (kibitzer/model.py build_value_head +
KibitzerConfig.value_hidden/value_layers, backward-compatible: hidden<=0 reproduces the legacy head bit-for-bit
so every existing checkpoint loads unchanged and ModelEvaluator/search_lab rebuild the enlarged head from the
saved config). Enlarged to 131,841 params (hidden=256, 3 layers) and retrained ONLY the value head (trunk/
encoder/policy frozen) on the Stockfish depth-14 label cache (250k positions, game-disjoint eval), on BOTH the
100M online base and the 142M competition model. Script scripts/train_value_head_big.py.

Offline: the enlarged head roughly HALVED held-out value MSE , base 0.0403->0.0196 (-51%), comp 0.0569->0.0178
(-69%), Pearson 0.84/0.85 -> 0.90/0.91. Best at epoch 2 (overfits the 250k cache after that). The competition
trunk supported the better head (0.0178 < 0.0196).

Play (the decision): the offline win did NOT transfer , it consistently REGRESSED strength.
- PUCT vs SF-1900 @64 sims, 20g: base 0.775 -> 0.625 (enlarged), comp 0.783 -> 0.65. Both DOWN ~0.13-0.15
  (~1.3 sigma each individually, but the SAME direction on two independent models => a real small regression).
- vs Leela t1-256x10-distilled @ nodes=1 (~2700, lowest-ceiling tactical yardstick), 20g: legacy comp 0.225
  (3W/3D/14L) vs enlarged 0.150 (1W/4D/15L) => worse tactically, not better.
- alpha-beta diagnostic @128 (search_lab, uses the value head directly as a minimax leaf): still collapses at
  ~0.19 (vs D50's ~0.075 with the weak head) , better offline value is STILL not a usable leaf evaluator.

Verdict: **negative** , enlarging + retraining the value head is a large OFFLINE metric win (halved MSE) that
consistently DEGRADES real play on both models and both opponents, and does not rescue alpha-beta. Mechanism:
hard-fitting Stockfish evals recalibrates the value to the narrow depth-14-cache distribution, giving worse
backup signal during MCTS on real-play positions (D35 confirmed and sharpened). The value head is now a CLOSED
lever for strength: value-target ACCURACY is not the bottleneck; the bottleneck is that offline value quality
does not equal play value. This is the 7th non-scale attempt (D35, D40, D45, D48, D49, D50, D52; D51 competition
data was neutral). SCALE (params/data, D43) remains the only lever with a positive slope. Artifacts:
scripts/train_value_head_big.py, kibitzer/model.py (build_value_head + config fields), runs/value_big/
value_big_{base,comp}.pt, reports/scaling_law_2/ (value_big_sf1900*, leela_*), reports/value_head/ figures.
Next step: stop tuning the fixed-scale model; the blog's ceiling story is complete , pursue scale (S3+ params
and/or more same-distribution data) as the only remaining lever, or write up the ceiling result.

---

## D53 , on-policy reverse-KL distillation from lc0 regressed (negative; botched config)

Best-motivated distill attempt: lc0 (leela t1-256x10) is a demonstrably stronger teacher (its nodes=1 raw policy
beat our 64-sim PUCT 0.78, D-Leela), so distilled its search visit-distribution (400 nodes ~2900) into the comp
base with the modern on-policy + reverse-KL recipe (Thinking Machines OPD). Built scripts/distill_lc0_opd.py
(gen: student self-play, lc0 verbose-move-stats visit dist per position via the working CUDA lc0 backend; train:
reverse KL(student||teacher) over legal moves, warm-start S2_shaw_142M_comp.pt). Gen 17,044 on-policy positions,
reverse_kl 0.71->0.24 (clean). Head-to-head vs the base it started from: **0W/1D/8L, score 0.056** over 9 games
(stopped early) => hard REGRESSION.

Diagnosis: the CONFIG was botched, not (only) the objective. (1) gen temperature 1.0 made the student play
near-random moves, so the on-policy positions were junk states never seen in real PUCT play , we distilled lc0's
replies to nonsense. (2) trained ALL params at lr 2e-4 for 4 epochs; reverse-KL is zero-forcing on lc0's sharp
visit dist, so it drove the whole model toward a near-delta and nuked the base's calibrated policy (converged
loss != good play, cf D49). (3) no anchor to the base policy => catastrophic forgetting (violates the D3 raw-
policy-preservation rule).

Verdict: **negative** (this run). Note reverse-KL is not the culprit , forward-KL policy distillation ALSO failed
twice (ChessBot policy distill converged sub-SFT; D49 AZ soft-CE regressed), so KL direction is not the lever.
That is now THREE flavors of engine-policy distillation that fail to lift our 15M net , consistent with the
finding that our net's strength is not reachable by policy imitation (real-strength is search/value-based and our
capacity caps it). A properly-tuned OPD retry (PUCT/low-temp gen positions, freeze trunk + low LR, base-policy KL
anchor) is the only version not yet tried and could be run once, but the prior is weak. 8th non-scale attempt
(D35,D40,D45,D48,D49,D50,D52,D53; D51 neutral). SCALE (D43: bigger params / more data) remains the ONLY lever
with a positive slope. Decision: stop non-scale tinkering; commit to a scale run or ship the ceiling write-up.
Artifacts: scripts/distill_lc0_opd.py, runs/opd/, reports/scaling_law_2/opd_*.

---

## D54 , on-policy distillation from lc0: closed lever in every configuration (negative)

Followed D53's botched OPD with a properly-configured retry and a 3-round iteration. All warm-started from the
142M comp base; teacher = lc0 t1-256x10 at 400-node visit-distribution (~2900); measured vs the external Leela
opponent at nodes=1 (~2700, base scores 0.225) and head-to-head vs the comp base. Scripts: distill_lc0_opd.py
(gen temp 0.3 near-real positions, train reverse-KL to teacher + KL-anchor to base to prevent forgetting, freeze
trunk / policy-head-only / lr 5e-5).

Full arc, win-rate vs Leela-2700 (n=20 each):
- base 0.225 | v1 (aggressive, all-params lr2e-4, D53) 0.056 | v2 (gentle, 1 round) 0.100 |
  iter R1 0.200 | iter R2 0.125 | iter R3 0.150.
Head-to-head vs comp base: v2 = 0.625 (beat base); iter R3 = ~0.30 (now WEAKER than base).

Read: (1) v2's "win" vs its own base (0.625) was STYLE EXPLOITATION, not strength , the same model scored only
0.100 vs the external 2700, worse than base's 0.225. (2) Iterating did NOT climb toward the teacher; vs-2700
stayed pinned well below base (0.20->0.125->0.150) and the iterated model became weaker than base head-to-head too
(0.30). Mechanism: reverse-KL pulls the policy toward lc0's sharp 400-node distribution, which collapses the
search-usable policy diversity our PUCT needs , fitting the teacher harder REDUCES real strength.

Verdict: **negative** , engine-policy distillation is a CLOSED lever in every flavor tried (aggressive regress,
gentle safe-but-not-stronger, iterative corrosive). Confirms: our 15M net's strength is not reachable by imitating
a stronger engine's policy (its edge is search/value-based; our capacity caps policy imitation). 8th non-scale
attempt (D35,D40,D45,D48,D49,D50,D52,D53/54; D51 neutral). Reconfirms the ~2500-2600 ceiling (also: our base
loses 0.775 to lc0 nodes=1). CRITICAL EVAL LESSON: beating one's own base head-to-head does NOT mean stronger ,
always measure vs an EXTERNAL fixed opponent. Next: proper multi-iteration AlphaZero self-play from the comp base
with outcome-trained value (the one untried mechanism) , running. Artifacts: scripts/distill_lc0_opd.py,
runs/opd/, reports/scaling_law_2/opd_*, reports/opd/ figures.
