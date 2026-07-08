# Kibitzer Training Diagnosis

## Scope

This is a read-only diagnosis of the training history recorded in
`DECISIONS.md`, the surviving public checkpoints, the old `main` training code,
and the current `clean-rebuild` branch.

Ten representative checkpoints were downloaded from the public Kibitzer
Hugging Face collection and inspected with the matching old model code:

- SFT steps 2,000, 6,000, 10,000, 14,000, and 50,000
- canonical `kibitzer-sft`
- AZ iterations 1, 5, and 20
- Stockfish-distillation steps 500 and 1,000

Behavioral measurements used 85 positions from realistic opening lines. These
are useful controlled diagnostics, not a replacement for a large held-out
position set or paired engine matches.

## Executive conclusion

The project does not have one isolated entropy or learning-rate problem. The
observed failures come from several interacting issues:

1. The old value head learned almost no useful chess evaluation.
2. The old run optimized parameters and Adam moments directly in BF16, freezing
   all normalization scales.
3. Training and inference changed the history/context contract between runs.
4. Batched inference in the old model mishandled padded histories.
5. Small match samples were repeatedly treated as reliable Elo changes.
6. Some distillation targets were weak or nearly uniform.

The current `clean-rebuild` avoids several of these old failures, but it has not
been trained yet and its trainer is still too minimal for a decision-grade run.

## Checkpoint comparison

The SNR column is a proxy derived from bias-corrected Adam first and second
moments. It measures recent gradient-direction agreement; it is not a formal
per-microbatch gradient-noise-scale measurement.

| Checkpoint | Relative parameter drift from SFT 14k | Effective policy moves | Value result | Adam SNR proxy |
|---|---:|---:|---|---:|
| SFT 14k | baseline | 2.16 | effectively dead | 0.22 |
| SFT 50k | 23.9% | 2.20 | still dead | 0.22 |
| AZ iteration 1 | 0.075% | 2.16 | unchanged | 1.75 |
| AZ iteration 5, context 1 | 5.13% | 2.42 | badly corrupted | 0.056 |
| AZ iteration 20 | 2.48% | 2.18 | weak, slightly changed | 0.21 |
| Distillation step 500 | 2.44% | 2.47 | wrong material response | 0.23 |
| Distillation step 1,000 | 3.72% | 2.61 | still wrong | 0.23 |

## How to interpret the calculated metrics

No single scalar establishes chess strength. Each metric below answers a
different question, and its value is useful only when compared on the same data,
model contract, and evaluation settings.

### M1 , Policy and value training loss

#### Why this metric matters

The loss is the signal directly optimized by gradient descent. Policy
cross-entropy measures how much probability the model assigns to the target
move or target distribution. Value MSE measures the distance between the value
prediction and its target.

#### Mathematical equation

For a one-hot move target (a^*), legal-move policy cross-entropy is:

$$
L_{\text{policy}} = -\log p_\theta(a^* \mid s, a \in A_{\text{legal}}).
$$

For a dense target distribution (q(a \mid s)):

$$
L_{\text{policy}} = -\sum_{a \in A_{\text{legal}}}
q(a \mid s)\log p_\theta(a \mid s).
$$

The scalar value loss is:

$$
L_{\text{value}} = \frac{1}{N}\sum_{i=1}^{N}
\left(v_\theta(s_i)-z_i\right)^2.
$$

The current clean-rebuild objective is:

$$
L = L_{\text{policy}} + 0.25L_{\text{value}}.
$$

#### How to comprehend the output

- Lower training loss means the model fits the supplied targets better.
- Lower held-out loss means that fit transfers to unseen positions from the
  same distribution.
- Falling training loss with flat or rising held-out loss indicates overfitting.
- Losses from different objectives, temperatures, or target distributions are
  not directly comparable.
- A low dense-policy loss can still produce weak chess if the teacher
  distribution itself is weak or nearly uniform.

#### What the Kibitzer values mean

The failed ChessBot-policy run reduced policy loss from about `3.35` to `1.77`
but remained much weaker than SFT. The action-value run converged near `2.8`
while playing almost randomly. Optimization worked in both cases; target quality
and transfer did not. Training loss alone was therefore not a valid promotion
metric.

### M2 , Relative parameter drift

#### Why this metric matters

Parameter drift measures how far a checkpoint moved from a trusted reference.
It helps distinguish a checkpoint that is functionally almost unchanged from
one that has substantially rewritten its representation or output heads.

#### Mathematical equation

For checkpoint parameters \(\theta\) and reference parameters
\(\theta_{\text{ref}}\):

$$
D_{\text{rel}} =
\frac{\lVert\theta-\theta_{\text{ref}}\rVert_2}
{\lVert\theta_{\text{ref}}\rVert_2}.
$$

The same equation is applied to a parameter group, such as the value head, to
obtain module-specific drift.

#### How to comprehend the output

- `0` means the parameters are identical.
- A very small value means claims of catastrophic forgetting require strong
  behavioral evidence; the model barely moved.
- Large drift means substantial rewriting, but does not say whether the change
  is helpful.
- Disproportionately large head drift is a warning that one output was rewritten
  faster than the shared representation.
- Drift must be paired with output divergence because neural networks can move
  in parameter space while retaining similar behavior.

#### What the Kibitzer values mean

- AZ iteration 1 drifted only `0.075%`; its `0/4` result cannot reasonably be
  attributed to wholesale forgetting.
- AZ iteration 5 drifted `5.13%` overall but `26.9%` in the value head. That
  imbalance matches the observed value collapse.
- SFT 50k drifted `23.9%` from SFT 14k while retaining `95.3%` top-move
  agreement. Parameter drift alone would have overstated the behavioral change.

### M3 , Policy entropy, normalized entropy, and effective move count

#### Why this metric matters

Entropy measures how spread out the legal-move policy is. It detects policies
that are nearly uniform, excessively indecisive, or implausibly concentrated on
one move. Effective move count translates entropy into a more intuitive number
of equally likely choices.

#### Mathematical equation

For legal-move probabilities \(p(a \mid s)\):

$$
H(p) = -\sum_{a \in A_{\text{legal}}}p(a \mid s)\log p(a \mid s).
$$

The effective move count is:

$$
N_{\text{eff}} = \exp(H(p)).
$$

To compare positions with different numbers of legal moves:

$$
H_{\text{norm}} = \frac{H(p)}{\log|A_{\text{legal}}|}.
$$

#### How to comprehend the output

- `H = 0` and `N_eff = 1` describe a deterministic policy.
- `H_norm` near `1` describes an approximately uniform legal-move policy.
- Increasing entropy can mean useful uncertainty or loss of a decisive signal.
- Decreasing entropy can mean improved confidence or brittle overconfidence.
- The desired value depends on the position; tactical positions should often be
  sharper than quiet positions.
- Entropy should be compared with move accuracy, calibration, and playing
  strength. It is not a standalone quality score.

#### What the Kibitzer values mean

- SFT 14k: `H=0.77`, `N_eff=2.16`. The opening policy was sharp, not uniform.
- Distillation step 1,000: `H=0.96`, `N_eff=2.61`. Distillation made the policy
  less decisive without repairing value understanding.
- Failed action-value run: `H≈2.79`, `N_eff≈16`. This matched its nearly uniform
  training targets and explained the random-looking move selection.

### M4 , Top-1 probability, margin, target accuracy, and negative log-likelihood

#### Why this metric matters

These metrics separate confidence from correctness. A model may be highly
confident in the wrong move, or assign moderate probability to the correct move
without ranking it first.

#### Mathematical equation

Let \(p_{(1)}\) and \(p_{(2)}\) be the highest and second-highest legal-move
probabilities:

$$
\text{margin} = p_{(1)} - p_{(2)}.
$$

For target move \(a_i^*\), top-1 accuracy is:

$$
\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\arg\max_a p_i(a)=a_i^*].
$$

Negative log-likelihood is:

$$
\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log p_i(a_i^*).
$$

#### How to comprehend the output

- High top-1 probability plus high accuracy indicates useful confidence.
- High top-1 probability plus low accuracy indicates overconfidence.
- A small margin means the top two moves are difficult for the model to
  distinguish.
- Lower NLL is better for a fixed target set because it rewards assigning more
  probability to the target even when it is not ranked first.
- Human-move accuracy is not engine accuracy: several legal moves may be equally
  strong, and the human target may not be best.

#### What the Kibitzer values mean

On the controlled opening-continuation set:

| Checkpoint | Mean top-1 probability | Mean margin | Target top-1 accuracy | NLL |
|---|---:|---:|---:|---:|
| SFT 14k | 0.736 | 0.591 | 84.7% | 0.489 |
| SFT 50k | 0.724 | 0.571 | 88.2% | 0.476 |
| Distillation 500 | 0.689 | 0.523 | 83.5% | 0.544 |
| Distillation 1,000 | 0.670 | 0.501 | 81.2% | 0.568 |

SFT 50k was not behaviorally worse on this narrow set despite its lower
20-game Elo label. Distillation reduced confidence and worsened target fit.
Because this suite contains known openings, the absolute accuracies are not a
generalization estimate; their value is in same-suite checkpoint comparison.

### M5 , Jensen-Shannon divergence and top-move agreement

#### Why this metric matters

Jensen-Shannon divergence measures how much the complete policy distribution
changed between two models or two input representations. Top-move agreement
measures whether the final greedy decision changed. Together they distinguish a
small probability reshuffle from a move-changing behavioral shift.

#### Mathematical equation

For policies \(P\) and \(Q\), define \(M=(P+Q)/2\). Then:

$$
\operatorname{JSD}(P,Q) =
\frac{1}{2}\operatorname{KL}(P\|M) +
\frac{1}{2}\operatorname{KL}(Q\|M).
$$

Top-move agreement is:

$$
A_{\text{top1}} = \frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\arg\max P_i=\arg\max Q_i].
$$

#### How to comprehend the output

- `JSD = 0` means identical distributions.
- Larger JSD means stronger distributional change; interpret it relative to
  known-good and known-bad checkpoint pairs rather than using a universal
  cutoff.
- High top-move agreement with nonzero JSD means confidence changed but the
  greedy choice usually did not.
- Low agreement means the change affects actual move selection.

#### What the Kibitzer values mean

- AZ iteration 1 versus SFT: `JSD=0.0000045`, agreement `100%`. It was
  functionally unchanged.
- SFT 50k versus SFT 14k: `JSD=0.0064`, agreement `95.3%`. The Elo labels made
  the difference look much larger than the policy behavior did.
- Full history versus standalone position for SFT 14k: `JSD=0.360`, agreement
  `35.3%`. History removal fundamentally changed the model's decisions.

### M6 , Value spread, material sensitivity, and ordering

#### Why this metric matters

A value head can achieve a superficially acceptable MSE by predicting values
near zero everywhere. Value spread detects this collapse. Material perturbation
tests ask whether the model responds in the correct direction to an obvious
change in position strength.

#### Mathematical equation

For predictions \(v_i\):

$$
\mu_v = \frac{1}{N}\sum_i v_i,
\qquad
\sigma_v = \sqrt{\frac{1}{N}\sum_i(v_i-\mu_v)^2},
$$

$$
\operatorname{mean}|v| = \frac{1}{N}\sum_i |v_i|.
$$

For a position with both queens, define:

$$
\Delta_Q =
v(s\text{ with opponent queen removed}) -
v(s\text{ with own queen removed}).
$$

The strict ordering indicator is:

$$
\mathbf{1}[v_{\text{favorable}} > v_{\text{original}} >
v_{\text{unfavorable}}],
$$

averaged over tested positions to obtain the ordering rate.

#### How to comprehend the output

- Very small \(\sigma_v\) and mean absolute value indicate a near-constant
  value head.
- Positive \(\Delta_Q\) is required: winning the opponent queen must be valued
  above losing one's own queen.
- A negative queen swing is a direct sign error.
- Ordering rate near `100%` is expected for such a basic perturbation; values
  near chance or below show that search cannot safely trust the head.
- This is a sanity test, not complete calibration. Engine correlation and
  tactical suites are still required.

#### What the Kibitzer values mean

SFT 14k had `σ=0.055`, `mean|v|=0.050`, `ΔQ=-0.017`, and only `16.7%`
correct ordering. The head was nearly constant and moved in the wrong direction
under an obvious material change. That is why the diagnosis calls it
functionally dead.

AZ iteration 5 shifted the mean opening value to `-0.241` and changed the value
head by `26.9%`. This was not improved calibration; it was a large biased rewrite
of an already weak head.

### M7 , Adam moment SNR proxy

#### Why this metric matters

Adam stores a moving average of gradients and squared gradients. Their ratio can
be used as a checkpoint-available proxy for how consistently recent gradients
pointed in one direction. Low agreement means optimizer steps are dominated by
changing or noisy gradients.

#### Mathematical equation

Bias-correct the Adam moments:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}.
$$

The aggregate coherence ratio used here is:

$$
r = \frac{\sum_i \hat m_{t,i}^2}{\sum_i \hat v_{t,i}}.
$$

The reported proxy is:

$$
\operatorname{SNR}_{\text{Adam}} =
\sqrt{\frac{r}{1-r}}.
$$

#### How to comprehend the output

- Values near `0` indicate weak recent gradient-direction agreement.
- Larger values indicate more coherent recent updates.
- High SNR is not automatically good: repeatedly training on a tiny replay
  buffer can produce consistent but overfit gradients.
- Compare the total and per-module values. A noisy value head with a coherent
  policy head identifies a localized problem.
- This is not the formal gradient noise scale because individual microbatch
  gradients were unavailable.

#### What the Kibitzer values mean

- SFT checkpoints stayed around `0.19–0.22`: learnable signal existed, but it
  was weak relative to gradient variation.
- AZ iteration 5 fell to `0.056`: its updates were extremely inconsistent while
  the value head was being heavily rewritten.
- AZ iteration 1 reached `1.75`, but it repeatedly optimized only 50 replay
  samples. The high value indicates repetition and alignment, not demonstrated
  chess improvement.

### M8 , Consecutive update cosine similarity

#### Why this metric matters

This metric asks whether training moves in a stable long-term direction across
checkpoint intervals. It complements the short-horizon Adam proxy.

#### Mathematical equation

For parameter changes \(\Delta_a=\theta_{t_2}-\theta_{t_1}\) and
\(\Delta_b=\theta_{t_3}-\theta_{t_2}\):

$$
\cos(\Delta_a,\Delta_b) =
\frac{\Delta_a \cdot \Delta_b}
{\lVert\Delta_a\rVert_2\lVert\Delta_b\rVert_2}.
$$

#### How to comprehend the output

- `+1` means both intervals moved in the same direction.
- `0` means the long-range updates were orthogonal.
- `-1` means the later interval reversed the earlier update.
- Near-zero cosine can arise from noisy data, convergence into a broad basin,
  objective conflict, or optimizer quantization. It is a warning, not a proof
  of random gradients.

#### What the Kibitzer values mean

- SFT 2k–6k versus 6k–10k: `0.126`
- SFT 6k–10k versus 10k–14k: `0.052`

The SFT trajectory had little persistent long-range direction even though its
training loss continued to fall. This is consistent with fitting noisy targets
without reliable strength improvement.

### M9 , Exact unchanged fraction and BF16 quantization

#### Why this metric matters

When parameters are stored and optimized in low precision, valid small updates
may round back to the original bit pattern. Counting exactly unchanged elements
can expose parameters that are effectively frozen.

#### Mathematical equation

Between checkpoints \(t_1\) and \(t_2\):

$$
U = \frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\theta_{t_1,i}=\theta_{t_2,i}].
$$

#### How to comprehend the output

- A high value can be legitimate for unused action rows or explicitly frozen
  parameters.
- A value of `100%` for trainable normalization scales across thousands of
  updates is suspicious.
- Interpret by module and parameter role; the global fraction alone can be
  misleading.

#### What the Kibitzer values mean

All 32 trainable RMSNorm scale tensors were exactly unchanged from SFT step
2,000 to step 14,000. Combined with BF16 parameters and BF16 Adam moments, this
is strong evidence that low-precision rounding froze those scales.

### M10 , Match score, Elo estimate, confidence interval, and significance

#### Why this metric matters

Engine matches are the final behavioral test, but a match score is a random
sample. Confidence intervals quantify uncertainty, while a significance test
asks whether two observed checkpoint scores are distinguishable at the current
sample size.

#### Mathematical equation

For wins \(W\), draws \(D\), losses \(L\), and
\(N=W+D+L\):

$$
S = \frac{W+0.5D}{N}.
$$

Against an opponent with a fixed rating, the conventional logistic Elo
difference estimate is:

$$
\Delta\operatorname{Elo} = 400\log_{10}\left(\frac{S}{1-S}\right).
$$

For decisive Bernoulli outcomes, the Wilson score interval center and half-width
are:

$$
c = \frac{\hat p + z^2/(2N)}{1+z^2/N},
$$

$$
h = \frac{z}{1+z^2/N}
\sqrt{\frac{\hat p(1-\hat p)}{N}+\frac{z^2}{4N^2}},
$$

giving interval \([c-h,c+h]\). Here \(z=1.96\) for an approximate 95% interval.
The reported Fisher exact-test p-value compares two small win/loss tables without
using a large-sample approximation.

#### How to comprehend the output

- A score above `50%` suggests the model outscored the opponent in that sample.
- If the confidence interval crosses `50%`, superiority is not established.
- A p-value such as `0.27` means the observed checkpoint difference is plausible
  under equal underlying strength; it is not evidence of a real regression.
- Infinite Elo from `0/N` or `N/N` is a mathematical boundary artifact, not a
  precise strength estimate.
- Paired openings reduce variance but do not remove the need for enough games.

#### What the Kibitzer values mean

SFT 14k scored `7/20` with a 95% Wilson interval of `18%–57%`. SFT 50k scored
`3/20` with interval `5%–36%`. The intervals overlap, and Fisher's exact test
gave `p≈0.27`. The data did not establish that SFT 14k was truly stronger than
SFT 50k.

### M11 , Input-contract and batch-invariance metrics

#### Why this metric matters

A model should produce the same output for the same input regardless of which
unrelated samples share its batch. Separately, comparing full-history and
position-only inputs quantifies dependence on the historical input contract.

#### Mathematical equation

For the same position evaluated in modes \(x\) and \(y\), report:

$$
\operatorname{JSD}(p_x,p_y),
\qquad
\mathbf{1}[\arg\max p_x=\arg\max p_y],
\qquad
|v_x-v_y|.
$$

Aggregate these over the suite using mean JSD, top-move agreement rate, mean
absolute value difference, and maximum absolute value difference.

#### How to comprehend the output

- For a pure batching comparison, all differences should be approximately zero.
- Nonzero batch differences identify a masking, padding, state, or numerical
  consistency bug.
- For full-history versus position-only comparison, large differences indicate
  input-distribution dependence rather than a batching bug.
- Low top-move agreement means the difference is large enough to change actual
  play, not just confidence.

#### What the Kibitzer values mean

SFT 14k full-history versus position-only evaluation produced `JSD=0.360` and
only `35.3%` top-move agreement. The old model depended strongly on history.

When only the batch composition changed, the old padding bug changed SFT 14k's
top move in `4.7%` of positions and AZ iteration 5's top move in `12.9%`.
Maximum value differences reached `0.157` and `0.552`. A correct implementation
should not exhibit these changes.

## 1. The value head is the clearest functional failure

For the SFT 14k checkpoint:

- prediction standard deviation: `0.055`
- mean absolute prediction: `0.050`
- value swing between removing the opponent queen and removing the player's
  queen: `-0.017`
- correctly ordered queen-advantage triplets: `16.7%`

The queen-swing sign is wrong on average. In this controlled probe, the model
frequently considered losing its own queen better than winning the opponent's
queen.

This makes learned-value MCTS unreliable. The search improvement reported in
Decision D14 was likely driven primarily by `material_weight=0.85`, not by a
strong learned value function.

The value target contributes to the problem. Every position receives the final
game result. For early and middle-game positions, that label says which player
eventually won, not whether the current position is objectively good. Player
errors and the remainder of the game create substantial target noise.

## 2. The old BF16 optimizer setup impaired learning

The old trainer converted the full model to BF16 before constructing AdamW.
Checkpoint inspection confirmed:

- all 116 parameter tensors were BF16
- all 232 Adam moment tensors were BF16
- all 32 RMSNorm scale tensors were bit-for-bit unchanged from step 2,000 to
  step 14,000

Small updates around an RMSNorm scale of `1.0` cannot reliably survive BF16
quantization. This is not ordinary mixed-precision training: there were no FP32
master parameters or FP32 Adam moments.

The current `clean-rebuild` trainer keeps model parameters in FP32 by default,
so this specific defect is not present there.

## 3. Not every bad match result was catastrophic forgetting

The decision log correctly identified some genuine collapse, but the term was
applied too broadly.

### AZ iteration 1: no demonstrated collapse

AZ iteration 1 had:

- relative parameter drift: `0.075%`
- JS divergence from SFT: `0.0000045`
- top-move agreement with SFT: `100%`

The checkpoint was functionally unchanged. Its `0/4` match did not demonstrate
collapse; four games were insufficient evidence.

Its high SNR proxy of `1.75` is also not proof of useful learning. The run made
20 updates over only 50 replay samples, so repeatedly seeing a tiny dataset can
produce highly aligned gradients.

### AZ iteration 5 with context 1: genuine collapse

This checkpoint showed a real failure:

- overall parameter drift: `5.13%`
- value-head drift: `26.9%`
- Adam SNR proxy: `0.056`
- mean value on the opening suite: `-0.241`
- mean full-history versus position-only value difference: `0.679`

It trained a history-dependent SFT model with `context_window=1`, weak reference
KL of `0.05`, and only 1,693 replay positions. The run rewrote the value head
under a different input contract while its gradients had very low directional
agreement.

### AZ iteration 20: inconclusive rather than healthy

AZ iteration 20 stayed closer to SFT:

- overall drift: `2.48%`
- value-head drift: `10.2%`
- SNR proxy: `0.21`
- top-move agreement with SFT: `96.5%`

Its `4/12` evaluation is too small to establish improvement or degradation.

## 4. Elo checkpoint selection was underpowered

SFT 14k scored `7/20` against Stockfish 1320. SFT 50k scored `3/20`.

Their Wilson 95% score intervals were:

- SFT 14k: `18%–57%`
- SFT 50k: `5%–36%`

A two-sided Fisher exact test gives approximately `p=0.27`. The observed
difference is not statistically decisive.

The models were also behaviorally similar:

- JS divergence: `0.0064`
- top-move agreement: `95.3%`
- SFT 50k had slightly better accuracy on the controlled opening continuation
  set

The apparent 1212-to-1019 Elo regression was therefore probably dominated by
match variance. Selecting the best result after repeatedly evaluating 20-game
samples also creates winner's curse: the highest reported checkpoint is likely
partly the luckiest checkpoint.

The later move to paired 40-game gates was materially better, though larger
samples are still required for small expected differences.

## 5. History dependence is large and measurable

For SFT 14k, evaluating the same position with its full history and as a
standalone FEN produced:

- mean JS divergence: `0.360`
- identical top move: only `35.3%` of positions
- mean absolute value difference: `0.053`

The old model did not use history as a small auxiliary feature. Removing history
substantially changed its move selection. This supports Decision D24's move to a
position-only model.

It also explains why training on generated histories, testing with different
context windows, or switching abruptly to context 1 caused severe distribution
shift.

## 6. Old batched inference had a padding bug

The old inference implementation left-padded shorter histories to the longest
history in a batch. Its model then discarded `pad_mask`, allowing real positions
to attend to synthetic empty-board prefixes.

Across the controlled suite, batched padding changed:

- SFT 14k's top move in `4.7%` of positions
- AZ context-1 iteration 5's top move in `12.9%` of positions
- SFT values by as much as `0.157`
- AZ context-1 values by as much as `0.552`

This means some batched search or evaluation outputs depended on which other
histories happened to share the batch.

The current clean model passes its padding mask into its attention and SSM
blocks, so the old implementation bug does not directly carry over.

## 7. Entropy was a symptom, not the root cause

There is no entropy bonus in the SFT, Stockfish-distillation, AZ, or current
clean-rebuild objectives. Entropy was logged as a diagnostic in the AZ runs but
was not an optimized term.

The measured behavior was:

- SFT 14k entropy: `0.77` nats, equivalent to `2.16` effective moves
- Distillation step 500: `0.91` nats, or `2.47` effective moves
- Distillation step 1,000: `0.96` nats, or `2.61` effective moves
- Failed action-value run: approximately `2.79` nats, or about `16` effective
  moves

SFT was sharp but not uniformly collapsed. Stockfish distillation made it less
decisive while failing to repair value understanding.

The action-value run's high entropy came from target construction. At
`av_temp=0.1`, tightly clustered ChessBot action-values became a nearly uniform
target, with only `0.14–0.21` probability on the best move. The model correctly
learned the supplied distribution; the supplied distribution was poor.

Changing an entropy coefficient would not repair that target. Raw action-values
should be stored, and target sharpness should be selected at training time using
a validated temperature sweep.

## 8. SFT update direction was weakly coherent

The SFT Adam SNR proxy stayed around `0.19–0.22`. Consecutive long-interval
parameter-update cosines were:

- steps 2k–6k versus 6k–10k: `0.126`
- steps 6k–10k versus 10k–14k: `0.052`

The optimizer continued reducing training loss, but long-range update directions
were almost orthogonal. Possible contributors include:

- noisy outcome-value labels
- changing game samples
- BF16 optimizer quantization
- a policy objective dominated by opening/style correlations
- approaching a shallow supervised optimum

This does not prove gradients were random, but it shows why lower training loss
did not translate cleanly into higher chess strength.

## 9. What the current clean rebuild fixes

The current `clean-rebuild` branch:

- creates sequence-length-one position batches
- trains from a standalone FEN rather than game history
- keeps parameters in FP32 by default
- uses a new 32.1M-parameter transformer/SSM hybrid
- uses legal-move masking for policy learning
- defaults Stockfish action-value temperature to `0.02`, substantially sharper
  than the failed `0.1` ChessBot action-value setup

These changes remove the old history contract and direct-BF16 optimizer problem.

## 10. What remains wrong or unverified in the clean rebuild

The branch currently has no trained checkpoints. Its training scripts are
scaffolds, not yet a reliable large-run system.

Current limitations include:

- all PGN samples are materialized in RAM before training
- no train/validation split
- no checkpoint resume or periodic checkpointing
- no held-out policy accuracy or cross-entropy
- no entropy, calibration, gradient norm, SNR, or policy-drift logging
- no Elo/player-quality filtering in `iter_pgn_samples`
- unfinished games receive a zero value target
- final game result is still used as every position's value label
- phase-2 Stockfish labels are generated and retained in memory
- phase 2 defaults to only 1,000 positions
- no paired engine gate is integrated into checkpoint promotion

The current loss is only:

```text
policy cross-entropy + 0.25 * value MSE
```

That is acceptable as a minimal objective, but the current metrics cannot tell
whether it is learning transferable chess rather than memorizing frequent moves.

## Recommended training-health panel

Future training should report these metrics on both training data and a fixed
held-out set.

### Policy health

- legal-move cross-entropy
- top-1, top-3, and top-5 accuracy
- entropy and effective move count
- top-1 probability and top-1/top-2 margin
- expected calibration error or confidence buckets
- KL/JS divergence from the initialization or last promoted checkpoint
- agreement with the human move, teacher argmax, and engine argmax separately

### Value health

- value MAE and MSE
- correlation with an engine evaluation
- sign accuracy
- saturation rate near `-1` or `+1`
- queen/material perturbation ordering
- tactical-position ordering
- calibration by predicted-value bucket

### Optimization health

- gradient norm before clipping
- clipping frequency
- parameter norm
- update-to-weight ratio by module
- Adam first/second-moment SNR proxy by module
- fraction of exactly unchanged parameters
- NaN/Inf counts
- policy-head and value-head drift separately

### Evaluation health

- fixed held-out position suite
- paired openings and common random seeds
- raw-policy and search-mode results reported separately
- confidence intervals for match score and Elo
- at least 40 paired games for rejection gates, with more games for small
  promotion margins
- identical time controls, context rules, and Stockfish versions

## Final diagnosis

The failure hierarchy is:

1. **Dead value supervision:** the largest functional problem, especially for
   search.
2. **Old BF16 parameter and optimizer precision:** a real training defect that
   froze all normalization scales.
3. **History and context mismatch:** the main cause of the genuine AZ collapse.
4. **Bad distillation targets:** weak policy teachers and over-softened
   action-values.
5. **Underpowered Elo gates:** the source of several false collapse and best-model
   conclusions.
6. **No held-out validation:** training loss was repeatedly mistaken for chess
   improvement.

The correct next direction remains a position-only model trained from scratch,
but it should not be launched at scale until the clean trainer has validation,
checkpointing, FP32-master mixed precision, value diagnostics, and statistically
defensible promotion gates.
