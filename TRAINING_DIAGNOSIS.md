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
