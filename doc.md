# Kibitzer

## Main motive

Kibitzer is an attempt to build a learned chess engine that can eventually beat
the strongest version of Stockfish.

That is the north-star goal of the project. The target is not merely to imitate
human games, reach a respectable online rating, or beat a deliberately weakened
engine. The long-term objective is to produce a policy/value system whose best
playing configuration can defeat full-strength Stockfish under a fixed,
reproducible match protocol.

The project is not currently close to that result. The strongest previous
Kibitzer checkpoint played around the lower Stockfish ladder, and the training
diagnosis in `TRAINING_DIAGNOSIS.md` documents why later experiments failed to
improve it. Beating full-strength Stockfish is therefore an ambition and research
direction, not a claim about the present model.

The immediate purpose of the clean rebuild is to establish a technically sound
base that can keep improving without repeating the old failure modes:

- learn from the current chess position rather than game-history shortcuts;
- produce a strong legal-move policy;
- learn a calibrated value estimate that search can trust;
- absorb richer action-value information from Stockfish;
- preserve strength as training stages are added;
- make every promotion decision through reproducible, statistically meaningful
  matches.

## What the model is supposed to learn

Kibitzer has two outputs for every position:

1. **Policy:** which legal move should be played.
2. **Value:** how favorable the position is for the side to move.

The policy provides a move prior. The value head provides a position evaluation.
Together they are intended to support both direct play and future search. A good
policy narrows the useful branches; a good value head lets search compare the
positions reached by those branches.

This distinction matters. A model can imitate plausible moves while having no
real understanding of whether a position is winning or losing. Previous
Kibitzer checkpoints exhibited exactly that failure, so policy quality and value
quality must be measured independently.

## Current architecture

The clean-rebuild model has **32,129,153 trainable parameters**. It combines a
square-level transformer position encoder with a lightweight transformer/SSM
hybrid trunk and separate policy and value heads.

```text
current chess position
        │
        ├── 64 square piece tokens
        └── 7 auxiliary state features
                │
                ▼
       square-level position encoder
       3 bidirectional transformer blocks
                │
                ▼
          one position vector
             d_model=320
                │
                ▼
      10-block transformer/SSM trunk
      4 causal-attention + 6 SSM blocks
                │
        ┌───────┴────────┐
        ▼                ▼
  policy head        value head
  4,672 logits       scalar in [-1, 1]
```

### Parameter breakdown

| Component | Parameters | Purpose |
|---|---:|---|
| Position encoder | 4,944,320 | Represent pieces, squares, and board state |
| Hybrid trunk | 25,633,280 | Transform the position representation |
| Policy head | 1,499,712 | Score the fixed move action space |
| Value head | 51,521 | Estimate the side-to-move position value |
| **Total** | **32,129,153** | |

## Input representation

### Board tokens

The board is represented as 64 tokens in `a1` through `h8` order. Each token is
one of 13 values:

- empty square;
- six white piece types;
- six black piece types.

Each square receives both a learned piece embedding and a learned square
embedding. This preserves piece identity and absolute board location.

### Auxiliary state

Seven scalar features are included:

1. side to move;
2. white kingside castling right;
3. white queenside castling right;
4. black kingside castling right;
5. black queenside castling right;
6. en-passant file, or `-1` when absent;
7. normalized halfmove clock.

These features are projected to the model dimension and added to every square
token.

The current FEN-level representation is enough for ordinary move selection.
Exact threefold-repetition state is not fully represented; that can later be
added explicitly without returning to a long, learned game-history dependency.

## Position encoder

The position encoder operates across the 64 board squares. Its default settings
are:

- model width: `320`;
- attention heads: `8`;
- encoder blocks: `3`;
- feed-forward expansion: `4x`;
- normalization: RMSNorm;
- feed-forward activation: SwiGLU;
- dropout: none.

Each encoder block performs bidirectional self-attention over all squares,
followed by a SwiGLU feed-forward network. After the final RMSNorm, mean pooling
compresses the 64 square representations into one 320-dimensional position
vector.

Bidirectional square attention is intentional: within one chess position, every
piece should be able to interact with every other piece. Pins, attacks, defenses,
king safety, and pawn structure are global board relationships.

## Hybrid trunk

The pooled position vector passes through ten residual blocks:

- blocks 0, 3, 6, and 9 are causal attention blocks;
- the other six blocks are lightweight selective SSM blocks.

The attention blocks use RMSNorm, multi-head attention, and SwiGLU feed-forward
layers.

Each SSM block contains:

- input-dependent gating;
- depthwise local convolution;
- learned recurrent-state decay;
- input and output state projections;
- a learned skip connection;
- a residual SwiGLU feed-forward layer.

The implementation is Mamba-like, but it is a small native PyTorch block rather
than the official Mamba kernel.

The architecture supports sequences of up to 128 positions. The current phase-1
and phase-2 data pipeline deliberately supplies **one position at a time**. At
sequence length one, the trunk behaves as a deep position-processing network and
cannot learn shortcuts from the preceding game history. Sequence support remains
available for future controlled experiments, but it is not the current training
contract.

## Policy head

The policy head is a linear projection from the 320-dimensional trunk output to
4,672 move logits.

The action space follows the AlphaZero-style `64 × 73` layout:

- 56 queen-like move planes: eight directions at distances one through seven;
- eight knight-move planes;
- nine underpromotion planes.

Queen promotions use the corresponding queen-like plane. Before normalization,
illegal actions are masked so that probability mass is restricted to legal
moves.

## Value head

The value head is:

```text
320 → 160 → GELU → 1 → tanh
```

Its output lies in `[-1, 1]` and is interpreted from the side-to-move
perspective:

- `+1`: strongly favorable or winning;
- `0`: approximately balanced;
- `-1`: strongly unfavorable or losing.

The value head is small relative to the trunk. That is acceptable only if its
targets are informative. The previous model's value head remained near zero and
failed basic material-ordering tests, so value calibration is a first-class
requirement rather than a secondary metric.

## Current training phases

### Phase 1: human move cloning

`scripts/train_bc.py` trains from PGN positions.

For every position:

- the policy target is the move played in the game;
- the value target is the final game result from the side-to-move perspective.

The current loss is:

```text
policy cross-entropy + 0.25 × value MSE
```

The purpose of phase 1 is to build a stable legal-move prior from real positions.
It is not expected to beat full-strength Stockfish by itself.

### Phase 2: Stockfish action-value distillation

`scripts/distill_stockfish.py` asks Stockfish for up to eight candidate moves and
their evaluations.

For every labeled position:

- Stockfish scores are converted to bounded side-to-move values;
- a dense policy target is formed from the candidate action-values;
- the best candidate's value becomes the scalar value target;
- the model trains with dense-policy cross-entropy and value MSE.

The default policy temperature is `0.02`. Target sharpness must be measured
before scaling the run. Previous action-value training failed because a higher
temperature produced nearly uniform targets.

## Path from the current model to the main goal

The following stages describe the intended research path. Only the first two are
implemented on the clean branch today.

### Stage 0: make training trustworthy

- held-out validation positions;
- periodic and resumable checkpoints;
- FP32 master parameters and optimizer state with mixed-precision compute;
- policy entropy, calibration, gradient norm, update ratio, and SNR diagnostics;
- independent value calibration and material-ordering tests;
- paired engine gates with confidence intervals.

### Stage 1: strong position-only policy

Train on a large, filtered set of real high-level games. The model must generalize
off-book and preserve strength under randomized paired openings.

### Stage 2: calibrated Stockfish action-values

Store raw per-move engine values, tune target temperature during training, and
verify that the value head correctly orders material and tactical positions.

### Stage 3: search

Add reproducible policy/value-guided search. Raw policy and search-assisted
strength must be reported separately. Hand-written material blending must never
be mistaken for learned-value quality.

### Stage 4: search-coupled training

Use search targets, replay, and a frozen-reference trust region without allowing
the raw policy to regress. Every candidate checkpoint must pass the previous
checkpoint under identical conditions before becoming the next training seed.

### Stage 5: scale

Beating strongest Stockfish will likely require substantially more than the
current 32M model and laptop-scale data generation. Possible requirements include:

- much larger models;
- many more engine-labeled positions;
- stronger and deeper action-value labels;
- distributed self-play and search;
- large-scale replay;
- efficient batched inference;
- extensive match evaluation.

The current model should therefore be treated as the experimental base used to
prove the learning system, not as a claim that 32M parameters are sufficient for
the final goal.

## Strength ladder

Progress should be measured as a ladder rather than jumping directly from the
current baseline to strongest Stockfish:

1. reliably beat Stockfish 1350;
2. reliably beat Stockfish 1800;
3. reliably beat Stockfish 2200;
4. reach strong master-level engine performance;
5. beat high-strength Stockfish configurations;
6. challenge and beat full-strength current Stockfish.

A checkpoint advances only when its improvement is statistically credible and
its raw policy, value calibration, and search performance do not hide regressions
in one another.

## Definition of the final goal

"Beat the strongest version of Stockfish" needs a fixed protocol before it can
be a scientific result. The final protocol should specify:

- exact Stockfish release and settings;
- hardware available to each engine;
- threads and hash size;
- opening suite;
- time control;
- pondering and tablebase rules;
- whether Kibitzer uses search;
- number of paired games;
- confidence interval and promotion threshold.

The intended result is not one lucky win. It is a statistically defensible match
victory against full-strength Stockfish under published, repeatable conditions.

## Project principle

The ambition is intentionally larger than the current system. That is useful
only if every experiment remains honest.

Lower loss is not automatically stronger chess. Higher entropy is not
automatically better exploration. Search strength is not automatically learned
value strength. A four-game result is not an Elo estimate.

Kibitzer moves toward its main goal only when the complete system becomes
stronger under a reproducible test.
