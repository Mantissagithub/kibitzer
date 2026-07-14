---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- chess
- policy-value-network
- alphazero
- mcts
- puct
- board-games
- reinforcement-learning
base_model:
- Pradheep1647/kibitzer-s2-shaw-142m-comp
---

# Kibitzer, tactical repair (`tactical_repair.pt`)

![Kibitzer 2581 Elo](https://raw.githubusercontent.com/Mantissagithub/kibitzer/main/reports/official_elo/elo_card.png)

**Kibitzer is a 15.2M-parameter, attention-first chess policy/value model trained end to end on a single laptop RTX 4060 (8 GB).** Wrapped in 512-simulation PUCT search it reaches an official **2581.6 ± 102.3 Elo** on a Stockfish-anchored tournament. This checkpoint, `tactical_repair.pt`, is the strongest trained branch in the project.

A *kibitzer* (Yiddish, from the German *kiebitzen*, "to look on at a card or chess game") is the onlooker who leans over your shoulder and tells you the best move. That is exactly what this model does: it never owns a game of its own, it reads a single position and gives its opinion.

## TL;DR

| | |
|---|---|
| parameters | 15.2M |
| architecture | Shaw relative-attention encoder + hybrid causal-attention / selective-SSM trunk |
| training data | 142M Lichess Elite positions |
| official Elo | **2581.6 ± 102.3** (512-sim PUCT, Ordo, Stockfish 2200-3100 ladder, 171 games) |
| searchless-ladder Elo | 2483 ± 32 (64-sim PUCT vs Stockfish `UCI_Elo` ladder) |
| hardware | one RTX 4060 laptop, 8 GB VRAM |
| base checkpoint | [`Pradheep1647/kibitzer-s2-shaw-142m-comp`](https://huggingface.co/Pradheep1647/kibitzer-s2-shaw-142m-comp) |
| code + full lab notebook | [github.com/Mantissagithub/kibitzer](https://github.com/Mantissagithub/kibitzer) |

## Intended use

Kibitzer is a research artifact: a study of how far a small model can be pushed at chess on consumer hardware. Use it to analyze a position and get a move (as an engine wrapped in PUCT search), to reproduce the evaluations, or as a base for further chess policy/value research. It is not a drop-in UCI engine binary, though a UCI wrapper is provided in the repo (`scripts/uci_engine.py`).

## Architecture

![Kibitzer hybrid chess architecture](https://raw.githubusercontent.com/Mantissagithub/kibitzer/main/docs/archi.png)

A position is fed as structured chess state, not pixels or a move list: each of the 64 squares becomes one of 13 piece tokens, alongside a 7-dim auxiliary vector (side-to-move, four castling rights, en passant, halfmove clock). The model can also consume up to 128 plies of history.

The board first passes through a 3-layer **position encoder** built on **Shaw-style relative attention**: every pair of squares attends through its file and rank offset, bucketed into 15x15 = 225 learned relative vectors folded into the attention bias, which makes the geometry translation-invariant (a knight-move relationship means the same thing anywhere on the board). The 64 encoded squares are mean-pooled into one dense vector per position.

Those vectors flow into a **10-layer temporal trunk over the ply history** that interleaves two block types: every third block is **causal self-attention** (global reasoning), the rest are **selective state-space blocks** (SSM, Mamba-style: causal scan, input-dependent gates, depthwise convolution, cheap linear-time state). Attention is powerful but quadratic, SSM is cheap and linear, so the trunk pays full-attention cost only where it earns its keep. An RMSNorm then feeds two heads: an AlphaZero-style **policy** over the fixed 4,672-move action space and a **tanh-bounded value** head. RMSNorm and SwiGLU are used throughout.

## Evaluation

Official tournament Elo, Ordo-rated over 171 clean games against a Stockfish `UCI_Elo` ladder, both colors, model wrapped in 512-sim PUCT, anchored SF-2500 = 2500:

| player | rating | error | games | score % |
|---|---:|---:|---:|---:|
| SF-3100 | 2979.1 | 207.1 | 32 | 91 |
| SF-2900 | 2857.7 | 164.3 | 35 | 83 |
| SF-2700 | 2660.8 | 138.5 | 36 | 61 |
| **Kibitzer @ 512 sims** | **2581.6** | **102.3** | **171** | **42** |
| SF-2500 (anchor) | 2500.0 | ---- | 35 | 39 |
| SF-2200 | 2318.0 | 158.8 | 33 | 18 |

It crushes SF-2200 and SF-2500, loses to SF-2700 and above, with the 50% crossover near SF-2600. This is the strength of the model *with* search: deep PUCT does real work that does not distill back into the raw weights.

![Elo ladder](https://raw.githubusercontent.com/Mantissagithub/kibitzer/main/reports/scaling_law/fig_elo.png)

Search is the single largest inference-time lever. On the same checkpoint versus a fixed Leela/Maia-2700 proxy, only changing simulation count:

![Search simulation sweep](https://raw.githubusercontent.com/Mantissagithub/kibitzer/main/reports/sims_sweep/fig_sims_sweep.png)

| sims | score vs 2700 proxy |
|---:|---:|
| 64 | 0.087 |
| 128 | 0.287 |
| 256 | 0.325 |
| 512 | 0.825 |

## How to use

Kibitzer is a custom architecture, so you need the `kibitzer` package from the [GitHub repo](https://github.com/Mantissagithub/kibitzer), not `transformers`.

```python
import chess
from huggingface_hub import hf_hub_download
from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search

ckpt = hf_hub_download("Pradheep1647/kibitzer-tactical-repair", "tactical_repair.pt")
evaluator = ModelEvaluator.from_checkpoint(ckpt, device="cuda")

board = chess.Board()
result = puct_search(board, evaluator, simulations=512)
print(result.move)          # best move
print(result.root_value)    # side-to-move value estimate
```

## Limitations and honest caveats

- The **2581 Elo is a with-search, compute-asymmetric match statistic**, not an intrinsic raw-weights rating. The Stockfish `UCI_Elo` ladder is an internal relative scale, not FIDE. The ±102 interval is wide (171 games across five opponents).
- The **value head is the weak link**: 33k parameters bolted onto a 15M trunk, correct in sign but noisy and late. Enlarging it improved offline metrics and *hurt* play.
- Strength plateaus around 2500-2600. Nine separate post-training directions (self-play, expert iteration, on-policy distillation, TD-Leaf, GRPO/DPPO RL, preference optimization, process-reward repair) were tried to break past it. All nine held or lost strength against a fixed external opponent. The ceiling, it turns out, is remarkably committed to being a ceiling.
- The only lever left with a positive slope is parameter and data scale, which is a deliberate not-now: the point of the project was the ceiling of a small model on a laptop.

## Provenance

- Base: `Pradheep1647/kibitzer-s2-shaw-142m-comp` (S2 Shaw, 14.9M params, 142M Lichess Elite positions).
- This checkpoint: a competition/puzzle supervised tactical repair (`tactical_supervised_repair_r1_policy_only`) on top of `policy_regret_repair.pt`, promoted as the strongest branch that survived a paired external gate.
- Full decision-by-decision history, including every failed experiment, is in the project [LOGBOOK.md](https://github.com/Mantissagithub/kibitzer/blob/main/LOGBOOK.md).

## Citation

```bibtex
@software{kibitzer2026,
  author  = {Pradheep},
  title   = {Kibitzer: a 15M-parameter chess policy/value model trained on a laptop},
  year    = {2026},
  url     = {https://github.com/Mantissagithub/kibitzer}
}
```
