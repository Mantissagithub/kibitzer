# Where Kibitzer is actually failing

A post-mortem on 336 real games. No new matches were played, no GPU was harmed. We just took
the PGNs that already exist, handed every single move the model made to Stockfish, and asked
the boring question nobody wants to ask about their own engine: how much are you throwing away,
and where.

The short version: the opening is fine, the model bleeds in the middlegame, and the endgame is
where it quietly falls apart. It does not usually lose by hanging a queen. It loses the way a
frog boils, one slightly-too-optimistic move at a time, until the position is gone and it never
noticed. GPT (consulted via codex as a second opinion) put it better than I could: "the engine
slowly believes bad positions are playable."

## What we measured

The model emits no eval of its own, so we reconstruct one the way a Lichess computer review
does. For every position where Kibitzer was to move, Stockfish (dev build, depth 12, which is
already ~2800 and more than qualified to grade a 2500 model) scores the position, then scores it
again after the move Kibitzer actually played. The difference is the centipawn loss (CPL) for
that move. Average it and you get ACPL, the per-move tax the model pays.

Two honesty knobs matter:

- We only count moves from **contested** positions (`|eval| <= 600cp`). Sloppy technique while
  you are already up a rook, or flailing while already lost, is not why games are decided. It is
  noise, and including it makes every engine look like a genius or an idiot depending on the
  scoreline.
- We drop the void games. The old official-Elo run had four time forfeits from an `st=5` clock
  too tight for 512 sims. Those are the referee disqualifying the model for being slow, not chess
  losses, so they are out.

Corpus: the current strongest checkpoint (`runs/tactical/tactical_repair.pt`) at 64 to 512 sims,
versus the Maia-2700 human-like net and a Stockfish `UCI_Elo` ladder from 1900 to 2900. 336 valid
games survive the filters.

## The one figure that matters

![failure taxonomy](figures/failure_taxonomy.png)

Read it left to right.

**Where it bleeds.** ACPL by phase is 8 in the opening, 33 in the middlegame, 44 in the endgame.
The opening is a solved problem for this model. It has memorized strong opening move
distributions and the book carries the rest, so it plays those moves at roughly Stockfish
accuracy. Then the book runs out, real decisions start, and the tax more than quadruples. By the
endgame it is worse still, and against Maia-2700 specifically the endgame ACPL is a brutal **53**.

**How it loses.** Of 106 decisive losses, **81 (76%) are gradual** and only 25 (24%) are sudden.
Gradual means there was no single move that lost the game. The eval just rots. Sudden means one
throw of 300cp or more from a still-playable position. So three out of four losses are death by a
thousand cuts, not a blunder you can point at.

**Blunder count by phase.** When the model does throw (CPL >= 200), it happens in the middlegame
(151 times) and endgame (119 times) and essentially **never in the opening (1 time in 336
games)**. Whatever is wrong with this engine, it is not an opening-preparation problem.

## What the two loss shapes look like

![loss shapes](figures/loss_shapes.png)

Blue is the common case, red is the rare one. The gradual games (top) are descending staircases.
The model sits around equality, drifts to minus one, shrugs, drifts to minus two, shrugs again,
and by the time the eval is clearly lost there was never a moment where it could have said "that
was the mistake." The sudden games (bottom) are cliffs: fine, fine, fine, and then one move
walks off the edge. Both are real. But the staircase is the disease and the cliff is a symptom.

## Why this specific architecture fails this specific way

This is not a random pattern, it is exactly what you would predict from what Kibitzer is. It is a
single-position model. Context window one. A three-layer attention encoder over the 64 squares,
mean-pooled into one vector, a trunk, and a tiny (~33k param) value head that PUCT leans on at the
leaves. That value head is already established as weak, noisy, load-bearing, and a closed lever
(enlarging it or reweighting it both made play worse, see D52 and D62).

Openings reward pattern memorization, which the policy prior is great at. Endgames punish
everything this architecture cannot do: no game history, no tablebase, no long forcing-line
calculation beyond what the sims scrape together, and precise technique (opposition, pawn races,
zugzwang, conversion) that depends entirely on an accurate value signal. As material thins, the
priors stop carrying the model and it has to actually evaluate, using the one component that is
known to be broken. So it drifts. It keeps playing locally plausible moves while the position
quality decays, because the leaf value is not sharp enough to steer PUCT away from the strategically
losing branches. That is the whole story in one sentence.

GPT's independent read (fed only the numbers and six positions, no leading questions) landed in the
same place: dominant disease is value-driven positional drift, tactical blindness is a significant
24% symptom, and the endgame weakness is the signature of a mean-pooled weak-value net. It also,
unprompted, labeled the six sample blunders as king-in-the-center, loose-piece and back-rank
tactics, premature pawn pushes near its own king, and letting advantages dissipate. Which is to say:
the model's mistakes are the mistakes of something that does not have a stable sense of how good its
own position is.

## The uncomfortable part

We lose real games to weak opponents, and it is the same disease. There are gradual losses to
SF-1900 and SF-2300 in this corpus. Not many, but they exist, and they are not flukes where the
model hung something. They are the staircase again: the model reached a slightly worse middlegame
against a 1900, decided it was fine, and slowly proved itself wrong. A 2500-strength engine should
not be losing to 1900 by conviction, and yet.

## What this says about the next move

This is the part where the analysis earns its keep, because it kills a tempting idea and points at
the boring correct one.

- **The mate-repair patch (D64) is aimed at the wrong target.** Mate delivery and mate defense are
  a slice of the 24% sudden bucket. Even if we fixed every one of them, three out of four losses are
  the staircase, which mate puzzles do nothing for. You do not cure a boiling frog by teaching it to
  recognize the exact instant the water hits 100.
- **The opening-curriculum idea is aimed at the one phase that already works.** ACPL 8, one blunder
  in 336 games. There is no headroom there.
- **Value-head tinkering is closed and this confirms why.** The failure is precisely a weak-value
  failure, and we already know from D52/D62 that you cannot bolt accuracy onto that head in isolation.

What the data supports, and what GPT independently recommended, is the lever this project keeps
arriving at from every direction: **scale the shared backbone.** The model needs a stronger internal
evaluator across the middlegame and endgame, and the representation is where that lives. Everything
else on the table is either a closed lever or a patch for the smaller leak. The bigger leak is that
the engine slowly believes bad positions are playable, and you do not fix belief with a puzzle set.

## Reproduce

```bash
# re-score every model move (CPU, no GPU, ~10 min)
uv run python scripts/analyze_failures.py \
  --pgns reports/sims_sweep/kibitzer_vs2700_s*_g40_seed23.pgn \
         reports/official_elo/official_elo_s512_gpp40.pgn \
         reports/scaling_law/elo_tactical/eval_*.pgn \
  --out reports/failure_analysis/data/failures.jsonl --depth 12 --threads 4

# roll up + figures
uv run python scripts/summarize_failures.py --jsonl reports/failure_analysis/data/failures.jsonl
uv run python scripts/plot_failures.py --jsonl reports/failure_analysis/data/failures.jsonl \
  --out reports/failure_analysis/figures/failure_taxonomy.png
PYTHONPATH=scripts uv run python scripts/plot_eval_curves.py \
  --out reports/failure_analysis/figures/loss_shapes.png
```

Data: `data/failures.jsonl` (per-game records), `data/summary.json` (rollup). The GPT consult
prompt and reply are in the session scratchpad.
