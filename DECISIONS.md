# Kibitzer — Decision Log

Goal: make the kibitzer chess model strong (target **2500+** Elo), climbing toward
beating Stockfish. Hard constraints: **no GitHub pushes**, **≤ $4 GPU budget**,
**non-spot rentals only**, `.env`/HF token **stays local** (never synced to rentals),
push intermediate datasets + models to the HF **`kibitzer` collection**.

Ordering agreed with user: **(1) 100k Stockfish distillation → (2) on-policy
distillation from a strong HF chess model → (3) self-play RL.**

---

## Context inherited from prior sessions
- Student `kibitzer`: ~28M-param causal transformer, policy over 4672 AZ moves + tanh
  value. **History-dependent** (forwards full game history; single-position play ≈ 0%).
- SFT baseline ≈ Stockfish-1320. Value head poorly calibrated (≈0 even up a queen).
- Prior RL/distill attempts (AZ outcome, AZ Stockfish-value, offline Stockfish
  best-move distillation @100k) **all failed to beat the ~1320 baseline**. Offline
  distillation overfit to **unrealistic generated-game histories** → 0% in real games.
  → Lesson: train on **realistic histories** + use a **richer teacher signal**.

## Decisions this session

### D1 — Rent the cheapest *apt* non-spot GPU, not the cheapest GPU
Chose **runpod RTX3090, 32 vCPU (256 cores visible), 125GB RAM, 24GB, $0.48/hr**
(offer aeba94) over a $0.41 L4 (6 vCPU). Rationale: 100k Stockfish labeling is
**CPU-bound**, so vCPU count dominates throughput-per-dollar; 24GB trivially fits the
28M student + 34M teacher. Image `ubuntu_22_cuda_12` (runpod rejected the pytorch
aliases). Pod id `0719dab48a6c4983845a4ae99dfe27dd`.

### D2 — Keep `.env` local; push to HF only from the laptop
Synced a tarball of code + `sft_best.pt` + the cached 22k dataset to the pod, **explicitly
excluding `.env`**. Training runs `hf_push=false` on the pod; checkpoints are pulled back
and pushed to HF locally via `scripts/hf_persist.py` (reads token from `.env`).

### D3 — Persist all artifacts to the HF `kibitzer` collection
Wrote `scripts/hf_persist.py` (create repo + upload + add to collection). So far pushed:
- `Pradheep1647/kibitzer-sft` (baseline model)
- `Pradheep1647/kibitzer-distill-100k` (the 100k distillation dataset, 100,323 positions)
Rationale: never regenerate/retrain the same artifact twice.

### D4 — Extend dataset to 100k, generate on the pod
Used `build_or_extend_dataset` (parallel Stockfish-14.1 labeling, depth 12, 64 workers,
~125 live SF procs) to grow 22k → **100,323 positions** in ~4 rounds. Realistic-ish
generation (random_open_plies=3, mixed Elos 1320–2850).

### D5 — Train distillation with eval disabled on the pod; eval locally
Pod has no cutechess-cli. So: train with `eval_every_steps=0`, save per-epoch checkpoints,
pull them local, eval vs the Stockfish Elo ladder with cutechess here (catches the best
epoch — important given the prior epoch-2 overfitting collapse). batch 32, peak_lr 3e-4.

### D6 — Verified ChessBot as the on-policy distillation teacher (~2500–2600)
`Maxlegrec/ChessBot` (34.7M, PyTorch, LCzero-style, FEN→policy+value). Local bracket
vs Stockfish (4 games each, 50ms): **100% @1500, 100% @2000, 88% @2500**. First teacher
capable of pushing the student past the ~1320 ceiling. API: `get_move_from_fen_no_thinking
(fen, T, return_probs=True)` → `{uci:prob}`; `get_position_value(fen)` → `[bw,draw,ww]`.
Env pitfall: needs transformers 4.44.x + huggingface-hub<1.0; run via `.venv/bin/python`
(uv re-syncs lockfile and reverts pins).

### D7 — Built on-policy distillation pipeline (`kibitzer/opd.py`)
Student self-plays (on-policy → realistic histories *for it*); ChessBot labels each
position with a **dense policy distribution + value**; train with the existing dense
`az_loss` (CE to dense target + value MSE + ref-KL to frozen init). Richer than offline
single-best-move distillation and fixes the history-mismatch failure.

### D8 — Cap Stockfish distillation at 2 epochs, then pivot to OPD  *(user decision)*
Distillation loss was flat (~2.5–2.9, top1 ~0.2, value MAE ~0.25 not improving) —
the known-weak approach. User: run **2 epochs**, eval+push the best checkpoint, then put
remaining budget into OPD (the real path to 2500). Also: maintain this decision log.

---

## Cost ledger
| Item | Rate | Notes |
|---|---|---|
| runpod RTX3090 pod | $0.48/hr | started ~03:19 UTC. ~$0.45 spent at the 2-epoch decision. |

## HF artifacts (collection: `kibitzer`)
- model `Pradheep1647/kibitzer-sft` — SFT baseline (~1320)
- dataset `Pradheep1647/kibitzer-distill-100k` — 100,323 Stockfish-labeled positions

### D9 — Use TRL's JSD loss inside the native kibitzer OPD loop  *(user decision)*
User asked to "use trl for opd." Verified TRL's `GKDTrainer`/`DistillationTrainer`
(on-policy distillation) only wraps HF **CausalLM** text models (token logits + shared
tokenizer + `.generate()`) — kibitzer (4672-move policy/value net, board-tensor input) and
ChessBot (FEN→move-dict) don't fit. User chose the middle ground: keep the custom net +
my loop, but swap the policy term to **TRL's `generalized_jsd_loss`**. Implemented in
`kibitzer/opd.py::opd_loss`: teacher dense dist → teacher logits (`log p`, legal-masked);
JSD(student, teacher) + value MSE + ref-KL. Gotcha fixed: TRL's `batchmean` divides by
`batch*seq`, so logits must be `(B,1,V)` — else the policy term is ~4672× too small.
Installed `trl==0.11.4` (coexists with the transformers 4.44 pin ChessBot needs).

## Open log (append below as work proceeds)
- 2-epoch Stockfish distillation running on pod; OPD (TRL-JSD) built + smoke-tested locally.
- AZ in-training cutechess eval is now disabled by setting `eval_every_iters <= 0`; enabled eval keeps the existing `evaluate_checkpoint` path.
- Added `scripts/train_selfplay.sh` for pod self-play from `checkpoints/sft_best.pt` with eval off and per-iteration checkpoints in `runs_rl/`.
- Local dry-run passed with eval off: 1 self-play game, 1 train update, checkpoint written to `/tmp/kibitzer_az_dryrun/az_iter_1.pt`.

### D10 — Stop distillation at epoch 1 (collapsed to 0%); init OPD from SFT
Pulled the epoch-1 distilled checkpoint (step 3125) and evaluated locally:
**0% vs SF-1320 and 0% vs SF-1800** — worse than the SFT baseline (42% / 17%). Same
overfitting collapse as the prior 100k attempt, now visible at epoch 1. Continuing to
epoch 2 would only waste budget, so killed distillation early. The failed distill
checkpoint is NOT pushed to HF (0% is not worth persisting). **OPD inits from
`checkpoints/sft_best.pt`**, not the distilled net. Phase-1 conclusion: offline
Stockfish best-move distillation does not help this net — confirmed again.
- Launched OPD (TRL-JSD, ChessBot teacher) on the pod from `sft_best.pt`: 3 rounds,
  60 games/round, eval off (eval locally), checkpoints to `runs_rl/opd_round_*.pt`.
- Fixed: ChessBot's `return_probs` crashes on legal moves missing from its vocab
  (knight underpromotions, e.g. `h7h8n`) — `teacher.label` now falls back to one-hot
  best-move, then uniform, so generation never dies.

### D11 — OPD round-1 also collapsed (0% vs 1320); trying a gentle, anchored retry
OPD round-1 (LR 1e-4, temp 0.9, 2 epochs) eval: **0% vs SF-1320, 8% vs SF-1800** — again
worse than SFT (42% / 17%). Pattern across distill + OPD: **any aggressive training
collapses this 28M net below its fragile SFT optimum** (looks like catastrophic
forgetting — overwriting SFT knowledge on the weird self-play positions faster than it
learns the teacher). Killed the 3-round run (rounds 2/3 trained from the degraded net).
Retry: **gentle OPD** = LR 2e-5, kl_coef 0.5 (strong anchor to SFT init), student_temp 0.4
(more realistic positions), 1 round / 1 epoch, 80 games. If this still degrades, the SFT
~1320 is the real architectural ceiling for this net and 2500 is out of reach via distill.

### D12 — Gate all further OPD/distillation spending on rented-GPU eval
User clarified that the laptop should not carry heavier runs because local space is tight,
so evaluation/training should move back to Prime. Current Prime wallet: **$4.93**. No
active pods. Availability check showed the same non-spot **RTX3090 24GB / 32 vCPU /
125GB RAM** shape (`aeba94`, ~$0.48/hr) is available; prefer it over H100/H200 because
this phase is decision-gated evaluation plus a short guarded run, not large-model
pretraining. Claude Code review was attempted for a second opinion, but the CLI failed
with `API Error: Unable to connect to API (ConnectionRefused)`, so Codex proceeds from
repo evidence.

Local sanity eval before the user correction was intentionally tiny and not decision-grade:
4 games at 200ms gave **SFT 0/4 vs SF-1320** and **`opd_gentle.pt` 0/4 vs SF-1320**;
PGNs showed one SFT time-forfeit and repeated tactically losing openings. This is useful
only as a warning that the current eval surface is noisy/sensitive, not as a final Elo
estimate. The rented-GPU gate should therefore run a paired, same-settings eval:

1. `checkpoints/sft_best.pt` vs SF-1320, enough games to re-anchor the baseline.
2. `runs_rl/opd_gentle.pt` vs SF-1320 under the same conditions.
3. If gentle OPD is **not clearly better than SFT** (or at least non-degrading), stop
   ChessBot OPD and offline distillation; do not spend the remaining budget on a path
   with repeated catastrophic forgetting.
4. If gentle OPD preserves/improves the baseline, run only one more **trust-region OPD**
   probe: lower LR, higher reference KL, short horizon, save every round, then re-eval
   before any self-play escalation.

Technical rationale: three separate teacher-forcing paths have shown collapse or weak
transfer, so the next optimal decision is not "more epochs"; it is **sequential
hypothesis testing** under a hard compute budget. The invariant is monotonic strength
preservation against the rated ladder. A checkpoint that cannot preserve SF-1320 strength
must not become the seed for self-play toward 2300/2800, because self-play would amplify
the degraded policy distribution rather than repair it.

### D13 — Stop OPD/distillation; pivot remaining budget to search/AZ
Rented Prime pod `75e9d51acb5d491db2b009bad6406cc0` on RTX3090 24GB (`aeba94`,
~$0.48/hr). Setup copied only source, `checkpoints/sft_best.pt`, and
`runs_rl/opd_gentle.pt`; `.env` stayed local. Pod had Python 3.11 + CUDA but no `uv`,
Stockfish, or cutechess. Installed `uv`, `stockfish`, and used a direct
`python-chess`/`KibitzerEngine` eval because Ubuntu had no `cutechess-cli` package.
This Stockfish build enforces `UCI_Elo >= 1350`, so the rented-gate opponent is
**SF-1350**, not SF-1320.

Prime gate results written to `eval_pgns_prime/`:

| checkpoint / mode | opponent | games | result | score |
|---|---:|---:|---|---:|
| `checkpoints/sft_best.pt` raw | SF-1350 | 8 | 1W / 5L / 2D | 2.0/8 |
| `runs_rl/opd_gentle.pt` raw | SF-1350 | 8 | 2W / 6L / 0D | 2.0/8 |
| `checkpoints/sft_best.pt` + MCTS | SF-1350 | 4 | 2W / 1L / 1D | 2.5/4 |

Decision: **do not spend more GPU on offline distillation or ChessBot OPD**. Gentle OPD
failed the monotonic-improvement gate: same score as SFT and no clear evidence of
strength gain. The only positive signal is inference-time search with material blending
(`Sims=32`, `Material=0.85`, `Cpuct=1.5`), which scored above the raw policies in the
small Prime gate. Next spend should therefore go to **search-coupled AlphaZero-style
self-play/training from `sft_best.pt`**, not teacher-forcing. Any self-play checkpoint
must still pass a paired ladder gate before becoming the new seed.

Budget note: Prime wallet after the gate was **$4.83**; recent billings showed **$0.10**
compute for pod `75e9d51acb5d491db2b009bad6406cc0` plus the existing image charge. Keep
the pod only while actively running gated eval/training.

### D14 — One AZ iteration is the only keeper; continuation regressed raw strength
Ran a bounded AZ/search probe on the same Prime pod, then terminated the pod. Final pod
state: **0 active pods**, wallet **$4.64**, billed **$0.29 compute** for
`75e9d51acb5d491db2b009bad6406cc0` plus the image charge.

Training probe 1:
- Init: `checkpoints/sft_best.pt`
- Config: stockfish curriculum at **SF-1350**, `sims=32`, `material_weight=0.85`,
  `context_window=128`, `value_target=stockfish`, `value_stockfish_depth=4`,
  `peak_lr=5e-5`, `kl_coef=0.2`, 1 iteration, 4 games, 20 update steps.
- Result: `runs_rl/az_prime_gate/az_iter_1.pt`
- Trainer metrics: loss **2.4156**, policy **2.1776**, value **0.2380**, ref-KL
  **0.0001**, entropy **2.0369**, 130 samples, 20 updates.
- Gate: raw **2.0/8** vs SF-1350 (same as SFT), search **3.0/4** vs SF-1350
  (better than SFT+search **2.5/4** in the small gate).

Training probe 2:
- Init: `runs_rl/az_prime_gate/az_iter_1.pt`
- Config: same curriculum, lower `peak_lr=3e-5`, 2 continuation iterations.
- Results pulled: `runs_rl/az_prime_gate_cont/az_iter_1.pt` and
  `runs_rl/az_prime_gate_cont/az_iter_2.pt`.
- Iter metrics: iter1 loss **2.9372**, policy **2.6678**, value **0.2694**,
  ref-KL **0.0001**, 127 samples; iter2 loss **2.9219**, policy **2.6555**,
  value **0.2663**, ref-KL **0.0002**, 221 samples.
- Gate: `az_prime_gate_cont/az_iter_2.pt` raw eval was **0.0/8** vs SF-1350.
  Search eval was interrupted after raw failed the stop condition; partial search was
  2/2, but raw collapse dominates the decision because raw policy preservation is the
  seed-quality invariant for future self-play.

Decision: keep `az_prime_gate/az_iter_1.pt` as an experimental search-compatible
checkpoint, but **do not continue from `az_prime_gate_cont/az_iter_2.pt`** and do not
promote it. The pattern is now consistent across OPD, distillation, and longer AZ
continuation: the net can get a short search-assisted tactical bump, but additional
gradient steps rapidly damage the raw policy. Next architecture/training decision should
focus on preserving raw policy while improving search targets: smaller LR, replay mixing
with SFT positions, stronger frozen-reference KL to `sft_best`, and promotion only when
raw SF-1350 score is non-decreasing and search score improves.

### D15 — Make the Prime gate reproducible before spending more GPU
The previous Prime eval was run through one-off Python snippets because the rented image
had Stockfish but no `cutechess-cli`. To avoid future decision drift, added
`scripts/gate_stockfish.py`: a direct `python-chess`/`KibitzerEngine` gate that supports
multiple checkpoints, raw or MCTS-backed play, JSON output, context-window control, and
automatic clamping when the installed Stockfish build has a higher `UCI_Elo` minimum
(e.g. Ubuntu Stockfish 14.1 rejected 1320 and required 1350).

This turns checkpoint promotion into a repeatable invariant:

```bash
uv run python scripts/gate_stockfish.py \
  --checkpoint sft=checkpoints/sft_best.pt \
  --checkpoint az1=runs_rl/az_prime_gate/az_iter_1.pt \
  --stockfish-path /usr/games/stockfish \
  --elo 1350 --n-games 8 --device cuda \
  --out eval_pgns_prime/raw_gate.json

uv run python scripts/gate_stockfish.py \
  --checkpoint az1=runs_rl/az_prime_gate/az_iter_1.pt \
  --stockfish-path /usr/games/stockfish \
  --elo 1350 --n-games 4 --device cuda --search \
  --sims 32 --material 0.85 \
  --out eval_pgns_prime/search_gate.json
```

Verification: `uv run pytest tests/test_gate_stockfish.py -q` passed (**3 passed**),
`uv run python scripts/gate_stockfish.py --help` worked, and
`uv run python -m compileall -q scripts/gate_stockfish.py tests/test_gate_stockfish.py`
passed. This is a tooling decision, not a model-strength claim; no GPU was rented for it.

### D16 — Add SFT-anchor replay before the next AZ spend
D14 showed the failure mode clearly: short AZ/search training can improve search-mode
play, but more gradient steps can collapse the **raw policy manifold** to 0/8. Before
renting again, added an optional supervised-anchor replay path in `kibitzer/az_trainer.py`.
When `--sft-anchor-data-dir` is provided, the trainer parses elite PGNs into replay
anchors: real game histories, one-hot targets for the human move, and side-to-move
game-result values. Each AZ iteration mixes a configurable fraction of those SFT anchors
into replay after collecting search/Stockfish samples.

New knobs:

```bash
--sft-anchor-data-dir data/raw
--sft-anchor-min-elo 2400
--sft-anchor-max-games 64
--sft-anchor-max-plies 160
--sft-anchor-fraction 0.25
```

Decision: the next Prime AZ probe should start from `checkpoints/sft_best.pt` or the
non-collapsed `runs_rl/az_prime_gate/az_iter_1.pt`, but must include SFT-anchor replay,
strong reference KL, and the reproducible SF-1350 raw/search gate from D15. Promotion
requires raw SF-1350 score to be non-decreasing and search score to improve; any raw
collapse means immediate rollback. This is a **trust-region / rehearsal-buffer**
intervention against catastrophic forgetting, not another attempt to push epochs harder.

Suggested next Prime command shape:

```bash
uv run python scripts/train_az.py \
  --init-checkpoint checkpoints/sft_best.pt \
  --output-dir runs_rl/az_anchor_gate \
  --device cuda --dtype bfloat16 \
  --opponent stockfish --stockfish-path /usr/games/stockfish \
  --stockfish-levels 1350,1500,1800 --stockfish-time-ms 50 \
  --value-target stockfish --value-stockfish-depth 4 \
  --sft-anchor-data-dir data/raw --sft-anchor-fraction 0.25 \
  --sft-anchor-max-games 64 --sft-anchor-max-plies 160 \
  --n-iterations 1 --games-per-iter 4 --max-plies 120 \
  --sims 32 --material-weight 0.85 --context-window 128 \
  --train-steps-per-iter 20 --batch-size 32 \
  --peak-lr 3e-5 --min-lr 1e-5 --kl-coef 0.5 \
  --eval-every-iters 0 --checkpoint-every 1 --hf-push false
```

Verification: `uv run pytest tests/test_az_anchor.py tests/test_gate_stockfish.py -q`
passed (**7 passed**), `uv run python -m compileall -q kibitzer/az_trainer.py
scripts/gate_stockfish.py tests/test_az_anchor.py tests/test_gate_stockfish.py` passed,
and `.venv/bin/python scripts/train_az.py --help` showed the `--sft-anchor-*` flags.
No GPU was rented for this change.

### D17 — Claude review confirms SFT start; Prime pod attempts were not connectable
Claude Code was available again and reviewed D14-D16. Recommendation: start the anchored
AZ probe from **`checkpoints/sft_best.pt`**, not `az_prime_gate/az_iter_1.pt`, because
`az_iter_1` only had a small 4-game search-mode bump and no raw improvement. The anchored
run should be treated as an intervention test, not as a continuation of an unvalidated
checkpoint. Claude also recommended a stricter stop rule:

- run **one anchored iteration only**;
- abort immediately if raw SF-1350 falls below the SFT raw baseline;
- promote only on a larger gate, at least **16 raw games** and **8 search games**, where
  raw is non-decreasing and search improves beyond obvious sampling noise;
- if raw nudges down but does not collapse, the next cheaper intervention is higher
  anchor fraction, not more iterations.

Local prerequisite check found **no local `data/*.pgn` anchor files**, so the correct
execution plan is to download a small Lichess Elite sample on the rented pod, not onto
the laptop. Prime availability initially failed in-sandbox with DNS errors, then worked
with an escalated query. Cheap known RTX3090 was unavailable, so three short pod attempts
were made and terminated when they failed to produce SSH:

| pod | provider / GPU | result | billed |
|---|---|---|---:|
| `b154d3e39af546999b425affa41c2a29` | massedcompute A6000 48GB | stuck `PROVISIONING`, no SSH | $0.00 |
| `cf23443355c74787b1f47bab63e2f41c` | massedcompute RTX6000 Ada 48GB | stuck `PROVISIONING`, no SSH | $0.00 |
| `92b1b40e3c344ea8a708ad9298165e00` | datacrunch A100 40GB | stuck `PROVISIONING`, no SSH | $0.00 |

Final state after cleanup: **0 active pods**, wallet **$4.64**. Decision: do not keep
churning providers while the platform is failing to attach; retry later with the same
SFT-start anchored plan. If Prime availability is healthy, prefer a provider that reaches
SSH quickly over the absolute cheapest listed GPU, because setup latency is now the
dominant risk to the remaining budget.

### D18 — Retry with Crusoe also failed to attach; preserve budget
Retried after a successful escalated availability query. Prime showed wallet **$4.64** and
no active pods. Cheapest available non-spot GPUs were again massedcompute A6000/RTX6000
Ada, but those providers had just failed to expose SSH in D17, so chose a different
backend: **Crusoe L40S 48GB** (`0b28ef`, ~$1.00/hr, pod
`3d764385665d48368a9198078188e388`). Result: pod stayed `PROVISIONING` with installation
`PENDING`, no IP/SSH. Terminated before setup. Final state: **0 active pods**, wallet
**$4.64**, compute billing for the Crusoe attempt **$0.00**.

Decision: the anchored AZ run remains the correct next experiment, but Prime provisioning
is the blocker, not model code. Do not keep spinning pods in a tight loop. Next retry
should first require a provider/pod that reaches SSH, then:

1. download a small Lichess Elite sample on the pod;
2. run one SFT-start anchored AZ iteration with `--sft-anchor-*`;
3. gate with `scripts/gate_stockfish.py` using SFT baseline vs anchored checkpoint;
4. promote only if the D17 criteria pass.

### D19 — Convert the next rented-GPU attempt into a single pod-side gate
Prime provisioning has become the immediate bottleneck, and the remaining budget is small
enough that setup latency matters. Added `scripts/run_anchor_gate_prime.sh` so the next
successful SSH pod can run the anchored AZ experiment without retyping one-off commands.

The script deliberately does **not** auto-promote or push anything. It:

1. verifies `uv`, `checkpoints/sft_best.pt`, and Stockfish are present;
2. downloads one Lichess Elite sample into `data/raw` only if no local anchor PGN exists;
3. runs exactly one SFT-start anchored AZ iteration with the D16/D17 settings;
4. gates SFT vs the anchored checkpoint with 16 raw SF-1350 games and 8 search games;
5. exits with rejection if raw regresses below SFT or search fails to improve.

Decision: use this runner only after a pod reaches SSH. This keeps the rented GPU path
reproducible and preserves the laptop by downloading anchor data on the pod. No Prime pod
was rented for D19, and nothing was committed or pushed.

Verification: `bash -n scripts/run_anchor_gate_prime.sh` passed, and the script was made
executable.

### D20 — Pivot to the searchless-chess recipe: dense ChessBot labels on REAL games, train a fresh base on the laptop
Repeated catastrophic forgetting (OPD/distill/AZ all produced `-minus-` Elo
checkpoints on HF) confirmed the SFT base is a fragile, *undertrained* optimum, not an
architectural ceiling. Grounded the next phase in literature instead of more probes:

- **Grandmaster-Level Chess Without Search** (DeepMind, arXiv:2402.04494): supervised
  distillation of an engine's **dense per-move action-values** on **real** games, **no
  self-play, no search**. 9M→2054, 136M→2156, 270M→2299 vs bots (270M = **2895 vs
  humans on Lichess**). Their ablation: strength "only emerges at sufficient dataset+
  model scale" — 10k–100k *games* is too little; we'd been training on 100k *positions*.
- **Stop Regressing** (arXiv:2403.03950): HL-Gauss value-as-classification for a
  scalable, calibrated value head (the documented fix for our dead value head).

**Target reframe (honest):** a 28M net sits between their 9M/136M, so the realistic
ceiling is **~2050–2150 vs engines**. The "2800" goal is a *Lichess-vs-humans* number:
their 270M was 2299 vs bots but 2895 vs humans (a +596 bot/human gap), so ~2100–2200 vs
engines plausibly reads ~2700–2800 on Lichess blitz vs humans. **2800 engine-strength is
not realistic for this net, and self-play is not the path** — a strong supervised base is.

**Decisions:**
- **Label locally, not on free Colab.** Dense labeling is CPU-bound (python-chess legal
  moves / encoding). The laptop has **24 threads** vs free Colab's ~2 vCPU; ChessBot
  (34.7M) needs **<1GB VRAM**, so the 4060's 8GB and 16GB T4 are both overkill. No robust
  `colab-cli` for headless GPU rental; rejected.
- **Real positions only.** Walk real Lichess PGNs; **do not** reuse
  `kibitzer-distill-100k` (the failed unrealistic-history set) or any `kibitzer-az/
  distill-elo-minus-*` HF checkpoint (all worse than baseline).
- **Train from scratch.** Every warm-start from the SFT manifold collapsed; random init +
  the searchless recipe cleanly escapes it. `checkpoints/sft_best.pt` (the ~1212 baseline,
  already local; canonical `kibitzer-sft`) stays only as the gate baseline / optional
  warm-start knob. **Nothing needs pulling from the HF collection for this.**
- **HL-Gauss deferred:** kept the scalar tanh value head + MSE for now (the deadness was
  collapse/undertraining, not MSE). HL-Gauss becomes an *additive* value-bucket head only
  if the gate shows value is still miscalibrated.

**Built (all local, 7 tests passing, compileall clean):**
- `kibitzer/chessbot_label.py` + `scripts/label_chessbot.py`: stream real PGNs → label each
  position with ChessBot's dense legal-move policy + value → **sparse, sharded** storage
  (~280 B/position → 1M ≈ 280 MB). Teacher imported lazily; parsing works in any env.
- `kibitzer/shard_trainer.py` + `scripts/train_shards.py`: **RAM-bounded streaming**
  (one shard + a fixed shuffle buffer, independent of dataset size), dense-policy CE +
  value MSE, grad accumulation (eff. batch 256 on 8GB), cosine LR, **`--resume`** for
  multi-day runs. Scale lever is purely labeling volume — aim for **5–10M positions**, not 1M.
- Tests: `tests/test_chessbot_label.py`, `tests/test_shard_trainer.py` (real CPU train + resume).

**Measured resource use:** base imports **367 MB** RSS; full labeling process est.
**~2–2.5 GB host RAM**, **<1 GB VRAM** — no need to close apps for RAM, only mild VRAM
headroom for a long run. Disk: 24 GB free, ample.

**Env pitfall (recurring):** the venv reverted to `huggingface-hub==1.14.0`, breaking the
`transformers 4.44` ChessBot needs. Re-pin via the venv python directly (sticks because we
do **not** use `uv run` for the teacher):
`.venv/bin/pip install "transformers==4.44.2" "huggingface-hub<1.0"`.

**Blocker / next step:** no local `data/raw/*.pgn` yet (same as D17). The chain is
label → train → gate; training cannot start until a Lichess Elite sample is fetched and
labeled. Run order:
```bash
.venv/bin/pip install "transformers==4.44.2" "huggingface-hub<1.0"
.venv/bin/python scripts/label_chessbot.py --pgn-dir data/raw \
  --out-dir data/chessbot_labeled --min-elo 2200 --position-stride 2 --device cuda
uv run python scripts/train_shards.py --data-dir data/chessbot_labeled \
  --output-dir runs_base --batch-size 16 --grad-accum-steps 16 --epochs 4
uv run python scripts/gate_stockfish.py --checkpoint sft=checkpoints/sft_best.pt \
  --checkpoint base=runs_base/base_final.pt --elo 1350 --n-games 16
```

### D21 — Executed the dense-policy base run end-to-end; it converged but is strictly weaker than SFT. Policy distillation from ChessBot is a dead end; pivot to action-values
Ran the full D20 plan locally on the 4060 (no GPU rental; wallet untouched).

**Pipeline executed (all built + tested this session):**
- **Labeling sped up ~12x.** `kibitzer/chessbot_label.py` got `BatchTeacherLabeler` (batches ChessBot's
  forward — policy *and* value in one pass — + dict move-index lookup replacing an O(1929) `list.index()`
  called ~60k times/position) and `label_pgn_dir_parallel` (3 GPU-sharing spawn workers, round-robin
  game shards). Bench: ~110 pos/s -> **~1,030 pos/s**, peak ~1.2GB VRAM. (6-worker test: same throughput,
  GPU-saturated at 3.)
- **Data:** labeled `data/raw/lichess_elite_2024-11.pgn` (272,548 games, 2300/2500+) ->
  **11,788,146 positions**, 546 sparse shards, ~3h. Pushed public to
  **`Pradheep1647/kibitzer-chessbot-dense-12m`** (added to the `kibitzer` collection) via a new
  `--folder` path in `scripts/hf_persist.py` (`upload_large_folder`).
- **Trainer:** `kibitzer/shard_trainer.py` streams shards RAM-bounded with a DataLoader **prefetch**
  pipeline (`ShardIterableDataset`, 6 workers encode in parallel, pin_memory) so the GPU stays fed.
- **Gate hardened:** `scripts/gate_stockfish.py` gained `--opening-random-plies` + `--seed` (seeded
  openings shared across checkpoints = paired, low-variance comparison) after 16-game evals proved too
  noisy (SFT swung 2.5/16 -> 5.5/16 between runs).

**Training:** from scratch, ChessBot **dense policy** targets + scalar value MSE, ctx-window 8, eff-batch
256, cosine LR. **GPU-bound at ~427 pos/s** (4060 power-capped at 77W; 92-95% util — prefetch worked, the
GPU itself is the ceiling, ~7 effective TFLOPS not the 15-30 I estimated). 2 epochs = ~15h. Loss converged
cleanly (policy 3.35->~1.77, value 0.28->0.056) — **the first training run in this project's history that
did NOT collapse** (cf. all the `-minus-` OPD/distill/AZ checkpoints). Stopped at **step 68,000/92,096**
(checkpoints every 1k; resumable) once the eval verdict was conclusive.

**Eval verdict (SF-1350):**
| | raw | search (sims32, mat0.85) |
|---|---:|---:|
| SFT baseline | 27.5% (11/40, paired) | **50% (4/8)** |
| base @ step 65-67k | **7.5% (3/40, paired)** | **6% (0.5/8)** |

The strengthened 40-game paired gate made it unambiguous: base is **-20pts vs SFT raw (0 wins/40)**, and
**search rescues SFT (27.5->50%) but does nothing for the base (7.5->6%)**. The base overfit to one greedy
line and flounders off-book; its value head/policy prior aren't good enough for MCTS to help.

**Conclusion:** distilling ChessBot's **policy head** is a confirmed dead end — loss converged perfectly
(matched the teacher's distribution) but **ChessBot's policy head is itself only ~SF-1300; its 2500 strength
lives in value-based move selection (`get_best_move_value`/`calculate_move_values`), which we didn't
distill.** Not pushed to HF. Decision: do **not** finish the run (evidence-based; converged + strictly
sub-SFT in both modes).

**Next:** the now-proven pipeline stays; only the **target** changes — relabel the same 12M positions with
**per-move action-values** (ChessBot's `calculate_move_values`, or Stockfish), the actual
searchless-chess recipe (DeepMind 2402.04494) that reached 2000+. Reuses labeler/trainer/prefetch/gate
wholesale. The 12M-position dataset, the non-collapsing training recipe, and the hardened paired gate are
the durable assets from this session.

### D22 — Built + launched the action-value pivot (reuse infra, swap target); cleaned disk first
User chose **reuse infra, swap to action-values** (not a from-scratch rebuild; keep DECISIONS.md — the
record of dead ends is the value). User also asked to clean up first ("more headspace").

**Cleanup (disk was at 98%, 5GB free):** deleted, with approval, the 67 abandoned policy checkpoints
(kept only `runs_base/base_step_0068000.pt` for the record), the local `data/chessbot_labeled` (5.6GB,
already public+verified on HF), and `runs_rl/*` (3.4GB of prior PPO/AZ/OPD/distill artifacts, D8-D16).
**5GB -> 25GB free.** Kept `data/raw` (PGN, needed) and `checkpoints/sft_best.pt` (baseline).

**Action-value labeler (additive to `kibitzer/chessbot_label.py`):** `LabelConfig` got `label_mode`
("policy"|"action_value") + `av_temp`; `BatchTeacherLabeler` got `_label_actionvalue` — for each board it
evaluates the position **after every legal move** (batched via ChessBot's `get_batch_position_values`),
takes the mover-perspective value (ww-bw, sign-flipped for black), and builds the policy target as
`softmax(move_values / av_temp)` (peaked on the value-best move) with the scalar value = best move-value.
**Verified correct:** the labeler's argmax move matches ChessBot's own `calculate_move_values` best move on
every test position. Shard format/trainer/gate all **unchanged** — full infra reuse.

**Throughput:** **~43 pos/s** (vs ~1030 for policy) — it does ~30x more forwards/position (one per legal
move). So 12M is infeasible (~77h); sized the first run small.

**Gate hardening recap (this session):** `scripts/gate_stockfish.py` gained `--opening-random-plies` +
`--seed` (seeded openings shared across checkpoints = paired, low-variance). 16-game gates were too noisy
(SFT swung 2.5->5.5/16); use the 40-game paired gate. Reference scores vs SF-1350 to beat:
**SFT raw 27.5% (11/40), SFT search 50% (4/8); failed policy-base raw 7.5%, search 6%.**

**Launched (overnight, ~5h, hands-off):** ~**500k action-value positions** (`--max-games 11500`,
stride 2, 3 workers -> `data/chessbot_av`, ~3.2h), then auto-pipeline: from-scratch train (batch 64,
grad-accum 4, **4 epochs**, prefetch 6 workers -> `runs_av`, ~1.3h), then the paired raw+search gate vs
SFT (`eval_pgns_prime/av_gate.json`, `av_search_gate.json`). Phased ScheduleWakeup loop drives it; will
NOT push to HF without asking. 500k chosen for a fast first read on whether action-values transfer; scale
to 1-2M only if it beats SFT.

### D23 — Action-value run scored 3% (flat); root-caused to a soft-target bug (av_temp), NOT the approach
Ran the 500k action-value pipeline. Intermediate gameplay gates (the D-eval the user asked for) showed a
**flat trajectory: step2000 = 3%, step5000 = 3%** vs SF-1350 (SFT 27.5%, failed policy-base 7.5%). User
chose **stop + analyze** rather than grind. The analysis found the real cause:

- The trained model plays **near-randomly**: greedy top-move prob ~0.10-0.16 (vs SFT 0.44-0.71), doesn't
  match ChessBot's value-best move, even plays junk (h7h5 in a QGD). Training entropy stayed ~2.79 (≈ uniform
  over ~16 legal moves).
- **Root cause = target construction.** ChessBot's per-move action-values cluster tightly (good moves all
  near the top value; only blunders far below), so `softmax(values / av_temp=0.1)` gave a **near-uniform
  target — best move only 0.14-0.21 prob.** We trained on mush; the model learned mush. ChessBot's 2500
  strength is its *decisive argmax*; av_temp=0.1 softened it away. Temp sweep confirmed: best-move target
  prob 0.14-0.21 @0.1 → 0.34-0.65 @0.03 → 0.65-0.93 @0.01.
- **Confound also noted:** 500k AV vs 12M policy is 24x less data, so the 3% is not a clean
  signal-quality comparison regardless.

**Conclusion:** action-value distillation is NOT disproven — the av_temp=0.1 default neutered the signal.
**Fix:** sharpen the target — one-hot on the value-best move (pure behavioral cloning of ChessBot's argmax
play) or a small temp (~0.01-0.02). Better still, store the **raw per-move values** in the shards so the
temperature becomes a free train-time knob (no re-label to retune). Either fix needs one re-label (~3.2h for
500k; the shards stored the softmaxed distribution, not raw values). Stopped training at step 7000; latest
ckpt `runs_av/base_step_0007000.pt`. Diagnostic method (greedy-move vs teacher-argmax + target-sharpness
sweep) is the reusable tool that caught this — gameplay/entropy, not loss, is the signal (loss "converged"
at 2.8 while play was random).

### D24 — Drop history-dependence: rebuild single-position from scratch, Maia-style human-move cloning
After a brutal-honesty review of the whole project: **nothing has ever beaten the SFT ~1320 baseline** across
23 decisions. Root insight — we kept changing the *signal* (policy/value/self-play/distill) but never the
real handicap: **kibitzer is history-dependent** (causal trunk over a sequence of past boards), which is wrong
for chess.

Why history is a liability here, not an asset:
- **Chess is fully observable + Markovian.** The current FEN (board + side-to-move + castling + en-passant +
  halfmove clock) is a *complete sufficient statistic* for the best move. The path to a position does not
  change which move is best. History carries ~zero extra signal for move quality (only threefold-repetition /
  50-move depend on history, and those need just the FEN clock / a tiny fixed window — AlphaZero's 8-position
  stack is for repetition, not better moves).
- **With limited data, history actively hurts:** the model overfits to game-path/style correlations instead of
  position→move. That IS the recurring failure — D10 ("overfit to unrealistic histories → 0% real games"),
  the off-book fragility (opening-random-plies wrecking from-scratch models), the ctx-8-vs-128 mismatch. A
  position-only model *cannot* make that mistake.
- Best case a history model only *ties* position-only (no extra signal to win with) and needs far more data to
  learn to ignore the history. Every strong engine (Stockfish, Leela, AlphaZero, searchless-chess) is
  position-based.

**Decision (user):** drop history; train single-position from scratch. Architecturally this is NOT a new model
family (not an RNN — that's *more* history) — it is the existing transformer with the **history half removed**:
keep the position encoder (transformer over the 64 squares), feed **one board** (context_window=1; the causal
trunk over a length-1 sequence is a no-op for history). Reuse the whole pipeline (labeler shard format,
shard_trainer, paired gate) at context_window=1.

**Phase-1 signal = behavioral cloning of elite human moves (Maia-style), NO teacher:** target = one-hot of the
move the 2300/2500+ player actually played + side-to-move game result as value. This removes every signal bug
we've hit (no unrealistic histories, no weak teacher policy head, no near-uniform action-value targets) and the
target is unambiguous/trivially validated. Labeling needs no GPU/teacher (just PGN parsing) so it is fast and
CPU-parallel → "more games": label the **full elite set** (272k games) at single-position scale. Realistic
ceiling for BC-of-humans is ~1800-2000 (can't exceed the humans it learns from); exceeding that later needs
value+search. Methodology fix from the review: **validate small first** (overfit a tiny batch / check target
sharpness) before any long run.

### D25 — Completed the clean-rebuild policy/value run; search reduces some errors but the model is still below SF-1320
Executed the new position-only training pipeline locally on the RTX 4060 Laptop GPU. The launcher downloaded
and cached six Lichess Elite months (`2025-06` through `2025-11`), then trained policy and value in separate
stages with mandatory Hugging Face uploads.

**Policy stage:** elite-human move cloning, 5M positions/epoch, batch 128, 3 epochs. Each epoch had 39,064
batches and took about 56 minutes; the full policy stage took about 2h50m. The final checkpoint was uploaded to
`Pradheep1647/kibitzer-clean-policy`. This stage trains the shared representation and policy head; the value
head stays frozen.

**Value stage:** 250k positions labeled by Stockfish at depth 14. The first implementation labeled positions
sequentially; a live benchmark exposed that bottleneck, so labeling was changed to eight independent Stockfish
workers. The real run labeled all positions in **45m10s at 92.23 positions/s**. Games were split before training
to prevent position leakage: **224,990 train / 25,010 held-out evaluation**. Only the value head was trainable.

| epoch | MSE ↓ | MAE ↓ | Pearson ↑ | sign accuracy ↑ | R² ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0654 | 0.1671 | 0.5065 | 65.08% | 0.2552 |
| 2 | 0.0645 | 0.1653 | 0.5153 | 64.93% | 0.2647 |
| 3 | 0.0638 | 0.1639 | 0.5225 | 65.60% | 0.2730 |
| 4 | **0.0637** | 0.1639 | **0.5235** | 66.28% | **0.2736** |
| 5 | 0.0640 | **0.1638** | 0.5207 | **66.58%** | 0.2709 |

Epoch 5 was a plateau, not a reason to keep training the small head: MSE/Pearson/R² peaked at epoch 4 while
MAE/sign accuracy improved only marginally at epoch 5. The final checkpoint was uploaded to
`Pradheep1647/kibitzer-clean-value`.

**PUCT gate:** added legal policy/value inference, side-to-move-correct PUCT backup, paired openings, color
alternation, PGN output, and a limited-strength Stockfish gate. The value-sign path was tested independently:
a synthetic favorable root branch received 123 visits versus 5 for the unfavorable branch, ruling out the
common negamax sign bug.

| gate vs SF-1320 | W-D-L | score | capped ACPL ↓ | major blunders ↓ |
|---|---:|---:|---:|---:|
| 64 simulations | 0-2-8 | 10% | 126.7 cp | 51 |
| 256 simulations | 0-1-9 | 5% | **118.7 cp** | **42** |

The 10-game match scores are too noisy to interpret as a precise Elo comparison. The independent move analysis
is clearer: 256 simulations modestly reduced average loss and major errors, but all decisive games were still
losses by checkmate. More search is therefore not the current lever. The policy is still human imitation and
the Stockfish-trained value head sits on a frozen human-policy representation; deeper PUCT can only partially
repair that mismatch.

### D26 — Next phase is cached joint Stockfish distillation, not more search
Built `scripts/train_joint_distill.sh` for the next run. It starts from `runs/value/value_final.pt` and teaches
the policy and value jointly from Stockfish depth-14, MultiPV-8 targets. To avoid destroying the useful base,
the default trainable scope is both heads, final norm, and only the last three trunk blocks; the position encoder
and early trunk stay frozen.

**Default run:** 250k positions, eight Stockfish workers, MultiPV 8, five epochs, batch 128, LR `1e-4`, value
weight `1.0`, and a 10% game-disjoint evaluation split. Teacher labels are written atomically to
`data/stockfish/joint_d14_mpv8_250000.pt`; a matching rerun loads the cache and skips Stockfish entirely, while
mismatched label settings fail explicitly instead of silently reusing stale targets.

The trainer now reports averaged epoch losses rather than the last batch, plus held-out policy cross-entropy,
teacher top-1 agreement, teacher-set coverage, value MSE/MAE/Pearson/sign accuracy/R², stage durations,
trainable parameter count, and whether a new best checkpoint was saved. The checkpoint chosen by held-out joint
score is saved to `runs/joint_distill/joint_best.pt` and HF push is fixed on to
`Pradheep1647/kibitzer-clean-joint-distill`. The known `pynvml` package warning is suppressed narrowly in all
launchers; unrelated warnings remain visible.

Both cache-miss and cache-hit smoke runs passed, including real Stockfish MultiPV labels and checkpoint reload.
The suite is at **40 passing tests**. The full joint run has **not** been executed yet. Command:

```bash
bash scripts/train_joint_distill.sh
```

### D27 — Joint distillation finished; 256-sim search regressed again, so diagnose value/search before any new training
Executed D26 end-to-end. Eight Stockfish workers labeled 250k depth-14 MultiPV-8 positions in **6h34m40s
(10.56 positions/s)** and saved the reusable cache at
`data/stockfish/joint_d14_mpv8_250000.pt`. The game-disjoint split stayed 224,990 train / 25,010 eval.
Training both heads, final norm, and the last three trunk blocks exposed 9,549,633 trainable parameters
(29.7%); each GPU epoch took about two minutes.

| epoch | policy CE ↓ | teacher top-1 ↑ | teacher coverage ↑ | value MSE ↓ | value sign ↑ | value R² ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6014 | 30.71% | 73.66% | **0.0717** | **66.74%** | **0.2872** |
| 2 | 2.5829 | **30.75%** | **73.77%** | 0.0721 | 66.12% | 0.2833 |
| 3 | **2.5801** | 30.72% | 73.65% | 0.0732 | 65.60% | 0.2731 |
| 4 | 2.5881 | 30.49% | 73.37% | 0.0744 | 65.26% | 0.2606 |
| 5 | 2.6015 | 30.41% | 73.01% | 0.0752 | 65.29% | 0.2530 |

Training overfit after epoch 3, but the more important discovery is a **checkpoint-selection scale bug**:
the selector minimized `policy_CE + value_MSE`. CE changes were about 10-20x larger than MSE changes, so it
selected epoch 3 for a negligible policy gain while value quality had already fallen from epoch 1. Only the
epoch-3 `joint_best.pt` survived because each new composite best overwrote the previous file. Future multi-head
training must save every epoch and use constrained/lexicographic selection, not add unlike metrics directly.
The selected checkpoint was uploaded to `Pradheep1647/kibitzer-clean-joint-distill`.

**Search gate:** joint at 64 simulations scored 2-0-8 (20%) versus nominal SF-1320; the same checkpoint at
256 simulations scored 0-1-9 (5%). This repeats the pre-joint `10% -> 5%` pattern. Ten games over five opening
pairs are too noisy for Elo or small model comparisons, but repeated deeper-search regression plus only
~65-67% value-sign accuracy makes value amplification the prime suspect. Stop increasing simulations and stop
using `UCI_LimitStrength + time` WDL as the primary development signal.

### D28 — Opus-reviewed next step: locked common-oracle diagnostics with explicit stop/go gates
Ran four direct `claude -p --model claude-opus-4-8` critique/revision cycles. The final plan was approved only
after fixing three experimental-design errors: Phase-2 (MultiPV 1) and joint (MultiPV 8) logged value metrics
are not directly comparable; checkpoint selection and final gating cannot reuse one dataset; and a 25-Elo SPRT
is not laptop-bounded. The accepted decision is **diagnose before training**.

**Common oracle contract:** sample only real games from entirely unseen PGN months (default Jan-May 2025).
A depth-10 prescan oversamples candidates, then Stockfish depth-20 MultiPV-1 re-bins them by absolute score:
`<0.5`, `0.5-2`, `2-5`, `>5` pawns. Lock at least 200 positions/bin in each of two game-disjoint splits:
validation for checkpoint/config selection and an untouched test split that can be consumed only once. Add
months rather than relaxing bin counts if decisive positions are scarce. All models use the identical target
transform `clip(cp / 1000, -1, 1)`; arbitrary selected moves get independent depth-20 evaluations cached by
FEN+move.

**Metrics/gates:** raw policy and PUCT report exact-best/within-50cp accuracy, mean/p90/p95 capped regret, and
paired bootstrap confidence intervals. Values report MAE/sign accuracy per oracle bin plus natural-distribution
reweighting. Test gates are: 2-5-pawn sign >=90%, >5-pawn sign >=95%; paired regret and near-best confidence
intervals above zero; p90 regret reduced by both >=15cp and >=10%; and no regression versus the baseline on a
fixed tactical sanity suite. Tune `value_scale={0,0.5,1}` at 64 simulations first; only spend on simulation or
`c_puct` sweeps if nonzero value passes.

**Diagnostic routing:** if policy coverage fails while value passes, try cached MultiPV-8 policy-head-only
distillation with trunk/value frozen; allow at most one trunk block plus policy-retention KL only if the cheap
head probe improves validation. If value fails (expected), train the value head plus at most one or two upper
trunk blocks with policy logits anchored to the current policy champion. Save every epoch; reject checkpoints
below baseline value floors, then rank decisive sign, decisive MAE, global MSE, policy CE. No large relabeling
until one cached-label branch passes validation and the untouched test.

**Implemented now:** `kibitzer/diagnostics.py`, `scripts/diagnose_search.py`, and
`scripts/run_search_diagnostics.sh`; PUCT and the match launcher now accept `value_scale`. The launcher has
three explicit actions: `build` creates the locked oracle, `validate` compares Phase-2/joint and value scales,
and `test` requires an explicitly selected checkpoint/scale and writes a permanent consumption lock. A fixed
30-position mate-in-one EPD suite provides baseline-relative legality/sign sanity (not an Elo benchmark).
Oracle and selected-move caches are atomic/reusable. Mocked oracle-build tests and a real CUDA + Stockfish
evaluation/cache smoke passed. The full unseen-month depth-20 oracle has **not** been built yet.

Run order:

```bash
ACTION=build bash scripts/run_search_diagnostics.sh
ACTION=validate bash scripts/run_search_diagnostics.sh
# ACTION=test only after validation selects one checkpoint/value scale.
```

### D29 — Common-oracle validation rejects joint and deeper search; audit the depth-14 label ceiling next
Built the full locked oracle from five unseen real-game months (`2025-01` through `2025-05`). Stockfish
depth-20 MultiPV-1 produced 800 validation and 800 untouched test positions, game-disjoint and exactly balanced
at 200 positions in each absolute-value bin. The natural unseen-game distribution estimated during prescan was
33.46% quiet, 32.29% edge, 24.47% decisive, and 9.78% won. Only validation has been consumed; the test split
remains locked.

**Value verdict:** both checkpoints fail on the positions that search most needs them to recognize.

| checkpoint | natural MAE ↓ | decisive sign ↑ | won sign ↑ | won MAE ↓ |
|---|---:|---:|---:|---:|
| Phase-2 | **0.16527** | **66.5%** | **74.5%** | 0.6625 |
| joint epoch 3 | 0.16544 | 66.0% | 71.5% | **0.6536** |

Joint did slightly improve raw policy (mean regret 235.9cp -> 229.7cp; near-best 56.0% -> 57.88%; exact
best 32.25% -> 33.25%) but damaged decisive/won value sign. This explains why its match score did not survive
deeper search.

**Search verdict:** Phase-2 at 64 simulations with `value_scale=0.5` is the only configuration whose mean
regret and near-best bootstrap intervals are both strictly positive. It reduced mean regret by 22.05cp
(95% CI 4.15..40.16), improved near-best by 1.625 points (CI 0.375..2.875), and reduced p90 by 61.4cp.
However p90 improved **9.21%**, narrowly below the predeclared 10% gate, and the absolute value-sign gates
failed badly. `value_scale=1.0` met the p90 threshold but its mean-regret CI crossed zero. No joint search
configuration passed. The fixed mate-in-one sanity solve rate was only 30% for Phase-2 scale 0.5 and 23.3%
for joint scale 0.5. Therefore do **not** consume the test split and do not increase simulations.

An additional Opus 4.8 review challenged the planned repair training on one unmeasured assumption: the model
was trained on depth-14 targets but graded by depth 20. If depth-14/depth-20 target disagreement already
accounts for most of the 0.165 validation MAE or flips decisive signs, balanced sampling/sign loss would train
the model harder on teacher errors. The mandatory next process is therefore a **label-ceiling audit**, not a
repair run: deterministically sample 3,000 cached training positions, re-evaluate them at depth-20 MultiPV-1,
and report bounded-value MAE/sign disagreement overall and per value bin plus the cached-label bin census.

Decision after the audit:
- disagreement >= `0.1653 - 0.02`: depth-14 is label-limited; do not repair on it. Relabel only a balanced
  30-50k subset at depth 20.
- disagreement < `0.10` and won-bin sign disagreement <=8%: cached labels have headroom; run staged value
  repair (head only -> norm -> last block + policy KL).
- otherwise: borderline; permit only a head-only probe, no trunk unfreeze or test consumption.

The earlier 90/95% sign thresholds remain long-term strength targets, not plausible one-step development
gates from a 66% baseline. Before any training and while test is still untouched, the next candidate gate will
be defined as bootstrap-separated improvement over Phase-2 plus the existing regret/calibration floors.
