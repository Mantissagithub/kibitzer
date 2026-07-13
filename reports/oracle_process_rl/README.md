# D66 oracle process-reward RL

## Result

The process-reward experiment did not improve the model. It successfully created dense, signed move-level credit from Stockfish evaluations and combined it with the real game outcome, but neither policy-update epoch improved the group-disjoint held-out RL objective. The selector restored epoch 0.

The final `runs/oracle_process_rl/oracle_process_rl.pt` checkpoint is tensor-identical to `runs/tactical/tactical_repair.pt`. Its maximum absolute parameter difference is `0.0`, across `0` changed tensors. The strongest checkpoint therefore remains the tactical base, with deeper 512-simulation PUCT as the only confirmed positive lever.

![signal funnel](fig1_signal_funnel.png)

## Experiment

The tactical checkpoint played 32 fresh games against Stockfish UCI Elo 2300. Kibitzer used 512-simulation PUCT, opening temperature `0.8` through ply 20, and a maximum of 160 plies. The rollout produced 1,665 model positions and finished 22W/8D/2L, score `0.812`.

Every sampled model move was labeled by full-strength Stockfish at 10,000 nodes with MultiPV 4. The process reward measured the chosen move against Stockfish's best value from the mover's perspective, clipped to `[-1, 1]`. Returns combined `0.25 * process_reward` with the terminal result in `{-1, 0, +1}` at discount `0.99`.

Only positions with regret at least `0.05` and absolute group-relative advantage at least `0.1` were retained. This left 220 of 1,665 positions, 87 with positive advantage and 133 with negative advantage.

![reward distributions](fig2_reward_distributions.png)

## Update

The position encoder, transformer trunk, and value head were frozen. Only the policy head and final normalization, 1,200,960 parameters, were trainable. The update used signed return policy gradients, exact legal-action TV control, and a KL anchor to the frozen tactical base.

The first selector used top-1 Stockfish regret. That metric was too discrete and remained unchanged, so both epochs were rejected. The run was repeated from the same labeled buffer with a smoother held-out selection metric, mean `advantage * log policy(sampled_action)`. Both epoch snapshots were preserved.

| metric | epoch 0 | epoch 1 | epoch 2 | read |
|---|---:|---:|---:|---|
| top-1 regret | 0.079830 | 0.079830 | 0.079830 | unchanged |
| expected teacher regret | 0.0845723 | 0.0845715 | 0.0845709 | negligible improvement |
| held-out signed log-probability | 0.5807979 | 0.5807869 | 0.5807538 | worsened |
| mean TV to base | 0.000000 | 0.000163 | 0.000311 | safely below 0.08 ceiling |

The teacher-regret change is too small to matter, while the actual held-out RL objective moves in the wrong direction. Epoch 0 remains the correct selection.

![training selection](fig3_training_selection.png)

## External gate

The epoch-0 output completed an 80-game seed-31 gate at 128 simulations: 6W/21D/53L, score `0.206`. This is not evidence that RL regressed the model because the gated checkpoint contains no RL update. It is another match sample from the tactical base. The older seed-23 tactical gate scored `0.294`, 12W/23D/45L.

The separate seed-31 tactical-base rerun was interrupted after 29 games and is excluded from the decision. Running further candidate/base or 512-simulation gates would be redundant because the selected checkpoints are identical.

![external identity](fig4_external_identity.png)

## Interpretation

The original credit-assignment hypothesis was reasonable. Sparse terminal-only GRPO could not identify which moves caused the outcome, while this run supplied per-move Stockfish regret and retained the real terminal result. The failure is therefore narrower than the earlier outcome-only result: dense oracle credit was available, but a small policy-only update on the fixed representation still did not generalize across held-out rollout groups.

There are two evidence limits:

- Only 220 positions survived the meaningful-regret filter, split across eight rollout groups. That is a small RL sample.
- Search selected actions from PUCT visits, while the stored `mu` distribution is the raw network prior. The update is best read as oracle-shaped searched-policy repair, not an exact importance-corrected policy gradient for the raw network.
- Moves outside Stockfish's MultiPV top four received a separate dedicated root-move search, while the teacher best retained its MultiPV score. The resulting search-budget mismatch produced some positive process rewards, up to `+0.454`, even though chosen-minus-best should theoretically be non-positive. A clean retry must rescore both moves under identical root-move limits.

At this budget, more coefficient tuning has a weak prior. The result supports the existing fixed-representation ceiling but does not establish that process-reward RL would fail with matched teacher searches, a larger population, exact search-behavior probabilities, more diverse states, or a different value representation.

## Rebuild

```bash
uv run python reports/oracle_process_rl/make_figures.py
```
