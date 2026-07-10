# pure math for the grpo + dppo update. kept critic-free on purpose: the group
# baseline replaces the value head (the proven-weak lever, D52), and the dppo
# divergence mask replaces ppo ratio clipping with a real distribution-shift check.

from __future__ import annotations

import torch


# group-relative advantage (grpo): z-score the game rewards within each group so
# the group mean is the baseline. a degenerate group (all games same result ->
# std 0) yields zero advantage and contributes no gradient, which is exactly what
# we want -- an uninformative group teaches nothing.
def group_zscore(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    for gid in torch.unique(group_ids):
        m = group_ids == gid
        r = rewards[m].float()
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv


# exact total-variation distance between the current policy p and the rollout
# policy q, over the legal moves only. chess has ~30 legal moves per position so
# this full-distribution tv is cheap -- no need for the paper's binary/top-k
# approximation. p and q are dense over the action space, legal is a bool mask.
def exact_tv(p: torch.Tensor, q: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((p - q).abs() * legal).sum(dim=-1)


# dppo asymmetric mask: block an update only when it pushes the sampled action
# further in the reward-relevant direction AND the distribution has already moved
# past the trust radius delta. moves back toward the rollout policy are always
# allowed. this preserves ppo's useful asymmetry but keys off actual policy
# divergence instead of the noisy single-token ratio.
def dppo_mask(adv: torch.Tensor, ratio: torch.Tensor, div: torch.Tensor, delta: float) -> torch.Tensor:
    push_up = (adv > 0) & (ratio > 1.0) & (div > delta)
    push_down = (adv < 0) & (ratio < 1.0) & (div > delta)
    return (~(push_up | push_down)).float()
