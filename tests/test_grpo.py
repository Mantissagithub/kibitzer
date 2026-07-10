from __future__ import annotations

import torch

from kibitzer.grpo import dppo_mask, exact_tv, group_zscore


def test_group_zscore_centers_within_group() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = group_zscore(rewards, groups)
    # each group has mean 0.5, so winners are positive and losers negative, and
    # the per-group advantages sum to zero.
    assert adv[0] > 0 and adv[1] < 0
    assert torch.allclose(adv[:2].sum(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(adv[2:].sum(), torch.tensor(0.0), atol=1e-5)


def test_group_zscore_degenerate_group_is_zero() -> None:
    # a group where every game had the same result carries no learning signal.
    rewards = torch.tensor([1.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 0])
    adv = group_zscore(rewards, groups)
    assert torch.allclose(adv, torch.zeros(3), atol=1e-4)


def test_exact_tv_matches_hand_computation() -> None:
    # two legal moves: p=[0.7,0.3], q=[0.4,0.6]; tv = 0.5*(0.3+0.3)=0.3.
    p = torch.tensor([[0.7, 0.3, 0.0]])
    q = torch.tensor([[0.4, 0.6, 0.0]])
    legal = torch.tensor([[True, True, False]])
    assert torch.allclose(exact_tv(p, q, legal), torch.tensor([0.3]), atol=1e-6)


def test_exact_tv_ignores_illegal_mass() -> None:
    p = torch.tensor([[0.5, 0.5, 0.0]])
    q = torch.tensor([[0.5, 0.5, 0.9]])  # illegal slot differs but is masked out
    legal = torch.tensor([[True, True, False]])
    assert torch.allclose(exact_tv(p, q, legal), torch.tensor([0.0]), atol=1e-6)


def test_dppo_mask_truth_table() -> None:
    # four quadrants at div beyond delta: only "push further in reward direction"
    # is blocked. A>0&r>1 and A<0&r<1 block; the other two allow.
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    ratio = torch.tensor([1.5, 0.5, 0.5, 1.5])
    div = torch.tensor([0.9, 0.9, 0.9, 0.9])
    mask = dppo_mask(adv, ratio, div, delta=0.2)
    assert torch.equal(mask, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_dppo_mask_allows_inside_trust_region() -> None:
    # even a reward-direction push is allowed while divergence is under delta.
    adv = torch.tensor([1.0])
    ratio = torch.tensor([2.0])
    div = torch.tensor([0.1])
    assert torch.equal(dppo_mask(adv, ratio, div, delta=0.2), torch.tensor([1.0]))
