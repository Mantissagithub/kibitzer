from __future__ import annotations

import torch

from kibitzer.rl_ppo import compute_gae, normalize_advantages, ppo_loss


def test_compute_gae_shapes() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.5]])
    values = torch.tensor([[0.2, 0.1, 0.0]])
    dones = torch.tensor([[False, False, True]])
    adv, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isclose(returns[0, 2], torch.tensor(0.5))


def test_normalize_advantages_respects_mask() -> None:
    advantages = torch.tensor([[1.0, 2.0, 100.0]])
    valid_mask = torch.tensor([[True, True, False]])
    normalized = normalize_advantages(advantages, valid_mask)
    assert normalized[0, 2] == 0.0
    assert abs(float(normalized[0, :2].mean())) < 1e-6


def test_ppo_loss_outputs_scalars() -> None:
    logits = torch.zeros(2, 4672)
    legal_mask = torch.zeros(2, 4672, dtype=torch.bool)
    legal_mask[:, :4] = True
    actions = torch.tensor([1, 2])
    old_log_probs = torch.log(torch.tensor([0.25, 0.25]))
    advantages = torch.tensor([1.0, -1.0])
    returns = torch.tensor([0.5, -0.5])
    value_pred = torch.tensor([0.1, -0.1])
    old_values = torch.tensor([0.0, 0.0])
    valid_mask = torch.tensor([True, True])
    ref_log_probs = old_log_probs.clone()

    out = ppo_loss(
        logits=logits,
        old_log_probs=old_log_probs,
        actions=actions,
        legal_mask=legal_mask,
        advantages=advantages,
        returns=returns,
        value_pred=value_pred,
        old_values=old_values,
        valid_mask=valid_mask,
        clip_eps=0.2,
        value_clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        ref_log_probs=ref_log_probs,
        kl_coef=0.02,
    )
    assert out.loss.dim() == 0
    assert out.policy_loss.dim() == 0
    assert out.value_loss.dim() == 0
    assert out.entropy.dim() == 0
    assert out.approx_kl.dim() == 0
    assert out.ref_kl.dim() == 0
    assert out.clipfrac.dim() == 0
    assert out.search_policy_loss.dim() == 0
    assert out.search_value_loss.dim() == 0


def test_ppo_loss_supports_search_targets() -> None:
    logits = torch.zeros(2, 4672)
    legal_mask = torch.zeros(2, 4672, dtype=torch.bool)
    legal_mask[:, :4] = True
    actions = torch.tensor([1, 2])
    old_log_probs = torch.log(torch.tensor([0.25, 0.25]))
    advantages = torch.tensor([1.0, -1.0])
    returns = torch.tensor([0.5, -0.5])
    value_pred = torch.tensor([0.1, -0.1])
    old_values = torch.tensor([0.0, 0.0])
    valid_mask = torch.tensor([True, True])
    ref_log_probs = old_log_probs.clone()
    search_actions = torch.tensor([3, 0])
    has_search_target = torch.tensor([True, False])
    search_value_targets = torch.tensor([0.8, 0.0])

    out = ppo_loss(
        logits=logits,
        old_log_probs=old_log_probs,
        actions=actions,
        legal_mask=legal_mask,
        advantages=advantages,
        returns=returns,
        value_pred=value_pred,
        old_values=old_values,
        valid_mask=valid_mask,
        clip_eps=0.2,
        value_clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        ref_log_probs=ref_log_probs,
        kl_coef=0.02,
        search_actions=search_actions,
        has_search_target=has_search_target,
        search_value_targets=search_value_targets,
        search_policy_coef=0.2,
        search_value_coef=0.1,
    )
    assert out.search_policy_loss.item() > 0.0
    assert out.search_value_loss.item() > 0.0
