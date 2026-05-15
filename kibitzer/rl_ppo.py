"""PPO losses and trajectory advantage utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_log_probs(logits: Tensor, legal_mask: Tensor) -> Tensor:
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def gather_action_log_probs(
    logits: Tensor,
    actions: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    log_probs = masked_log_probs(logits, legal_mask)
    return log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """generalized advantage estimation over batched trajectories."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have the same shape")

    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    next_values = torch.zeros_like(rewards)
    next_values[:, :-1] = values[:, 1:]

    for t in range(rewards.shape[1] - 1, -1, -1):
        not_done = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * next_values[:, t] * not_done - values[:, t]
        last_adv = delta + gamma * gae_lambda * not_done * last_adv
        adv[:, t] = last_adv

    returns = adv + values
    return adv, returns


def normalize_advantages(advantages: Tensor, valid_mask: Tensor) -> Tensor:
    valid = advantages[valid_mask]
    if valid.numel() == 0:
        return torch.zeros_like(advantages)
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp(min=1e-6)
    normalized = (advantages - mean) / std
    return torch.where(valid_mask, normalized, torch.zeros_like(normalized))


@dataclass
class PPOOutputs:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clipfrac: Tensor


def ppo_loss(
    *,
    logits: Tensor,
    old_log_probs: Tensor,
    actions: Tensor,
    legal_mask: Tensor,
    advantages: Tensor,
    returns: Tensor,
    value_pred: Tensor,
    old_values: Tensor,
    valid_mask: Tensor,
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    ref_log_probs: Tensor | None = None,
    kl_coef: float = 0.0,
) -> PPOOutputs:
    """compute clipped PPO losses for a legal-masked categorical policy."""
    log_probs = gather_action_log_probs(logits, actions, legal_mask)
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_losses = -advantages * ratio
    pg_losses_clipped = -advantages * clipped_ratio
    policy_loss = torch.maximum(pg_losses, pg_losses_clipped)

    value_delta = torch.clamp(value_pred - old_values, -value_clip_eps, value_clip_eps)
    value_clipped = old_values + value_delta
    value_loss_unclipped = (value_pred - returns) ** 2
    value_loss_clipped = (value_clipped - returns) ** 2
    value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped)

    log_all = masked_log_probs(logits, legal_mask)
    probs = log_all.exp()
    entropy = -(probs * log_all).sum(dim=-1)

    approx_kl = old_log_probs - log_probs
    clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float()

    if ref_log_probs is not None and kl_coef > 0.0:
        approx_kl = approx_kl + kl_coef * (log_probs - ref_log_probs)

    mask = valid_mask.float()
    denom = mask.sum().clamp(min=1.0)
    total = (
        (policy_loss * mask).sum()
        + value_coef * (value_loss * mask).sum()
        - entropy_coef * (entropy * mask).sum()
    ) / denom
    return PPOOutputs(
        loss=total,
        policy_loss=(policy_loss * mask).sum() / denom,
        value_loss=(value_loss * mask).sum() / denom,
        entropy=(entropy * mask).sum() / denom,
        approx_kl=(approx_kl * mask).sum() / denom,
        clipfrac=(clipfrac * mask).sum() / denom,
    )
