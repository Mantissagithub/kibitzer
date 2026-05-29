"""Advantage-weighted policy improvement losses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from kibitzer.rl_ppo import gather_action_log_probs, masked_log_probs


@dataclass
class AWROutputs:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    ref_kl: Tensor
    mean_weight: Tensor
    max_weight: Tensor
    search_policy_loss: Tensor
    search_value_loss: Tensor


def advantage_weights(
    advantages: Tensor,
    valid_mask: Tensor,
    *,
    beta: float,
    max_weight: float,
    normalize: bool = True,
) -> Tensor:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    raw = torch.exp((advantages / beta).clamp(max=math.log(max_weight)))
    raw = torch.where(valid_mask, raw, torch.zeros_like(raw))
    if normalize:
        denom = raw[valid_mask].mean().clamp(min=1e-6) if valid_mask.any() else 1.0
        raw = raw / denom
    return raw.clamp(max=max_weight)


def awr_loss(
    *,
    logits: Tensor,
    actions: Tensor,
    legal_mask: Tensor,
    advantages: Tensor,
    returns: Tensor,
    value_pred: Tensor,
    valid_mask: Tensor,
    beta: float,
    max_weight: float,
    normalize_weights: bool,
    entropy_coef: float,
    value_coef: float,
    ref_log_probs: Tensor | None = None,
    kl_coef: float = 0.0,
    search_actions: Tensor | None = None,
    has_search_target: Tensor | None = None,
    search_value_targets: Tensor | None = None,
    search_policy_coef: float = 0.0,
    search_value_coef: float = 0.0,
) -> AWROutputs:
    """Compute an AWR/AWAC-style weighted maximum-likelihood update."""
    action_log_probs = gather_action_log_probs(logits, actions, legal_mask)
    weights = advantage_weights(
        advantages.detach(),
        valid_mask,
        beta=beta,
        max_weight=max_weight,
        normalize=normalize_weights,
    )
    policy_loss = -weights * action_log_probs
    value_loss = 0.5 * (value_pred - returns) ** 2

    log_all = masked_log_probs(logits, legal_mask)
    probs = log_all.exp()
    safe_log_all = torch.where(legal_mask, log_all, torch.zeros_like(log_all))
    entropy = -(probs * safe_log_all).sum(dim=-1)

    ref_kl = torch.zeros_like(policy_loss)
    if ref_log_probs is not None and kl_coef > 0.0:
        ref_kl = 0.5 * (action_log_probs - ref_log_probs) ** 2

    search_policy_loss = torch.zeros_like(policy_loss)
    search_value_loss = torch.zeros_like(value_loss)
    if (
        search_actions is not None
        and has_search_target is not None
        and search_policy_coef > 0.0
    ):
        search_mask = valid_mask & has_search_target
        if search_mask.any():
            search_policy_loss = -log_all.gather(
                -1, search_actions.unsqueeze(-1)
            ).squeeze(-1)
            search_policy_loss = torch.where(
                search_mask, search_policy_loss, torch.zeros_like(search_policy_loss)
            )

    if (
        search_value_targets is not None
        and has_search_target is not None
        and search_value_coef > 0.0
    ):
        search_mask = valid_mask & has_search_target
        search_value_loss = 0.5 * (value_pred - search_value_targets) ** 2
        search_value_loss = torch.where(
            search_mask, search_value_loss, torch.zeros_like(search_value_loss)
        )

    mask = valid_mask.float()
    denom = mask.sum().clamp(min=1.0)
    total = (
        (policy_loss * mask).sum()
        + value_coef * (value_loss * mask).sum()
        + kl_coef * (ref_kl * mask).sum()
        + search_policy_coef * (search_policy_loss * mask).sum()
        + search_value_coef * (search_value_loss * mask).sum()
        - entropy_coef * (entropy * mask).sum()
    ) / denom
    valid_weights = weights[valid_mask]
    return AWROutputs(
        loss=total,
        policy_loss=(policy_loss * mask).sum() / denom,
        value_loss=(value_loss * mask).sum() / denom,
        entropy=(entropy * mask).sum() / denom,
        ref_kl=(ref_kl * mask).sum() / denom,
        mean_weight=(
            valid_weights.mean() if valid_weights.numel() else weights.new_tensor(0.0)
        ),
        max_weight=valid_weights.max() if valid_weights.numel() else weights.new_tensor(0.0),
        search_policy_loss=(search_policy_loss * mask).sum() / denom,
        search_value_loss=(search_value_loss * mask).sum() / denom,
    )
