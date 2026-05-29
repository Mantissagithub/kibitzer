from __future__ import annotations

import torch

from kibitzer.rl_awr import advantage_weights, awr_loss


def test_advantage_weights_respect_mask_and_cap() -> None:
    advantages = torch.tensor([0.0, 10.0, 4.0])
    valid_mask = torch.tensor([True, True, False])

    weights = advantage_weights(
        advantages,
        valid_mask,
        beta=0.5,
        max_weight=3.0,
        normalize=False,
    )

    assert weights[0].item() == 1.0
    assert weights[1].item() == 3.0
    assert weights[2].item() == 0.0


def test_awr_loss_outputs_scalars() -> None:
    logits = torch.zeros(2, 4672)
    legal_mask = torch.zeros(2, 4672, dtype=torch.bool)
    legal_mask[:, :4] = True
    actions = torch.tensor([1, 2])
    advantages = torch.tensor([1.0, -1.0])
    returns = torch.tensor([0.5, -0.5])
    value_pred = torch.tensor([0.1, -0.1])
    valid_mask = torch.tensor([True, True])
    ref_log_probs = torch.log(torch.tensor([0.25, 0.25]))

    out = awr_loss(
        logits=logits,
        actions=actions,
        legal_mask=legal_mask,
        advantages=advantages,
        returns=returns,
        value_pred=value_pred,
        valid_mask=valid_mask,
        beta=0.5,
        max_weight=20.0,
        normalize_weights=True,
        entropy_coef=0.01,
        value_coef=0.5,
        ref_log_probs=ref_log_probs,
        kl_coef=0.02,
    )

    assert out.loss.dim() == 0
    assert out.policy_loss.dim() == 0
    assert out.value_loss.dim() == 0
    assert out.entropy.dim() == 0
    assert out.ref_kl.dim() == 0
    assert out.mean_weight.dim() == 0
    assert out.max_weight.dim() == 0
    assert out.search_policy_loss.dim() == 0
    assert out.search_value_loss.dim() == 0


def test_awr_loss_supports_search_targets() -> None:
    logits = torch.zeros(2, 4672)
    legal_mask = torch.zeros(2, 4672, dtype=torch.bool)
    legal_mask[:, :4] = True
    actions = torch.tensor([1, 2])
    advantages = torch.tensor([1.0, -1.0])
    returns = torch.tensor([0.5, -0.5])
    value_pred = torch.tensor([0.1, -0.1])
    valid_mask = torch.tensor([True, True])
    search_actions = torch.tensor([3, 0])
    has_search_target = torch.tensor([True, False])
    search_value_targets = torch.tensor([0.8, 0.0])

    out = awr_loss(
        logits=logits,
        actions=actions,
        legal_mask=legal_mask,
        advantages=advantages,
        returns=returns,
        value_pred=value_pred,
        valid_mask=valid_mask,
        beta=0.5,
        max_weight=20.0,
        normalize_weights=True,
        entropy_coef=0.01,
        value_coef=0.5,
        search_actions=search_actions,
        has_search_target=has_search_target,
        search_value_targets=search_value_targets,
        search_policy_coef=0.2,
        search_value_coef=0.1,
    )

    assert out.search_policy_loss.item() > 0.0
    assert out.search_value_loss.item() > 0.0
