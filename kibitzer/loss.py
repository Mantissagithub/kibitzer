"""Policy + value losses for Kibitzer training.

Three public callables:

* :func:`policy_loss` — categorical cross-entropy on the played move, with the
  softmax restricted to legal moves via a -inf mask on illegal logits.
* :func:`value_loss`  — MSE between the value head and the per-ply target.
* :func:`combined_loss` — convenience wrapper that returns both losses, a
  weighted total, and a top-1 policy accuracy diagnostic, all keyed off a
  ``model_output`` / ``batch`` pair (so ``scripts/train.py`` can call this
  directly on a forward-pass result and the dataset's collate).

Mask handling notes:

* Illegal logits get ``-inf`` before ``log_softmax``. After the softmax this
  pushes their probability to exactly 0; gradients to those logits are zero
  too (the ``-inf`` is a constant, not a function of the parameter).
* A row with NO legal moves would make ``log_softmax`` produce NaN. We
  defensively zero-out such rows' loss contributions before averaging — the
  data pipeline shouldn't emit them, but this keeps a stray NaN from
  poisoning the whole step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    policy_logits: Tensor,
    move_idx: Tensor,
    legal_mask: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """Masked CE: -log P(played | legal moves), averaged over real plies.

    Shapes: ``policy_logits`` (B, T, A), ``move_idx`` (B, T), ``legal_mask``
    (B, T, A), ``loss_mask`` (B, T). Returns a 0-dim tensor.
    """
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # log P at the played action: gather along the action dim → (B, T).
    target_lp = log_probs.gather(-1, move_idx.unsqueeze(-1)).squeeze(-1)
    per_pos = -target_lp

    # Drop padded plies AND any defensive all-illegal rows (latter would have
    # produced NaN above). torch.where replaces the NaN-or-junk path before
    # the sum, so a NaN in a masked-out row can't propagate.
    has_legal = legal_mask.any(dim=-1)
    effective = loss_mask & has_legal
    per_pos = torch.where(effective, per_pos, torch.zeros_like(per_pos))

    denom = effective.float().sum().clamp(min=1.0)
    return per_pos.sum() / denom


def value_loss(
    value_pred: Tensor,
    value_target: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    """Masked MSE between predicted value (tanh in [-1, 1]) and the target."""
    sq = (value_pred - value_target) ** 2
    valid = loss_mask.float()
    denom = valid.sum().clamp(min=1.0)
    return (sq * valid).sum() / denom


def combined_loss(
    model_output: dict,
    batch: dict,
    value_weight: float = 1.0,
) -> dict:
    """Run both losses + top-1 policy accuracy on one (model_output, batch) pair.

    ``model_output`` keys: ``policy_logits`` (B, T, A), ``value`` (B, T).
    ``batch`` keys: ``move_idx``, ``legal_mask``, ``loss_mask``, ``value_target``.
    """
    logits = model_output["policy_logits"]
    move_idx = batch["move_idx"]
    legal_mask = batch["legal_mask"]
    loss_mask = batch["loss_mask"]

    p = policy_loss(logits, move_idx, legal_mask, loss_mask)
    v = value_loss(model_output["value"], batch["value_target"], loss_mask)
    total = p + value_weight * v

    masked = logits.masked_fill(~legal_mask, float("-inf"))
    pred = masked.argmax(dim=-1)
    correct = (pred == move_idx) & loss_mask
    acc_denom = loss_mask.float().sum().clamp(min=1.0)
    acc = correct.float().sum() / acc_denom

    return {
        "loss": total,
        "policy_loss": p,
        "value_loss": v,
        "policy_acc": acc,
    }
