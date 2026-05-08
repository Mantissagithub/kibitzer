"""Tests for kibitzer.loss.policy_loss / value_loss / combined_loss."""

from __future__ import annotations

import math

import torch

from kibitzer.loss import combined_loss, policy_loss, value_loss


A = 4672  # action space size, matching kibitzer.encoding.ACTION_SIZE


def _legal_mask(B: int, T: int, indices_per_pos: list[list[int]]) -> torch.Tensor:
    """Build a (B, T, A) Bool mask. ``indices_per_pos[b*T + t]`` lists legal idxs."""
    mask = torch.zeros(B, T, A, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            for i in indices_per_pos[b * T + t]:
                mask[b, t, i] = True
    return mask


def test_policy_loss_perfect() -> None:
    legal = [10, 20, 30, 40, 50]
    logits = torch.zeros(1, 1, A)
    logits[0, 0, 30] = 50.0  # softmax mass concentrates on idx 30
    legal_mask = _legal_mask(1, 1, [legal])
    move_idx = torch.tensor([[30]])
    loss_mask = torch.ones(1, 1, dtype=torch.bool)

    loss = policy_loss(logits, move_idx, legal_mask, loss_mask)
    assert loss.item() < 1e-10


def test_policy_loss_uniform() -> None:
    legal = list(range(20))
    logits = torch.zeros(1, 1, A)
    legal_mask = _legal_mask(1, 1, [legal])
    move_idx = torch.tensor([[0]])
    loss_mask = torch.ones(1, 1, dtype=torch.bool)

    loss = policy_loss(logits, move_idx, legal_mask, loss_mask)
    assert torch.isclose(loss, torch.tensor(math.log(20.0)), atol=1e-5)


def test_policy_loss_ignores_illegal() -> None:
    legal = [10, 20, 30]
    legal_mask = _legal_mask(1, 1, [legal])
    move_idx = torch.tensor([[20]])
    loss_mask = torch.ones(1, 1, dtype=torch.bool)

    base = torch.zeros(1, 1, A)
    perturbed = base.clone()
    perturbed[0, 0, 5] = 1e6        # illegal — should not affect loss
    perturbed[0, 0, 1000] = -1e6    # illegal — should not affect loss

    loss_a = policy_loss(base, move_idx, legal_mask, loss_mask)
    loss_b = policy_loss(perturbed, move_idx, legal_mask, loss_mask)
    assert torch.isclose(loss_a, loss_b)


def test_policy_loss_respects_loss_mask() -> None:
    # Ply 0: 100 legal moves, uniform → loss = log(100).
    # Ply 1: 1 legal move (the played one) with high logit → loss ≈ 0.
    legal_mask = torch.zeros(1, 2, A, dtype=torch.bool)
    for i in range(100):
        legal_mask[0, 0, i] = True
    legal_mask[0, 1, 42] = True

    logits = torch.zeros(1, 2, A)
    logits[0, 1, 42] = 50.0
    move_idx = torch.tensor([[0, 42]])

    loss_first = policy_loss(
        logits, move_idx, legal_mask, torch.tensor([[True, False]])
    )
    loss_second = policy_loss(
        logits, move_idx, legal_mask, torch.tensor([[False, True]])
    )
    loss_both = policy_loss(
        logits, move_idx, legal_mask, torch.ones(1, 2, dtype=torch.bool)
    )

    assert torch.isclose(loss_first, torch.tensor(math.log(100.0)), atol=1e-5)
    assert loss_second.item() < 1e-10
    assert torch.isclose(loss_both, (loss_first + loss_second) / 2, atol=1e-5)


def test_value_loss_perfect() -> None:
    pred = torch.tensor([[0.5, -0.3, 0.0]])
    target = pred.clone()
    loss_mask = torch.ones(1, 3, dtype=torch.bool)
    assert value_loss(pred, target, loss_mask).item() == 0.0


def test_value_loss_mse() -> None:
    pred = torch.tensor([[0.5, -0.3]])
    target = torch.tensor([[0.0, 0.0]])
    loss_mask = torch.ones(1, 2, dtype=torch.bool)
    expected = (0.5**2 + 0.3**2) / 2  # = 0.17
    assert torch.isclose(
        value_loss(pred, target, loss_mask), torch.tensor(expected), atol=1e-6
    )


def test_combined_loss_keys() -> None:
    B, T = 1, 1
    logits = torch.zeros(B, T, A)
    legal_mask = _legal_mask(B, T, [[10]])
    move_idx = torch.tensor([[10]])
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    out = combined_loss(
        {"policy_logits": logits, "value": torch.zeros(B, T)},
        {
            "move_idx": move_idx,
            "legal_mask": legal_mask,
            "loss_mask": loss_mask,
            "value_target": torch.zeros(B, T),
        },
    )
    assert set(out.keys()) == {"loss", "policy_loss", "value_loss", "policy_acc"}
    for v in out.values():
        assert isinstance(v, torch.Tensor)
        assert v.dim() == 0


def test_policy_accuracy() -> None:
    # Two plies, both real.
    # Ply 0: legal {10, 20, 30}, played 20, max logit at 20 → correct.
    # Ply 1: legal {40, 50, 60}, played 50, max logit at 60 → wrong.
    B, T = 1, 2
    logits = torch.zeros(B, T, A)
    legal_mask = torch.zeros(B, T, A, dtype=torch.bool)
    for i in [10, 20, 30]:
        legal_mask[0, 0, i] = True
    for i in [40, 50, 60]:
        legal_mask[0, 1, i] = True
    logits[0, 0, 20] = 5.0
    logits[0, 1, 60] = 5.0

    move_idx = torch.tensor([[20, 50]])
    loss_mask = torch.ones(B, T, dtype=torch.bool)

    out = combined_loss(
        {"policy_logits": logits, "value": torch.zeros(B, T)},
        {
            "move_idx": move_idx,
            "legal_mask": legal_mask,
            "loss_mask": loss_mask,
            "value_target": torch.zeros(B, T),
        },
    )
    assert torch.isclose(out["policy_acc"], torch.tensor(0.5))
