# reward and update math for stockfish-shaped on-policy rl. the terminal result
# stays in the return, while stockfish regret tells us which move caused damage.

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from kibitzer.grpo import dppo_mask, exact_tv
from kibitzer.model import Kibitzer


def signed_outcome(reward: float) -> float:
    return 2.0 * float(reward) - 1.0


def clipped_process_reward(best_value: float, chosen_value: float, clip: float) -> float:
    if clip <= 0.0:
        raise ValueError("process reward clip must be positive")
    delta = float(chosen_value) - float(best_value)
    return max(-1.0, min(1.0, delta / clip))


def add_returns_and_advantages(
    records: list[dict],
    *,
    gamma: float,
    process_weight: float,
    terminal_weight: float,
    eps: float = 1e-4,
) -> list[dict]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    out = [dict(record) for record in records]
    by_game: dict[int, list[dict]] = defaultdict(list)
    for record in out:
        by_game[int(record["game_id"])].append(record)

    for game_records in by_game.values():
        game_records.sort(key=lambda record: int(record["model_ply"]))
        terminal = float(game_records[0]["terminal_reward"])
        running = terminal_weight * terminal
        for record in reversed(game_records):
            running = process_weight * float(record["process_reward"]) + gamma * running
            record["return"] = running

    # games in one rollout group share the opening, color, and opponent. compare
    # the same model decision number so late-game returns do not baseline openings.
    by_slot: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in out:
        key = (int(record["group_id"]), int(record["model_ply"]))
        by_slot[key].append(record)
    for slot_records in by_slot.values():
        values = torch.tensor([float(record["return"]) for record in slot_records])
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        for record in slot_records:
            record["advantage"] = (float(record["return"]) - mean) / (std + eps)
    return out


def filter_training_records(
    records: list[dict],
    *,
    min_regret: float,
    min_abs_advantage: float,
) -> list[dict]:
    return [
        record
        for record in records
        if float(record["regret"]) >= min_regret
        and abs(float(record["advantage"])) >= min_abs_advantage
    ]


def configure_policy_scope(model: Kibitzer) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.policy_head, model.norm):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def oracle_policy_loss(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    legal: torch.Tensor,
    rollout_policy: torch.Tensor,
    action: torch.Tensor,
    advantage: torch.Tensor,
    *,
    delta: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.masked_fill(~legal, -1e9)
    base_logits = base_logits.masked_fill(~legal, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    base_logp = F.log_softmax(base_logits, dim=-1)
    policy = logp.exp()
    base_policy = base_logp.exp()

    policy_action = policy.gather(1, action.unsqueeze(1)).squeeze(1)
    rollout_action = rollout_policy.gather(1, action.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
    ratio = policy_action / rollout_action
    tv_rollout = exact_tv(policy, rollout_policy, legal)
    keep = dppo_mask(advantage, ratio, tv_rollout, delta)
    policy_gradient = -(keep * ratio * advantage).sum() / keep.sum().clamp_min(1.0)
    anchor_kl = ((policy * (logp - base_logp)) * legal).sum(dim=-1).mean()
    loss = policy_gradient + beta * anchor_kl

    entropy = -((policy * logp) * legal).sum(dim=-1).mean()
    tv_base = exact_tv(policy.detach(), base_policy, legal).mean()
    return loss, {
        "loss": loss.detach(),
        "policy_gradient": policy_gradient.detach(),
        "anchor_kl": anchor_kl.detach(),
        "tv_rollout": tv_rollout.mean().detach(),
        "tv_base": tv_base.detach(),
        "keep_rate": keep.mean().detach(),
        "entropy": entropy.detach(),
        "ratio_mean": ratio.mean().detach(),
    }
