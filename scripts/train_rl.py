"""PPO-based RL fine-tuning for kibitzer."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

import torch
import yaml

from kibitzer.model import Kibitzer
from kibitzer.rl_config import RLConfig, RewardMix
from kibitzer.rl_ppo import compute_gae, gather_action_log_probs, normalize_advantages, ppo_loss
from kibitzer.rl_rollout import (
    StockfishAnalyser,
    collect_selfplay_rollout,
    collect_stockfish_rollout,
    load_engine_from_checkpoint,
    pack_rollout_batch,
    sample_prev_checkpoint,
)
from kibitzer.training_utils import get_lr, load_checkpoint, save_checkpoint


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    for name, field in RLConfig.__dataclass_fields__.items():
        if is_dataclass(field.default):
            continue
        parser.add_argument(f"--{name.replace('_', '-')}", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    _add_config_args(parser)
    return parser.parse_args()


def _coerce_value(current, raw):
    if raw is None:
        return current
    if isinstance(current, bool):
        return str(raw).lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [int(x) for x in str(raw).split(",") if x]
    return raw


def load_config(args: argparse.Namespace) -> RLConfig:
    cfg = RLConfig(init_checkpoint="")
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                if key in {"phase1_reward", "phase2_reward"} and isinstance(value, dict):
                    value = RewardMix(**value)
                setattr(cfg, key, value)
    for name in RLConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, _coerce_value(getattr(cfg, name), value))
    if not cfg.init_checkpoint:
        raise ValueError("--init-checkpoint is required")
    return cfg


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def _forward_batch(model: Kibitzer, batch: dict, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    piece_idx = batch["piece_idx"].to(device)
    aux = batch["aux"].to(device)
    pad_mask = ~batch["valid_mask"].to(device)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=dtype in (torch.bfloat16, torch.float16),
    ):
        logits, values = model(piece_idx, aux, pad_mask)
    return logits, values[..., 0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    if args.dry_run:
        cfg.total_updates = min(cfg.total_updates, 1)
        cfg.rollouts_per_batch = min(cfg.rollouts_per_batch, 1)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype(cfg.dtype if device.type == "cuda" else "float32")
    model = Kibitzer()
    ckpt = load_checkpoint(cfg.init_checkpoint, model, map_location="cpu")
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, map_location="cpu")
    model = model.to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr)
    if args.resume and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    reference_model = copy.deepcopy(model).eval()
    target_model = copy.deepcopy(model).eval()
    prev_pool = [cfg.init_checkpoint]
    rng = random.Random(cfg.seed)

    for update in range(cfg.total_updates):
        lr = get_lr(
            update, cfg.warmup_steps, cfg.total_updates, cfg.peak_lr, cfg.min_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        learner_checkpoint = Path(cfg.output_dir) / "latest_actor.pt"
        save_checkpoint(
            learner_checkpoint,
            model,
            optimizer=None,
            scheduler_state=None,
            step=update,
            config=asdict(cfg),
        )

        actor_engine = load_engine_from_checkpoint(
            str(learner_checkpoint),
            device=str(device),
            dtype=dtype,
            context_window=cfg.context_window,
        )
        target_engine = load_engine_from_checkpoint(
            str(learner_checkpoint),
            device=str(device),
            dtype=dtype,
            context_window=cfg.context_window,
        )
        target_engine.model.load_state_dict(target_model.state_dict())

        batches = []
        current_elo = cfg.stockfish_levels[min(len(prev_pool) - 1, len(cfg.stockfish_levels) - 1)]
        with StockfishAnalyser(
            path=cfg.stockfish_path,
            depth=cfg.stockfish_depth,
            uci_elo=current_elo if cfg.phase == "stockfish" else None,
        ) as analyser:
            for _ in range(cfg.rollouts_per_batch):
                if cfg.phase == "stockfish":
                    steps = collect_stockfish_rollout(
                        actor_engine, target_engine, analyser, cfg, stockfish_elo=current_elo
                    )
                    batches.append(pack_rollout_batch(steps, cfg.chunk_len, source="stockfish"))
                else:
                    prev_ckpt = sample_prev_checkpoint(prev_pool[-cfg.prev_pool_size :], rng, cfg.init_checkpoint)
                    opponent_engine = load_engine_from_checkpoint(
                        prev_ckpt,
                        device=str(device),
                        dtype=dtype,
                        context_window=cfg.context_window,
                    )
                    steps = collect_selfplay_rollout(
                        actor_engine, opponent_engine, target_engine, analyser, cfg
                    )
                    batches.append(pack_rollout_batch(steps, cfg.chunk_len, source="selfplay"))

        batch = {
            "piece_idx": torch.cat([x.piece_idx for x in batches], dim=0),
            "aux": torch.cat([x.aux for x in batches], dim=0),
            "actions": torch.cat([x.actions for x in batches], dim=0),
            "legal_mask": torch.cat([x.legal_mask for x in batches], dim=0),
            "old_log_probs": torch.cat([x.old_log_probs for x in batches], dim=0),
            "old_values": torch.cat([x.old_values for x in batches], dim=0),
            "rewards": torch.cat([x.rewards for x in batches], dim=0),
            "dones": torch.cat([x.dones for x in batches], dim=0),
            "valid_mask": torch.cat([x.valid_mask for x in batches], dim=0),
        }

        advantages, returns = compute_gae(
            batch["rewards"],
            batch["old_values"],
            batch["dones"],
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        advantages = normalize_advantages(advantages, batch["valid_mask"])

        optimizer.zero_grad(set_to_none=True)
        final_metrics = None
        for _ in range(cfg.ppo_epochs):
            logits, values = _forward_batch(model, batch, device, dtype)
            with torch.no_grad():
                ref_logits, _ = _forward_batch(reference_model, batch, device, dtype)
            flat_valid = batch["valid_mask"].to(device).reshape(-1)
            flat_logits = logits.reshape(-1, logits.shape[-1])[flat_valid]
            flat_ref_logits = ref_logits.reshape(-1, ref_logits.shape[-1])[flat_valid]
            flat_legal = batch["legal_mask"].to(device).reshape(-1, logits.shape[-1])[flat_valid]
            flat_actions = batch["actions"].to(device).reshape(-1)[flat_valid]
            flat_old_log_probs = batch["old_log_probs"].to(device).reshape(-1)[flat_valid]
            flat_adv = advantages.to(device).reshape(-1)[flat_valid]
            flat_returns = returns.to(device).reshape(-1)[flat_valid]
            flat_values = values.reshape(-1)[flat_valid]
            flat_old_values = batch["old_values"].to(device).reshape(-1)[flat_valid]
            ref_action_log_probs = gather_action_log_probs(
                flat_ref_logits, flat_actions, flat_legal
            )
            metrics = ppo_loss(
                logits=flat_logits,
                old_log_probs=flat_old_log_probs,
                actions=flat_actions,
                legal_mask=flat_legal,
                advantages=flat_adv,
                returns=flat_returns,
                value_pred=flat_values,
                old_values=flat_old_values,
                valid_mask=torch.ones_like(flat_adv, dtype=torch.bool),
                clip_eps=cfg.clip_eps,
                value_clip_eps=cfg.value_clip_eps,
                entropy_coef=cfg.entropy_coef,
                value_coef=cfg.value_coef,
                ref_log_probs=ref_action_log_probs,
                kl_coef=cfg.kl_coef,
            )
            metrics.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            final_metrics = metrics

        if final_metrics is None:
            raise RuntimeError("ppo loop produced no metrics")

        if (update + 1) % cfg.target_sync_interval == 0:
            target_model.load_state_dict(model.state_dict())
        if (update + 1) % cfg.checkpoint_every == 0:
            out_path = Path(cfg.output_dir) / f"rl_step_{update + 1}.pt"
            save_checkpoint(
                out_path,
                model,
                optimizer,
                scheduler_state={"lr": lr},
                step=update + 1,
                config=asdict(cfg),
                metrics={
                    "loss": float(final_metrics.loss.item()),
                    "policy_loss": float(final_metrics.policy_loss.item()),
                    "value_loss": float(final_metrics.value_loss.item()),
                    "entropy": float(final_metrics.entropy.item()),
                    "approx_kl": float(final_metrics.approx_kl.item()),
                    "clipfrac": float(final_metrics.clipfrac.item()),
                    "avg_reward": float(batch["rewards"][batch["valid_mask"]].mean().item()),
                },
            )
            prev_pool.append(str(out_path))

        if (update + 1) % cfg.log_every == 0:
            avg_reward = batch["rewards"][batch["valid_mask"]].mean().item()
            print(
                "update="
                f"{update + 1} phase={cfg.phase} loss={final_metrics.loss.item():.4f} "
                f"policy={final_metrics.policy_loss.item():.4f} "
                f"value={final_metrics.value_loss.item():.4f} "
                f"entropy={final_metrics.entropy.item():.4f} "
                f"kl={final_metrics.approx_kl.item():.4f} "
                f"reward={avg_reward:.4f} lr={lr:.2e}"
            )


if __name__ == "__main__":
    main()
