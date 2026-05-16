"""PPO-based RL fine-tuning for kibitzer."""

from __future__ import annotations

import argparse
import copy
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import chess
import torch
import yaml

from kibitzer.eval import evaluate_checkpoint
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


def _default_trainer_state(cfg: RLConfig) -> dict:
    return {
        "current_stockfish_level_idx": 0,
        "best_checkpoint": cfg.init_checkpoint,
        "best_eval_score": float("-inf"),
        "latest_checkpoint": cfg.init_checkpoint,
        "prev_pool": [cfg.init_checkpoint],
    }


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    for name, field in RLConfig.__dataclass_fields__.items():
        if name in {"phase1_reward", "phase2_reward"}:
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


def _rollout_dtype(device: torch.device, cfg: RLConfig) -> torch.dtype:
    if device.type == "cuda":
        return _dtype(cfg.dtype)
    return torch.float32


def _rollout_summary(chunks: list[dict]) -> dict[str, float]:
    valid = sum(int(batch["valid_mask"].sum().item()) for batch in chunks)
    reward = sum(
        float(batch["rewards"][batch["valid_mask"]].sum().item()) for batch in chunks
    )
    stockfish = sum(
        float(batch["stockfish_components"][batch["valid_mask"]].sum().item())
        for batch in chunks
    )
    value = sum(
        float(batch["value_components"][batch["valid_mask"]].sum().item())
        for batch in chunks
    )
    terminal = sum(
        float(batch["terminal_components"][batch["valid_mask"]].sum().item())
        for batch in chunks
    )
    denom = max(valid, 1)
    return {
        "avg_reward": reward / denom,
        "avg_stockfish_component": stockfish / denom,
        "avg_value_component": value / denom,
        "avg_terminal_component": terminal / denom,
    }


def _concat_batches(batches: list) -> dict:
    return {
        "piece_idx": torch.cat([x.piece_idx for x in batches], dim=0),
        "aux": torch.cat([x.aux for x in batches], dim=0),
        "actions": torch.cat([x.actions for x in batches], dim=0),
        "legal_mask": torch.cat([x.legal_mask for x in batches], dim=0),
        "old_log_probs": torch.cat([x.old_log_probs for x in batches], dim=0),
        "old_values": torch.cat([x.old_values for x in batches], dim=0),
        "rewards": torch.cat([x.rewards for x in batches], dim=0),
        "dones": torch.cat([x.dones for x in batches], dim=0),
        "valid_mask": torch.cat([x.valid_mask for x in batches], dim=0),
        "has_search_target": torch.cat([x.has_search_target for x in batches], dim=0),
        "search_actions": torch.cat([x.search_actions for x in batches], dim=0),
        "search_value_targets": torch.cat(
            [x.search_value_targets for x in batches], dim=0
        ),
        "stockfish_components": torch.cat(
            [x.stockfish_components for x in batches], dim=0
        ),
        "value_components": torch.cat([x.value_components for x in batches], dim=0),
        "terminal_components": torch.cat(
            [x.terminal_components for x in batches], dim=0
        ),
    }


def _collect_one_rollout(
    cfg: RLConfig,
    checkpoint_path: str,
    target_state_dict: dict,
    *,
    current_elo: int,
    actor_color: bool,
    opponent_checkpoint: str | None,
    opponent_label: str,
) -> list:
    rollout_device = torch.device(cfg.rollout_device)
    rollout_dtype = _rollout_dtype(rollout_device, cfg)
    actor_engine = load_engine_from_checkpoint(
        checkpoint_path,
        device=str(rollout_device),
        dtype=rollout_dtype,
        context_window=cfg.context_window,
    )
    target_engine = load_engine_from_checkpoint(
        checkpoint_path,
        device=str(rollout_device),
        dtype=rollout_dtype,
        context_window=cfg.context_window,
    )
    target_engine.model.load_state_dict(target_state_dict)
    with StockfishAnalyser(
        path=cfg.stockfish_path,
        depth=cfg.stockfish_depth,
        uci_elo=current_elo if cfg.phase == "stockfish" else None,
    ) as analyser:
        if cfg.phase == "stockfish":
            steps = collect_stockfish_rollout(
                actor_engine,
                target_engine,
                analyser,
                cfg,
                stockfish_elo=current_elo,
            )
            return pack_rollout_batch(steps, cfg.chunk_len, source="stockfish")

        if opponent_checkpoint is None:
            raise ValueError("self-play rollout requires opponent_checkpoint")
        opponent_engine = load_engine_from_checkpoint(
            opponent_checkpoint,
            device=str(rollout_device),
            dtype=rollout_dtype,
            context_window=cfg.context_window,
        )
        steps = collect_selfplay_rollout(
            actor_engine,
            opponent_engine,
            target_engine,
            analyser,
            cfg,
            actor_color=actor_color,
            opponent_label=opponent_label,
        )
        return pack_rollout_batch(steps, cfg.chunk_len, source="selfplay")


def _evaluate_policy(
    cfg: RLConfig,
    checkpoint_path: str,
    trainer_state: dict,
) -> dict:
    level_idx = trainer_state["current_stockfish_level_idx"]
    current_elo = cfg.stockfish_levels[level_idx]
    if cfg.phase == "stockfish":
        return evaluate_checkpoint(
            checkpoint_path,
            opponent=f"stockfish-elo-{current_elo}",
            n_games=cfg.eval_n_games,
            time_per_move_ms=cfg.eval_time_per_move_ms,
            stockfish_path=cfg.stockfish_path,
            pgn_dir=cfg.eval_pgn_dir,
        )

    prev_checkpoint = trainer_state["best_checkpoint"]
    sf_eval = evaluate_checkpoint(
        checkpoint_path,
        opponent=f"stockfish-elo-{cfg.stockfish_levels[0]}",
        n_games=cfg.eval_n_games,
        time_per_move_ms=cfg.eval_time_per_move_ms,
        stockfish_path=cfg.stockfish_path,
        pgn_dir=cfg.eval_pgn_dir,
    )
    svp_eval = evaluate_checkpoint(
        checkpoint_path,
        opponent="self-vs-prev",
        prev_checkpoint=prev_checkpoint,
        n_games=cfg.eval_n_games,
        time_per_move_ms=cfg.eval_time_per_move_ms,
        stockfish_path=cfg.stockfish_path,
        pgn_dir=cfg.eval_pgn_dir,
    )
    return {"stockfish": sf_eval, "self_vs_prev": svp_eval}


def _advance_curriculum(cfg: RLConfig, trainer_state: dict, eval_result: dict) -> None:
    if cfg.phase != "stockfish":
        return
    level_idx = trainer_state["current_stockfish_level_idx"]
    if eval_result["score"] >= cfg.stockfish_promotion_score and level_idx < len(cfg.stockfish_levels) - 1:
        trainer_state["current_stockfish_level_idx"] += 1


def _trainer_state_from_checkpoint(cfg: RLConfig, ckpt: dict) -> dict:
    state = _default_trainer_state(cfg)
    saved = (ckpt.get("scheduler_state") or {}).get("trainer_state")
    if isinstance(saved, dict):
        state.update(saved)
    return state


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
    trainer_state = _trainer_state_from_checkpoint(cfg, ckpt)
    prev_pool = list(trainer_state["prev_pool"])
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
            scheduler_state={"trainer_state": trainer_state},
            step=update,
            config=asdict(cfg),
        )
        current_elo = cfg.stockfish_levels[trainer_state["current_stockfish_level_idx"]]
        rollout_futures = []
        target_state_dict = {
            k: v.detach().cpu() for k, v in target_model.state_dict().items()
        }
        with ThreadPoolExecutor(max_workers=max(1, cfg.rollout_workers)) as executor:
            for rollout_idx in range(cfg.rollouts_per_batch):
                actor_color = chess.WHITE if rollout_idx % 2 == 0 else chess.BLACK
                opponent_checkpoint = None
                opponent_label = f"stockfish-{current_elo}"
                if cfg.phase != "stockfish":
                    opponent_checkpoint = sample_prev_checkpoint(
                        prev_pool[-cfg.prev_pool_size :],
                        rng,
                        cfg.init_checkpoint,
                        latest_checkpoint=trainer_state.get("latest_checkpoint"),
                        best_checkpoint=trainer_state.get("best_checkpoint"),
                        latest_weight=cfg.selfplay_latest_weight,
                        best_weight=cfg.selfplay_best_weight,
                        older_weight=cfg.selfplay_older_weight,
                    )
                    opponent_label = Path(opponent_checkpoint).stem
                rollout_futures.append(
                    executor.submit(
                        _collect_one_rollout,
                        cfg,
                        str(learner_checkpoint),
                        target_state_dict,
                        current_elo=current_elo,
                        actor_color=actor_color,
                        opponent_checkpoint=opponent_checkpoint,
                        opponent_label=opponent_label,
                    )
                )
        packed_batches = []
        for future in rollout_futures:
            packed_batches.extend(future.result())
        batch = _concat_batches(packed_batches)
        rollout_metrics = _rollout_summary([batch])

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
        valid_positions = batch["valid_mask"].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        for _ in range(cfg.ppo_epochs):
            with torch.no_grad():
                ref_logits, _ = _forward_batch(reference_model, batch, device, dtype)
            perm = valid_positions[torch.randperm(valid_positions.numel())]
            stop_early = False
            accum = 0
            for start in range(0, perm.numel(), cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                if idx.numel() == 0:
                    continue
                logits, values = _forward_batch(model, batch, device, dtype)
                flat_logits_all = logits.reshape(-1, logits.shape[-1])
                flat_values_all = values.reshape(-1)
                flat_ref_logits_all = ref_logits.reshape(-1, ref_logits.shape[-1])
                flat_legal_all = batch["legal_mask"].to(device).reshape(-1, logits.shape[-1])
                flat_actions_all = batch["actions"].to(device).reshape(-1)
                flat_old_log_probs_all = batch["old_log_probs"].to(device).reshape(-1)
                flat_adv_all = advantages.to(device).reshape(-1)
                flat_returns_all = returns.to(device).reshape(-1)
                flat_old_values_all = batch["old_values"].to(device).reshape(-1)
                flat_has_search_all = batch["has_search_target"].to(device).reshape(-1)
                flat_search_actions_all = batch["search_actions"].to(device).reshape(-1)
                flat_search_value_targets_all = batch["search_value_targets"].to(device).reshape(-1)

                flat_logits = flat_logits_all[idx]
                flat_ref_logits = flat_ref_logits_all[idx]
                flat_legal = flat_legal_all[idx]
                flat_actions = flat_actions_all[idx]
                flat_old_log_probs = flat_old_log_probs_all[idx]
                flat_adv = flat_adv_all[idx]
                flat_returns = flat_returns_all[idx]
                flat_values = flat_values_all[idx]
                flat_old_values = flat_old_values_all[idx]
                flat_has_search = flat_has_search_all[idx]
                flat_search_actions = flat_search_actions_all[idx]
                flat_search_value_targets = flat_search_value_targets_all[idx]

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
                    search_actions=flat_search_actions,
                    has_search_target=flat_has_search,
                    search_value_targets=flat_search_value_targets,
                    search_policy_coef=cfg.search_policy_coef if cfg.phase == "selfplay" else 0.0,
                    search_value_coef=cfg.search_value_coef if cfg.phase == "selfplay" else 0.0,
                )
                (metrics.loss / cfg.grad_accum_steps).backward()
                accum += 1
                final_metrics = metrics
                if accum % cfg.grad_accum_steps == 0 or start + cfg.minibatch_size >= perm.numel():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if float(metrics.approx_kl.item()) > cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        if final_metrics is None:
            raise RuntimeError("ppo loop produced no metrics")

        if (update + 1) % cfg.target_sync_interval == 0:
            target_model.load_state_dict(model.state_dict())
        best_eval_improved = False
        eval_metrics = None
        if (update + 1) % cfg.eval_every_updates == 0:
            eval_metrics = _evaluate_policy(cfg, str(learner_checkpoint), trainer_state)
            if cfg.phase == "stockfish":
                _advance_curriculum(cfg, trainer_state, eval_metrics)
                if eval_metrics["score"] > trainer_state["best_eval_score"]:
                    trainer_state["best_eval_score"] = eval_metrics["score"]
                    best_eval_improved = True
            else:
                sf_score = eval_metrics["stockfish"]["score"]
                if sf_score > trainer_state["best_eval_score"]:
                    trainer_state["best_eval_score"] = sf_score
                    best_eval_improved = True
        if (update + 1) % cfg.checkpoint_every == 0:
            out_path = Path(cfg.output_dir) / f"rl_step_{update + 1}.pt"
            trainer_state["latest_checkpoint"] = str(out_path)
            if best_eval_improved:
                trainer_state["best_checkpoint"] = str(out_path)
            prev_pool.append(str(out_path))
            trainer_state["prev_pool"] = prev_pool[-max(cfg.prev_pool_size * 4, 8) :]
            save_checkpoint(
                out_path,
                model,
                optimizer,
                scheduler_state={"lr": lr, "trainer_state": trainer_state},
                step=update + 1,
                config=asdict(cfg),
                metrics={
                    "loss": float(final_metrics.loss.item()),
                    "policy_loss": float(final_metrics.policy_loss.item()),
                    "value_loss": float(final_metrics.value_loss.item()),
                    "entropy": float(final_metrics.entropy.item()),
                    "approx_kl": float(final_metrics.approx_kl.item()),
                    "ref_kl": float(final_metrics.ref_kl.item()),
                    "clipfrac": float(final_metrics.clipfrac.item()),
                    "search_policy_loss": float(final_metrics.search_policy_loss.item()),
                    "search_value_loss": float(final_metrics.search_value_loss.item()),
                    "avg_reward": rollout_metrics["avg_reward"],
                    "avg_stockfish_component": rollout_metrics["avg_stockfish_component"],
                    "avg_value_component": rollout_metrics["avg_value_component"],
                    "avg_terminal_component": rollout_metrics["avg_terminal_component"],
                    "curriculum_level": current_elo,
                    "eval": eval_metrics,
                },
            )
        elif best_eval_improved:
            best_path = Path(cfg.output_dir) / "best_rl.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer=None,
                scheduler_state={"trainer_state": trainer_state},
                step=update + 1,
                config=asdict(cfg),
                metrics={"eval": eval_metrics},
            )
            trainer_state["best_checkpoint"] = str(best_path)

        if (update + 1) % cfg.log_every == 0:
            eval_note = ""
            if eval_metrics is not None:
                if cfg.phase == "stockfish":
                    eval_note = f" eval_score={eval_metrics['score']:.2f}"
                else:
                    eval_note = (
                        f" sf_score={eval_metrics['stockfish']['score']:.2f}"
                        f" svp_score={eval_metrics['self_vs_prev']['score']:.2f}"
                    )
            print(
                "update="
                f"{update + 1} phase={cfg.phase} loss={final_metrics.loss.item():.4f} "
                f"policy={final_metrics.policy_loss.item():.4f} "
                f"value={final_metrics.value_loss.item():.4f} "
                f"entropy={final_metrics.entropy.item():.4f} "
                f"kl={final_metrics.approx_kl.item():.4f} "
                f"ref_kl={final_metrics.ref_kl.item():.4f} "
                f"reward={rollout_metrics['avg_reward']:.4f} "
                f"sf={rollout_metrics['avg_stockfish_component']:.4f} "
                f"vm={rollout_metrics['avg_value_component']:.4f} "
                f"term={rollout_metrics['avg_terminal_component']:.4f} "
                f"curriculum={current_elo} lr={lr:.2e}{eval_note}"
            )


if __name__ == "__main__":
    main()
