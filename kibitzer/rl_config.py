"""configuration for PPO-based reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardMix:
    """weights for one RL phase's reward mix."""

    stockfish_delta_weight: float = 0.25
    value_delta_weight: float = 0.0
    terminal_weight: float = 1.0


@dataclass
class RLConfig:
    """single-node PPO configuration for chess RL."""

    init_checkpoint: str
    output_dir: str = "runs_rl"
    device: str = "cuda"
    rollout_device: str = "cpu"
    dtype: str = "bfloat16"
    seed: int = 42

    phase: str = "stockfish"
    context_window: int = 128
    chunk_len: int = 96
    max_plies: int = 200
    rollouts_per_batch: int = 8
    rollout_workers: int = 4
    temperature: float = 0.8
    top_k: int = 8

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    ppo_epochs: int = 2
    minibatch_size: int = 256
    grad_accum_steps: int = 4
    peak_lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_steps: int = 50
    total_updates: int = 1000
    target_kl: float = 0.02
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0

    stockfish_path: str = "stockfish"
    stockfish_depth: int = 1
    stockfish_levels: list[int] = field(
        default_factory=lambda: [1320, 1500, 1800]
    )
    stockfish_promotion_score: float = 0.6

    search_fraction: float = 0.2
    search_top_k: int = 8
    search_stockfish_weight: float = 0.7
    search_value_weight: float = 0.3
    search_policy_coef: float = 0.2
    search_value_coef: float = 0.1
    prev_pool_size: int = 3
    target_sync_interval: int = 10
    selfplay_latest_weight: float = 0.5
    selfplay_best_weight: float = 0.3
    selfplay_older_weight: float = 0.2

    phase1_reward: RewardMix = field(default_factory=RewardMix)
    phase2_reward: RewardMix = field(
        default_factory=lambda: RewardMix(
            stockfish_delta_weight=0.15,
            value_delta_weight=0.10,
            terminal_weight=1.0,
        )
    )

    log_every: int = 1
    checkpoint_every: int = 20
    eval_every_updates: int = 20
    eval_n_games: int = 10
    eval_time_per_move_ms: int = 200
    eval_pgn_dir: str = "eval_pgns_rl"

    hf_push: bool = False
    hf_private: bool = False
    hf_repo_prefix: str = "kibitzer-rl"
