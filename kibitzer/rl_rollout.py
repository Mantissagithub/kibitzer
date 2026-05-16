"""rollout collection for stockfish play and self-play."""

from __future__ import annotations

import random
import shutil
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

from kibitzer.encoding import board_to_tensor, move_to_index
from kibitzer.inference import KibitzerEngine
from kibitzer.model import Kibitzer
from kibitzer.rl_config import RLConfig, RewardMix
from kibitzer.rl_reward import mix_rewards, score_to_scalar, terminal_reward
from kibitzer.training_utils import load_checkpoint

TrajectorySource = Literal["stockfish", "selfplay"]


@dataclass
class RolloutStep:
    piece_idx: torch.Tensor
    aux: torch.Tensor
    action: int
    legal_mask: torch.Tensor
    old_log_prob: float
    value_pred: float
    reward: float
    done: bool
    color: bool
    source: TrajectorySource
    opponent_label: str
    stockfish_before: float
    stockfish_after: float
    value_before: float
    value_after: float
    terminal_component: float = 0.0
    stockfish_component: float = 0.0
    value_component: float = 0.0
    has_search_target: bool = False
    search_action: int = 0
    search_value_target: float = 0.0


@dataclass
class TrajectoryBatch:
    piece_idx: torch.Tensor
    aux: torch.Tensor
    actions: torch.Tensor
    legal_mask: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    valid_mask: torch.Tensor
    has_search_target: torch.Tensor
    search_actions: torch.Tensor
    search_value_targets: torch.Tensor
    stockfish_components: torch.Tensor
    value_components: torch.Tensor
    terminal_components: torch.Tensor
    source: str
    opponent_label: str


@dataclass
class PackedRollout:
    chunks: list[TrajectoryBatch]
    total_reward: float
    stockfish_component: float
    value_component: float
    terminal_component: float
    n_steps: int


def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def load_engine_from_checkpoint(
    checkpoint_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    context_window: int,
) -> KibitzerEngine:
    model = Kibitzer()
    load_checkpoint(checkpoint_path, model, map_location="cpu")
    return KibitzerEngine(
        model, device=device, dtype=dtype, context_window=context_window
    )


class StockfishAnalyser(AbstractContextManager["StockfishAnalyser"]):
    """small helper for move choice and evaluation from Stockfish."""

    def __init__(
        self,
        *,
        path: str,
        depth: int = 1,
        uci_elo: int | None = None,
    ) -> None:
        resolved = shutil.which(path) if path == "stockfish" else path
        if resolved is None:
            raise FileNotFoundError("stockfish binary not found")
        self.path = resolved
        self.depth = depth
        self.uci_elo = uci_elo
        self._engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "StockfishAnalyser":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        if self.uci_elo is not None:
            self._engine.configure(
                {"UCI_LimitStrength": True, "UCI_Elo": self.uci_elo}
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def _limit(self) -> chess.engine.Limit:
        return chess.engine.Limit(depth=self.depth)

    def play(self, board: chess.Board) -> chess.Move:
        if self._engine is None:
            raise RuntimeError("StockfishAnalyser must be used as a context manager")
        result = self._engine.play(board, self._limit())
        if result.move is None:
            raise RuntimeError(f"stockfish returned no move at FEN {board.fen()}")
        return result.move

    def evaluate(self, board: chess.Board, acting_color: bool) -> float:
        if self._engine is None:
            raise RuntimeError("StockfishAnalyser must be used as a context manager")
        info = self._engine.analyse(board, self._limit())
        score = info["score"]
        return score_to_scalar(score, acting_color=acting_color)


def _step_payload(
    engine: KibitzerEngine,
    board: chess.Board,
    *,
    temperature: float,
    top_k: int | None,
) -> dict:
    evals = engine.evaluate_boards([board])[0]
    policy = torch.from_numpy(evals["policy"])
    if temperature == 0.0:
        move = evals["move_probs"][0][0]
    else:
        scaled = torch.log(policy.clamp(min=1e-20)) / temperature
        if top_k is not None:
            top_k = min(top_k, len(evals["move_probs"]))
            top_vals, top_idx = torch.topk(scaled, top_k)
            kept = torch.full_like(scaled, float("-inf"))
            kept[top_idx] = top_vals
            scaled = kept
        probs = F.softmax(scaled, dim=-1)
        move_idx = int(torch.multinomial(probs, num_samples=1).item())
        move = None
        for legal_move in evals["legal_moves"]:
            idx = move_to_index(legal_move, board)
            if idx == move_idx:
                move = legal_move
                break
        if move is None:
            move = evals["move_probs"][0][0]

    action = move_to_index(move, board)
    return {
        "move": move,
        "action": action,
        "piece_idx": board_to_tensor(board)["piece_idx"],
        "aux": board_to_tensor(board)["aux"],
        "legal_mask": torch.from_numpy(
            np.asarray(evals["policy"] > 0, dtype=np.bool_)
        ),
        "old_log_prob": float(torch.log(policy[action].clamp(min=1e-20)).item()),
        "value_pred": float(evals["value"]),
    }


def _search_target(
    board: chess.Board,
    engine: KibitzerEngine,
    target_engine: KibitzerEngine,
    analyser: StockfishAnalyser,
    cfg: RLConfig,
    acting_color: bool,
) -> tuple[int, float]:
    evals = engine.evaluate_boards([board])[0]
    candidates = evals["move_probs"][: cfg.search_top_k]
    if not candidates:
        raise RuntimeError("expected at least one legal move for search target")

    best_action = move_to_index(candidates[0][0], board)
    best_score = float("-inf")
    for move, _prob in candidates:
        child = board.copy(stack=False)
        child.push(move)
        sf = analyser.evaluate(child, acting_color=acting_color)
        vm = -float(target_engine.evaluate_boards([child])[0]["value"])
        score = cfg.search_stockfish_weight * sf + cfg.search_value_weight * vm
        if score > best_score:
            best_score = score
            best_action = move_to_index(move, board)
    return best_action, best_score


def _reward_mix_for_phase(cfg: RLConfig) -> RewardMix:
    return cfg.phase1_reward if cfg.phase == "stockfish" else cfg.phase2_reward


def collect_stockfish_rollout(
    engine: KibitzerEngine,
    target_engine: KibitzerEngine,
    analyser: StockfishAnalyser,
    cfg: RLConfig,
    *,
    stockfish_elo: int,
) -> list[RolloutStep]:
    board = chess.Board()
    steps: list[RolloutStep] = []
    reward_mix = _reward_mix_for_phase(cfg)

    while not board.is_game_over() and len(steps) < cfg.max_plies:
        if board.turn == chess.WHITE:
            payload = _step_payload(
                engine, board, temperature=cfg.temperature, top_k=cfg.top_k
            )
            sf_before = analyser.evaluate(board, acting_color=board.turn)
            value_before = payload["value_pred"]
            board.push(payload["move"])
            sf_after = analyser.evaluate(board, acting_color=chess.BLACK)
            value_after = float(target_engine.evaluate_boards([board])[0]["value"])
            done = board.is_game_over()
            outcome = terminal_reward(board, acting_color=chess.WHITE) if done else 0.0
            reward = mix_rewards(
                reward_mix,
                stockfish_before=sf_before,
                stockfish_after=-sf_after,
                value_before=value_before,
                value_after=-value_after,
                terminal=outcome,
            )
            steps.append(
                RolloutStep(
                    piece_idx=payload["piece_idx"],
                    aux=payload["aux"],
                    action=payload["action"],
                    legal_mask=payload["legal_mask"],
                    old_log_prob=payload["old_log_prob"],
                    value_pred=value_before,
                    reward=reward.total,
                    done=done,
                    color=chess.WHITE,
                    source="stockfish",
                    opponent_label=f"stockfish-{stockfish_elo}",
                    stockfish_before=sf_before,
                    stockfish_after=-sf_after,
                    value_before=value_before,
                    value_after=-value_after,
                    terminal_component=outcome,
                    stockfish_component=reward.stockfish_delta,
                    value_component=reward.value_delta,
                )
            )
        else:
            board.push(analyser.play(board))

    del stockfish_elo  # kept for metadata symmetry / future curricula
    return steps


def collect_selfplay_rollout(
    actor_engine: KibitzerEngine,
    opponent_engine: KibitzerEngine,
    target_engine: KibitzerEngine,
    analyser: StockfishAnalyser,
    cfg: RLConfig,
    *,
    actor_color: bool,
    opponent_label: str,
) -> list[RolloutStep]:
    board = chess.Board()
    steps: list[RolloutStep] = []
    reward_mix = _reward_mix_for_phase(cfg)

    while not board.is_game_over() and len(steps) < cfg.max_plies:
        current_engine = actor_engine if board.turn == actor_color else opponent_engine
        payload = _step_payload(
            current_engine, board, temperature=cfg.temperature, top_k=cfg.top_k
        )
        acting_color = board.turn
        sf_before = analyser.evaluate(board, acting_color=acting_color)
        value_before = payload["value_pred"]
        has_search_target = False
        search_action = 0
        search_value_target = 0.0
        if acting_color == actor_color and random.random() < cfg.search_fraction:
            has_search_target = True
            search_action, search_value_target = _search_target(
                board,
                actor_engine,
                target_engine,
                analyser,
                cfg,
                acting_color=acting_color,
            )
        board.push(payload["move"])
        sf_after = analyser.evaluate(board, acting_color=not acting_color)
        next_eval = target_engine.evaluate_boards([board])[0]["value"]
        done = board.is_game_over()
        terminal = terminal_reward(board, acting_color=acting_color) if done else 0.0
        reward = mix_rewards(
            reward_mix,
            stockfish_before=sf_before,
            stockfish_after=-sf_after,
            value_before=value_before,
            value_after=-float(next_eval),
            terminal=terminal,
        )
        if acting_color == actor_color:
            steps.append(
                RolloutStep(
                    piece_idx=payload["piece_idx"],
                    aux=payload["aux"],
                    action=payload["action"],
                    legal_mask=payload["legal_mask"],
                    old_log_prob=payload["old_log_prob"],
                    value_pred=value_before,
                    reward=reward.total,
                    done=done,
                    color=acting_color,
                    source="selfplay",
                    opponent_label=opponent_label,
                    stockfish_before=sf_before,
                    stockfish_after=-sf_after,
                    value_before=value_before,
                    value_after=-float(next_eval),
                    terminal_component=terminal,
                    stockfish_component=reward.stockfish_delta,
                    value_component=reward.value_delta,
                    has_search_target=has_search_target,
                    search_action=search_action,
                    search_value_target=search_value_target,
                )
            )
    return steps


def pack_rollout_batch(
    steps: list[RolloutStep], chunk_len: int, source: str
) -> list[TrajectoryBatch]:
    """split a rollout into padded PPO chunks."""
    if not steps:
        raise ValueError("steps must be non-empty")
    chunks: list[TrajectoryBatch] = []
    opponent_label = steps[0].opponent_label
    for start in range(0, len(steps), chunk_len):
        chunk_steps = steps[start : start + chunk_len]
        piece_idx = torch.zeros(1, chunk_len, 64, dtype=torch.long)
        aux = torch.zeros(1, chunk_len, 7, dtype=torch.float32)
        actions = torch.zeros(1, chunk_len, dtype=torch.long)
        legal_mask = torch.zeros(1, chunk_len, 4672, dtype=torch.bool)
        old_log_probs = torch.zeros(1, chunk_len, dtype=torch.float32)
        old_values = torch.zeros(1, chunk_len, dtype=torch.float32)
        rewards = torch.zeros(1, chunk_len, dtype=torch.float32)
        dones = torch.ones(1, chunk_len, dtype=torch.bool)
        valid_mask = torch.zeros(1, chunk_len, dtype=torch.bool)
        has_search_target = torch.zeros(1, chunk_len, dtype=torch.bool)
        search_actions = torch.zeros(1, chunk_len, dtype=torch.long)
        search_value_targets = torch.zeros(1, chunk_len, dtype=torch.float32)
        stockfish_components = torch.zeros(1, chunk_len, dtype=torch.float32)
        value_components = torch.zeros(1, chunk_len, dtype=torch.float32)
        terminal_components = torch.zeros(1, chunk_len, dtype=torch.float32)

        for idx, step in enumerate(chunk_steps):
            piece_idx[0, idx] = step.piece_idx
            aux[0, idx] = step.aux
            actions[0, idx] = step.action
            legal_mask[0, idx] = step.legal_mask
            old_log_probs[0, idx] = step.old_log_prob
            old_values[0, idx] = step.value_pred
            rewards[0, idx] = step.reward
            dones[0, idx] = step.done if idx == len(chunk_steps) - 1 else False
            valid_mask[0, idx] = True
            has_search_target[0, idx] = step.has_search_target
            search_actions[0, idx] = step.search_action
            search_value_targets[0, idx] = step.search_value_target
            stockfish_components[0, idx] = step.stockfish_component
            value_components[0, idx] = step.value_component
            terminal_components[0, idx] = step.terminal_component

        chunks.append(
            TrajectoryBatch(
                piece_idx=piece_idx,
                aux=aux,
                actions=actions,
                legal_mask=legal_mask,
                old_log_probs=old_log_probs,
                old_values=old_values,
                rewards=rewards,
                dones=dones,
                valid_mask=valid_mask,
                has_search_target=has_search_target,
                search_actions=search_actions,
                search_value_targets=search_value_targets,
                stockfish_components=stockfish_components,
                value_components=value_components,
                terminal_components=terminal_components,
                source=source,
                opponent_label=opponent_label,
            )
        )
    return chunks


def sample_prev_checkpoint(
    prev_pool: list[str],
    rng: random.Random,
    fallback: str,
    *,
    latest_checkpoint: str | None = None,
    best_checkpoint: str | None = None,
    latest_weight: float = 0.5,
    best_weight: float = 0.3,
    older_weight: float = 0.2,
) -> str:
    if not prev_pool:
        return fallback
    choices: list[str] = []
    weights: list[float] = []
    if latest_checkpoint is not None:
        choices.append(latest_checkpoint)
        weights.append(latest_weight)
    if best_checkpoint is not None:
        choices.append(best_checkpoint)
        weights.append(best_weight)
    older = [ckpt for ckpt in prev_pool if ckpt not in set(choices)]
    if older:
        for ckpt in older:
            choices.append(ckpt)
            weights.append(older_weight / len(older))
    if not choices:
        return fallback
    return rng.choices(choices, weights=weights, k=1)[0]
