"""Data helpers for phase-1 cloning and phase-2 distillation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index


@dataclass(frozen=True)
class PositionSample:
    fen: str
    move_uci: str
    value: float
    game_id: int = 0


def result_to_value(result: str, turn: bool) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else -1.0
    return 0.0


def iter_pgn_samples(
    paths: list[Path],
    *,
    max_games: int | None = None,
    max_positions: int | None = None,
) -> Iterator[PositionSample]:
    games_seen = 0
    positions_seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games_seen += 1
                result = game.headers.get("Result", "*")
                board = game.board()
                for move in game.mainline_moves():
                    # null moves (uci "0000") appear in some pgns and aren't encodable
                    # policy targets; push to keep the game going but don't train on them.
                    if not move:
                        board.push(move)
                        continue
                    yield PositionSample(
                        fen=board.fen(),
                        move_uci=move.uci(),
                        value=result_to_value(result, board.turn),
                        game_id=games_seen,
                    )
                    positions_seen += 1
                    if max_positions is not None and positions_seen >= max_positions:
                        return
                    board.push(move)
                if max_games is not None and games_seen >= max_games:
                    return


def encode_position_sample(sample: PositionSample) -> dict[str, torch.Tensor]:
    board = chess.Board(sample.fen)
    move = chess.Move.from_uci(sample.move_uci)
    encoded = board_to_tensor(board)
    return {
        "piece_idx": encoded["piece_idx"],
        "aux": encoded["aux"],
        "policy_target": torch.tensor(move_to_index(move, board), dtype=torch.long),
        "value_target": torch.tensor(sample.value, dtype=torch.float32),
        "legal_mask": legal_move_mask(board),
    }


class PositionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[PositionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return encode_position_sample(self.samples[idx])


class StreamingPositionDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        paths: list[Path],
        *,
        max_positions: int | None,
        shuffle_buffer_size: int,
        seed: int,
    ) -> None:
        self.paths = paths
        self.max_positions = max_positions
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        paths = self.paths[worker_id::num_workers]
        rng = random.Random(self.seed + self.epoch * 10_000 + worker_id)
        paths = list(paths)
        rng.shuffle(paths)

        max_positions = self.max_positions
        if max_positions is not None and worker is not None:
            max_positions = math.ceil(max_positions / num_workers)
        source = iter_pgn_samples(paths, max_positions=max_positions)
        buffer: list[PositionSample] = []
        for sample in source:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            index = rng.randrange(len(buffer))
            yield encode_position_sample(buffer[index])
            buffer[index] = sample

        rng.shuffle(buffer)
        for sample in buffer:
            yield encode_position_sample(sample)


def collate_positions(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "piece_idx": torch.stack([item["piece_idx"] for item in batch]).unsqueeze(1),
        "aux": torch.stack([item["aux"] for item in batch]).unsqueeze(1),
        "policy_target": torch.stack([item["policy_target"] for item in batch]).unsqueeze(1),
        "value_target": torch.stack([item["value_target"] for item in batch]).unsqueeze(1),
        "legal_mask": torch.stack([item["legal_mask"] for item in batch]).unsqueeze(1),
    }


def dense_policy_from_scores(scores: dict[int, float], temperature: float) -> torch.Tensor:
    target = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    if not scores:
        return target
    indices = torch.tensor(list(scores.keys()), dtype=torch.long)
    values = torch.tensor(list(scores.values()), dtype=torch.float32)
    probs = torch.softmax(values / temperature, dim=-1)
    target[indices] = probs
    return target
