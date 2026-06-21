"""Data helpers for phase-1 cloning and phase-2 distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index


@dataclass(frozen=True)
class PositionSample:
    fen: str
    move_uci: str
    value: float


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
                    yield PositionSample(
                        fen=board.fen(),
                        move_uci=move.uci(),
                        value=result_to_value(result, board.turn),
                    )
                    positions_seen += 1
                    if max_positions is not None and positions_seen >= max_positions:
                        return
                    board.push(move)
                if max_games is not None and games_seen >= max_games:
                    return


class PositionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[PositionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
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
