"""Checkpoint loading and single-position policy/value inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from kibitzer.encoding import board_to_tensor, move_to_index
from kibitzer.model import Kibitzer, KibitzerConfig


@dataclass(frozen=True)
class PositionEvaluation:
    priors: dict[chess.Move, float]
    value: float


class ModelEvaluator:
    def __init__(self, model: Kibitzer, *, device: str) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str) -> ModelEvaluator:
        payload = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("checkpoint must contain model weights")
        config = payload.get("config", KibitzerConfig())
        if isinstance(config, dict):
            config = KibitzerConfig(**config)
        model = Kibitzer(config)
        model.load_state_dict(payload["model"])
        return cls(model, device=device)

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        # claimable draws still need a legal UCI move when the tournament manager
        # has not claimed them yet. actual terminal boards are the only invalid input.
        if board.is_game_over(claim_draw=False):
            raise ValueError("cannot evaluate a terminal board")
        encoded = board_to_tensor(board)
        piece_idx = encoded["piece_idx"].unsqueeze(0).unsqueeze(0).to(self.device)
        aux = encoded["aux"].unsqueeze(0).unsqueeze(0).to(self.device)
        logits, value = self.model(piece_idx, aux)

        legal_moves = list(board.legal_moves)
        indices = torch.tensor(
            [move_to_index(move, board) for move in legal_moves],
            device=self.device,
        )
        probabilities = torch.softmax(logits[0, -1, indices].float(), dim=0).cpu().tolist()
        return PositionEvaluation(
            priors=dict(zip(legal_moves, probabilities, strict=True)),
            value=float(value[0, -1, 0].item()),
        )

    @torch.inference_mode()
    def evaluate_batch(self, boards: list[chess.Board]) -> list[PositionEvaluation]:
        b = len(boards)
        pieces = []
        auxes = []
        for board in boards:
            e = board_to_tensor(board)
            pieces.append(e["piece_idx"])
            auxes.append(e["aux"])
        piece_idx = torch.stack(pieces).unsqueeze(1).to(self.device)  # [B, 1, n_squares]
        aux = torch.stack(auxes).unsqueeze(1).to(self.device)  # [B, 1, n_aux]
        logits_all, values_all = self.model(piece_idx, aux)  # [B, 1, A], [B, 1, 1]

        results = []
        for i, board in enumerate(boards):
            legal_moves = list(board.legal_moves)
            indices = torch.tensor(
                [move_to_index(move, board) for move in legal_moves],
                device=self.device,
            )
            probs = torch.softmax(logits_all[i, -1, indices].float(), dim=0).cpu().tolist()
            results.append(PositionEvaluation(
                priors=dict(zip(legal_moves, probs, strict=True)),
                value=float(values_all[i, -1, 0].item()),
            ))
        return results
