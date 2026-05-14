"""legal-move masks for the policy head."""

from __future__ import annotations

import chess
import torch

from kibitzer.encoding import ACTION_SIZE, move_to_index


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """return a boolean action mask for legal moves in ``board``."""
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
