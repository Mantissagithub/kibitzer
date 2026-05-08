"""Legal-move mask construction for the policy head.

The policy network outputs logits over all 4672 ``(from_square, plane)``
actions. To make sampling and argmax respect the rules, we multiply (or fill
``-inf``) the logits with a Boolean mask that marks legal moves in the current
position.
"""

from __future__ import annotations

import chess
import torch

from kibitzer.encoding import ACTION_SIZE, move_to_index


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a Boolean mask of legal moves at every action index.

    Parameters
    ----------
    board : chess.Board
        Position whose legality is being queried.

    Returns
    -------
    torch.Tensor
        ``BoolTensor`` of shape ``(4672,)``. Entry ``i`` is ``True`` iff some
        legal move in ``board`` encodes to action index ``i`` via
        :func:`kibitzer.encoding.move_to_index`. All other entries are
        ``False``. In a terminal position (checkmate or stalemate) the mask is
        all-False.
    """
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
