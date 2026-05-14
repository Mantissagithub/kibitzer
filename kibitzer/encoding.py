"""board and move encoding for alphazero-style chess models.

this module is the bridge between ``python-chess`` objects and the tensor /
integer formats consumed by the policy/value network and search.

two responsibilities:

1. ``board_to_tensor`` — encode a ``chess.board`` as integer piece tokens (one
   per square) plus a small fixed-length vector of auxiliary scalars.
2. ``move_to_index`` / ``index_to_move`` — encode/decode a ``chess.move`` as a
   flat integer in ``[0, 4672)`` using alphazero's 64×73 action layout.

the public conventions here are fixed: square order, piece indices, plane
layout, direction order, and aux slot order.
"""

from __future__ import annotations

import chess
import torch

# piece token layout for ``piece_idx``:
#   0          empty
#   1..6       white  P, N, B, R, Q, K
#   7..12      black  P, N, B, R, Q, K
_PIECE_TO_IDX: dict[tuple[int, bool], int] = {
    (chess.PAWN, chess.WHITE): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5,
    (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 7,
    (chess.KNIGHT, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.ROOK, chess.BLACK): 10,
    (chess.QUEEN, chess.BLACK): 11,
    (chess.KING, chess.BLACK): 12,
}

NUM_PIECE_TOKENS = 13  # 0 (empty) + 12 piece-color combinations
AUX_SIZE = 7


def board_to_tensor(board: chess.Board) -> dict[str, torch.Tensor]:
    """encode a board as ``piece_idx`` and ``aux`` tensors.

    square order follows ``python-chess``: ``a1=0`` through ``h8=63``.
    ``aux`` slots are side to move, castling rights, en-passant file
    (``-1`` when absent), and capped halfmove clock.
    """
    piece_idx = torch.zeros(64, dtype=torch.long)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            piece_idx[sq] = _PIECE_TO_IDX[(piece.piece_type, piece.color)]

    side = 1.0 if board.turn == chess.WHITE else 0.0
    wk = float(board.has_kingside_castling_rights(chess.WHITE))
    wq = float(board.has_queenside_castling_rights(chess.WHITE))
    bk = float(board.has_kingside_castling_rights(chess.BLACK))
    bq = float(board.has_queenside_castling_rights(chess.BLACK))
    ep_file = (
        float(chess.square_file(board.ep_square))
        if board.ep_square is not None
        else -1.0
    )
    halfmove = min(board.halfmove_clock, 100) / 100.0

    aux = torch.tensor(
        [side, wk, wq, bk, bq, ep_file, halfmove], dtype=torch.float32
    )
    return {"piece_idx": piece_idx, "aux": aux}


# queen-like move directions, in alphazero order.
_QUEEN_DIRS: list[tuple[int, int]] = [
    (1, 0),    # 0  N
    (1, 1),    # 1  NE
    (0, 1),    # 2  E
    (-1, 1),   # 3  SE
    (-1, 0),   # 4  S
    (-1, -1),  # 5  SW
    (0, -1),   # 6  W
    (1, -1),   # 7  NW
]

# knight move offsets in fixed order.
_KNIGHT_DIRS: list[tuple[int, int]] = [
    (2, 1),    # 0
    (1, 2),    # 1
    (-1, 2),   # 2
    (-2, 1),   # 3
    (-2, -1),  # 4
    (-1, -2),  # 5
    (1, -2),   # 6
    (2, -1),   # 7
]

# underpromotion file deltas from the mover's point of view.
_UNDERPROMOTION_FILE_DELTAS: list[int] = [-1, 0, 1]
# queen promotions ride the queen-like planes; only these get special planes.
_UNDERPROMOTION_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_PLANES = 73
NUM_FROM_SQUARES = 64
ACTION_SIZE = NUM_FROM_SQUARES * NUM_PLANES  # 4672


def _rank_file(sq: int) -> tuple[int, int]:
    return chess.square_rank(sq), chess.square_file(sq)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """encode a move as ``from_square * 73 + plane``.

    planes 0..55 are queen-like moves, 56..63 are knight moves, and 64..72
    are knight/bishop/rook underpromotions. queen promotions use queen-like
    planes and get restored during decoding.
    """
    del board  # currently unused; kept for API symmetry / future extensions

    from_sq = move.from_square
    to_sq = move.to_square
    fr_rank, fr_file = _rank_file(from_sq)
    to_rank, to_file = _rank_file(to_sq)
    dr = to_rank - fr_rank
    df = to_file - fr_file

    if dr == 0 and df == 0:
        raise ValueError(f"Null move (from == to): {move}")

    # underpromotions use dedicated planes.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        if move.promotion not in _UNDERPROMOTION_PIECES:
            raise ValueError(f"Unsupported promotion piece: {move.promotion}")
        if df not in _UNDERPROMOTION_FILE_DELTAS or abs(dr) != 1:
            raise ValueError(
                f"Underpromotion {move} has invalid delta dr={dr} df={df}"
            )
        file_idx = _UNDERPROMOTION_FILE_DELTAS.index(df)
        piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
        plane = 64 + file_idx * 3 + piece_idx
        return from_sq * NUM_PLANES + plane

    # knight deltas never overlap with queen-like deltas.
    if (dr, df) in _KNIGHT_DIRS:
        knight_idx = _KNIGHT_DIRS.index((dr, df))
        plane = 56 + knight_idx
        return from_sq * NUM_PLANES + plane

    # queen-like moves also cover queen promotion by convention.
    distance = max(abs(dr), abs(df))
    if dr % distance != 0 or df % distance != 0:
        raise ValueError(
            f"Move {move} is neither queen-like nor knight: dr={dr} df={df}"
        )
    udir = (dr // distance, df // distance)
    if udir not in _QUEEN_DIRS:
        raise ValueError(f"Move {move} has unrecognized direction {udir}")
    if not 1 <= distance <= 7:
        raise ValueError(f"Move {move} has out-of-range distance {distance}")
    dir_idx = _QUEEN_DIRS.index(udir)
    plane = dir_idx * 7 + (distance - 1)
    return from_sq * NUM_PLANES + plane


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """decode an action index back to a move for ``board``.

    ``board`` supplies the pawn color for implicit queen promotions and the
    rank direction for underpromotion planes.
    """
    if not 0 <= idx < ACTION_SIZE:
        raise ValueError(f"Index {idx} out of range [0, {ACTION_SIZE})")

    from_sq, plane = divmod(idx, NUM_PLANES)
    fr_rank, fr_file = _rank_file(from_sq)

    if plane < 56:
        dir_idx, dist_minus_1 = divmod(plane, 7)
        distance = dist_minus_1 + 1
        dr_unit, df_unit = _QUEEN_DIRS[dir_idx]
        to_rank = fr_rank + dr_unit * distance
        to_file = fr_file + df_unit * distance
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            raise ValueError(
                f"Index {idx} (queen-like) decodes off-board: "
                f"to=({to_rank},{to_file})"
            )
        to_sq = chess.square(to_file, to_rank)

        # restore implicit queen promotion when a pawn reaches the back rank.
        promotion: int | None = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    if plane < 64:
        knight_idx = plane - 56
        dr, df = _KNIGHT_DIRS[knight_idx]
        to_rank = fr_rank + dr
        to_file = fr_file + df
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            raise ValueError(
                f"Index {idx} (knight) decodes off-board: "
                f"to=({to_rank},{to_file})"
            )
        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq)

    # underpromotion planes.
    up_idx = plane - 64
    file_idx, piece_idx = divmod(up_idx, 3)
    df = _UNDERPROMOTION_FILE_DELTAS[file_idx]
    promotion_piece = _UNDERPROMOTION_PIECES[piece_idx]
    dr = 1 if board.turn == chess.WHITE else -1
    to_rank = fr_rank + dr
    to_file = fr_file + df
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        raise ValueError(
            f"Index {idx} (underpromotion) decodes off-board: "
            f"to=({to_rank},{to_file})"
        )
    to_sq = chess.square(to_file, to_rank)
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)
