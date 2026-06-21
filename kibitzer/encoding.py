"""Chess board and AlphaZero-style move encoding."""

from __future__ import annotations

import chess
import torch

NUM_PIECE_TOKENS = 13
AUX_SIZE = 7
NUM_PLANES = 73
NUM_FROM_SQUARES = 64
ACTION_SIZE = NUM_FROM_SQUARES * NUM_PLANES

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

_QUEEN_DIRS: list[tuple[int, int]] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]
_KNIGHT_DIRS: list[tuple[int, int]] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]
_UNDERPROMOTION_FILE_DELTAS = [-1, 0, 1]
_UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def board_to_tensor(board: chess.Board) -> dict[str, torch.Tensor]:
    """Encode one board as piece indices plus small scalar state."""
    piece_idx = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx[square] = _PIECE_TO_IDX[(piece.piece_type, piece.color)]

    ep_file = -1.0
    if board.ep_square is not None:
        ep_file = float(chess.square_file(board.ep_square))

    aux = torch.tensor(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            ep_file,
            min(board.halfmove_clock, 100) / 100.0,
        ],
        dtype=torch.float32,
    )
    return {"piece_idx": piece_idx, "aux": aux}


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a bool mask over the fixed action space."""
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask


def _rank_file(square: int) -> tuple[int, int]:
    return chess.square_rank(square), chess.square_file(square)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a legal chess move into the 64 x 73 AlphaZero action layout."""
    del board
    from_sq = move.from_square
    fr_rank, fr_file = _rank_file(from_sq)
    to_rank, to_file = _rank_file(move.to_square)
    dr = to_rank - fr_rank
    df = to_file - fr_file

    if move.promotion is not None and move.promotion != chess.QUEEN:
        if move.promotion not in _UNDERPROMOTION_PIECES:
            raise ValueError(f"unsupported promotion: {move}")
        if abs(dr) != 1 or df not in _UNDERPROMOTION_FILE_DELTAS:
            raise ValueError(f"invalid underpromotion delta: {move}")
        file_idx = _UNDERPROMOTION_FILE_DELTAS.index(df)
        piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
        return from_sq * NUM_PLANES + 64 + file_idx * 3 + piece_idx

    if (dr, df) in _KNIGHT_DIRS:
        return from_sq * NUM_PLANES + 56 + _KNIGHT_DIRS.index((dr, df))

    distance = max(abs(dr), abs(df))
    if distance == 0 or dr % distance != 0 or df % distance != 0:
        raise ValueError(f"move is not encodable: {move}")
    direction = (dr // distance, df // distance)
    if direction not in _QUEEN_DIRS or not 1 <= distance <= 7:
        raise ValueError(f"move is not queen-like: {move}")
    return from_sq * NUM_PLANES + _QUEEN_DIRS.index(direction) * 7 + distance - 1


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode an action index for a board."""
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"action index out of range: {index}")

    from_sq, plane = divmod(index, NUM_PLANES)
    fr_rank, fr_file = _rank_file(from_sq)

    if plane < 56:
        dir_idx, dist_minus_1 = divmod(plane, 7)
        dr, df = _QUEEN_DIRS[dir_idx]
        to_rank = fr_rank + dr * (dist_minus_1 + 1)
        to_file = fr_file + df * (dist_minus_1 + 1)
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            raise ValueError(f"decoded off board: {index}")
        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN
        return chess.Move(from_sq, chess.square(to_file, to_rank), promotion=promotion)

    if plane < 64:
        dr, df = _KNIGHT_DIRS[plane - 56]
        to_rank = fr_rank + dr
        to_file = fr_file + df
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            raise ValueError(f"decoded off board: {index}")
        return chess.Move(from_sq, chess.square(to_file, to_rank))

    under_idx = plane - 64
    file_idx, piece_idx = divmod(under_idx, 3)
    dr = 1 if board.turn == chess.WHITE else -1
    df = _UNDERPROMOTION_FILE_DELTAS[file_idx]
    to_rank = fr_rank + dr
    to_file = fr_file + df
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        raise ValueError(f"decoded off board: {index}")
    return chess.Move(
        from_sq,
        chess.square(to_file, to_rank),
        promotion=_UNDERPROMOTION_PIECES[piece_idx],
    )
