"""Board and move encoding for AlphaZero-style chess models.

This module is the bridge between ``python-chess`` objects and the tensor /
integer representations consumed by the policy/value network and the search.

Two responsibilities:

1. ``board_to_tensor`` — encode a ``chess.Board`` as integer piece tokens (one
   per square) plus a small fixed-length vector of auxiliary scalars.
2. ``move_to_index`` / ``index_to_move`` — encode/decode a ``chess.Move`` as a
   flat integer in ``[0, 4672)`` using AlphaZero's 64×73 action layout.

All public conventions (square ordering, piece indices, plane layout, direction
orderings, aux slot order) are frozen here so that downstream code (encoder,
MCTS, dataset, tests) can rely on them.
"""

from __future__ import annotations

import chess
import torch

# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

# Piece token layout for ``piece_idx``:
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
    """Encode a ``chess.Board`` as integer piece tokens plus auxiliary scalars.

    The function returns a dict with two tensors kept separate because the
    encoder treats them differently: ``piece_idx`` is fed through an embedding
    table (categorical), while ``aux`` is concatenated as continuous features.

    Square ordering follows ``chess.SQUARES``: ``A1=0, B1=1, ..., H1=7, A2=8,
    ..., H8=63``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"piece_idx"`` : ``LongTensor`` of shape ``(64,)``.
            For each square, an integer in ``[0, 12]``:

            * ``0``    — empty
            * ``1..6`` — white pawn, knight, bishop, rook, queen, king
            * ``7..12``— black pawn, knight, bishop, rook, queen, king

            Kept as an index (not one-hot) because the network embeds it.

        ``"aux"`` : ``FloatTensor`` of shape ``(7,)``.
            Fixed slot order:

            ============= ==================================================
            Index         Feature
            ============= ==================================================
            0             side-to-move        (1.0 white, 0.0 black)
            1             castling W kingside (0.0 / 1.0)
            2             castling W queenside
            3             castling B kingside
            4             castling B queenside
            5             en passant file     (-1.0 if none, else 0.0..7.0)
            6             halfmove clock      (``min(hm, 100) / 100.0``)
            ============= ==================================================

            The en-passant slot is stored as a float for tensor homogeneity;
            the encoder may round it to an int for an embedding lookup.
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


# ---------------------------------------------------------------------------
# Move encoding (AlphaZero 64 × 73)
# ---------------------------------------------------------------------------

# 8 compass directions for queen-like moves, in canonical AlphaZero order.
# Each entry is a (rank_delta, file_delta) unit vector.
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

# 8 knight (rank_delta, file_delta) offsets in fixed order.
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

# Underpromotion file deltas (mover's POV). Rank delta is implied by side-to-move.
_UNDERPROMOTION_FILE_DELTAS: list[int] = [-1, 0, 1]
# Underpromotion piece order. Queen promotions are NOT encoded here — they ride
# the queen-like planes.
_UNDERPROMOTION_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_PLANES = 73
NUM_FROM_SQUARES = 64
ACTION_SIZE = NUM_FROM_SQUARES * NUM_PLANES  # 4672


def _rank_file(sq: int) -> tuple[int, int]:
    return chess.square_rank(sq), chess.square_file(sq)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a ``chess.Move`` as a flat action index in ``[0, 4672)``.

    Action layout::

        idx = from_square * 73 + plane

    where ``plane`` ∈ ``[0, 73)`` decomposes as:

    * **0..55  — queen-like moves** (covers king, queen, rook, bishop, pawn
      pushes/captures, and queen promotions). 8 compass directions × 7
      distances::

          plane = direction_idx * 7 + (distance - 1)

      Direction indices correspond to ``_QUEEN_DIRS`` (N, NE, E, SE, S, SW, W,
      NW). Distance is the Chebyshev distance from ``from_square`` to
      ``to_square``, in ``{1..7}``.

    * **56..63 — knight moves**::

          plane = 56 + knight_idx

      with ``knight_idx`` indexing ``_KNIGHT_DIRS``.

    * **64..72 — underpromotions** (knight, bishop, rook only)::

          plane = 64 + file_idx * 3 + piece_idx

      where ``file_idx`` selects the file delta from
      ``_UNDERPROMOTION_FILE_DELTAS = [-1, 0, +1]`` (capture-left, push,
      capture-right from the mover's POV) and ``piece_idx`` selects the piece
      from ``_UNDERPROMOTION_PIECES = [KNIGHT, BISHOP, ROOK]``. The rank delta
      is implicit (+1 for white, -1 for black) and is recovered from
      ``board.turn`` during decoding.

    Notes
    -----
    * A move with ``promotion=chess.QUEEN`` is routed through the queen-like
      planes (the index does not carry a promotion bit). The decoder
      re-attaches ``promotion=chess.QUEEN`` when the from-square holds a pawn
      reaching the back rank, so the round-trip is identity-preserving.
    * A pawn move to the back rank with ``promotion=None`` is also routed
      through the queen-like planes; the decoder will return the same
      destination but with ``promotion=chess.QUEEN`` set.
    * Castling in ``python-chess`` is the king moving two squares (e.g.
      ``E1→G1``); this is a queen-like move and needs no special handling.
    * The ``board`` argument is currently unused by the encoder but is part of
      the signature for symmetry with ``index_to_move`` and to support future
      board-dependent encodings (e.g. Chess960). Call sites should still pass
      it.

    Raises
    ------
    ValueError
        If the move geometry is not representable (null move, off-board,
        unknown promotion piece, or non-queen/non-knight delta).
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

    # Underpromotion to N / B / R uses dedicated planes.
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

    # Knight moves: dedicated planes. Knight deltas (|dr|+|df| == 3 with both
    # nonzero) never overlap with queen-like deltas, so this check is safe.
    if (dr, df) in _KNIGHT_DIRS:
        knight_idx = _KNIGHT_DIRS.index((dr, df))
        plane = 56 + knight_idx
        return from_sq * NUM_PLANES + plane

    # Queen-like move (also handles queen promotion and queen-promotion-by-
    # default for pawn-to-back-rank moves with promotion=None).
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
    """Decode a flat action index back to a ``chess.Move``.

    Inverse of :func:`move_to_index` for legal moves.

    The ``board`` argument is required for two reasons:

    1. **Implicit queen promotion.** Queen-like planes do not carry a
       promotion bit. When the index decodes to a pawn move that lands on the
       back rank, this function attaches ``promotion=chess.QUEEN`` based on
       the from-square piece.
    2. **Underpromotion rank delta.** Underpromotion planes (64..72) encode
       only the file delta and the promotion piece. The rank delta is read
       from ``board.turn`` (+1 for white, -1 for black).

    Parameters
    ----------
    idx : int
        Action index in ``[0, 4672)``.
    board : chess.Board
        Position from which the move is being played. Used as described above.

    Returns
    -------
    chess.Move
        The decoded move, with ``promotion`` set if applicable. The function
        does not validate that the move is legal; out-of-board geometry will
        raise.

    Raises
    ------
    ValueError
        If ``idx`` is out of range or decodes to an off-board destination.
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

        # Implicit queen promotion when a pawn reaches the back rank.
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

    # Underpromotion: planes 64..72.
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
