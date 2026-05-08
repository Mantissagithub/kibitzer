"""Streaming Lichess-PGN dataset for supervised pretraining.

``LichessGameDataset`` is a ``torch.utils.data.IterableDataset`` that walks
PGN files game-by-game (no full-file buffering), filters by Elo / time
control / termination, and emits one tensor dict per game. ``collate_games``
pads a list of those dicts into a (B, T_max, …) batch with a ``loss_mask``.

Per-ply encoding is delegated to the existing modules — DO NOT reimplement:

* :func:`kibitzer.encoding.board_to_tensor` → ``piece_idx`` (64,) + ``aux`` (7,)
* :func:`kibitzer.encoding.move_to_index`   → flat action index in [0, 4672)
* :func:`kibitzer.masking.legal_move_mask`  → BoolTensor (4672,)

The shuffle is **game-level**, not position-level: the buffer holds parsed
game dicts and yields a random one when full. Positions inside a game stay
consecutive, which is what the value-target supervision relies on.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info

from kibitzer.encoding import (
    ACTION_SIZE,
    AUX_SIZE,
    board_to_tensor,
    move_to_index,
)
from kibitzer.masking import legal_move_mask


_RESULT_TO_WHITE_SCORE: dict[str, float] = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}

# Lichess Termination headers we drop entirely. "Time forfeit" we keep —
# min_plies already protects against the very short ones, and a 60-ply game
# that ended on the clock is still useful supervision.
_BAD_TERMINATIONS = {"Abandoned", "Rules infraction", "Unterminated"}


def _partition_paths(
    paths: Sequence[str], worker_info
) -> list[str]:
    """Disjoint slice of ``paths`` for the current DataLoader worker."""
    if worker_info is None:
        return list(paths)
    return list(paths[worker_info.id :: worker_info.num_workers])


def _passes_filters(
    game: chess.pgn.Game,
    min_elo: int,
    time_controls: list[str] | None,
) -> bool:
    h = game.headers
    if h.get("Result") not in _RESULT_TO_WHITE_SCORE:
        return False
    if h.get("Termination") in _BAD_TERMINATIONS:
        return False
    try:
        we = int(h.get("WhiteElo", "0"))
        be = int(h.get("BlackElo", "0"))
    except ValueError:
        return False
    if we < min_elo or be < min_elo:
        return False
    if time_controls is not None:
        event = (h.get("Event") or "").lower()
        if not any(tc.lower() in event for tc in time_controls):
            return False
    return True


def _encode_game(
    game: chess.pgn.Game,
    skip_first_n_plies: int,
    max_plies: int,
    min_plies: int,
) -> dict | None:
    """Encode one game's mainline; return None if it fails the post-slice filter."""
    moves_all = list(game.mainline_moves())
    moves = moves_all[skip_first_n_plies : skip_first_n_plies + max_plies]
    if len(moves) < min_plies:
        return None

    white_score = _RESULT_TO_WHITE_SCORE[game.headers["Result"]]
    board = game.board()
    for i in range(skip_first_n_plies):
        board.push(moves_all[i])

    T = len(moves)
    piece_idx = torch.zeros(T, 64, dtype=torch.long)
    aux = torch.zeros(T, AUX_SIZE, dtype=torch.float32)
    move_idx = torch.zeros(T, dtype=torch.long)
    legal_mask = torch.zeros(T, ACTION_SIZE, dtype=torch.bool)
    value_target = torch.zeros(T, dtype=torch.float32)

    for t, mv in enumerate(moves):
        enc = board_to_tensor(board)
        piece_idx[t] = enc["piece_idx"]
        aux[t] = enc["aux"]
        legal_mask[t] = legal_move_mask(board)
        try:
            move_idx[t] = move_to_index(mv, board)
        except ValueError:
            return None  # malformed move geometry — drop the game
        # Value from side-to-move's perspective at this ply. White wins → +1
        # when it's white's turn, -1 when it's black's turn; flipped for black
        # wins; 0 throughout for draws.
        value_target[t] = white_score if board.turn == chess.WHITE else -white_score
        board.push(mv)

    return {
        "piece_idx": piece_idx,
        "aux": aux,
        "move_idx": move_idx,
        "legal_mask": legal_mask,
        "value_target": value_target,
        "ply_count": T,
    }


def _iter_games_from_path(path: str) -> Iterator[chess.pgn.Game]:
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            yield game


class LichessGameDataset(IterableDataset):
    def __init__(
        self,
        pgn_paths: list[str] | list[Path],
        min_elo: int = 2400,
        min_plies: int = 10,
        max_plies: int = 256,
        skip_first_n_plies: int = 0,
        shuffle_buffer_size: int = 4096,
        time_controls: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.pgn_paths = [str(p) for p in pgn_paths]
        self.min_elo = min_elo
        self.min_plies = min_plies
        self.max_plies = max_plies
        self.skip_first_n_plies = skip_first_n_plies
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.time_controls = list(time_controls) if time_controls else None
        self.seed = seed

    def _iter_raw(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        my_paths = _partition_paths(self.pgn_paths, worker_info)
        for path in my_paths:
            for game in _iter_games_from_path(path):
                if not _passes_filters(game, self.min_elo, self.time_controls):
                    continue
                encoded = _encode_game(
                    game,
                    self.skip_first_n_plies,
                    self.max_plies,
                    self.min_plies,
                )
                if encoded is not None:
                    yield encoded

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        if self.shuffle_buffer_size <= 1:
            yield from self._iter_raw()
            return

        buffer: list[dict] = []
        for item in self._iter_raw():
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(item)
            else:
                pos = rng.randrange(len(buffer))
                yield buffer[pos]
                buffer[pos] = item
        rng.shuffle(buffer)
        yield from buffer


def collate_games(games: list[dict]) -> dict:
    """Pad a list of game dicts into a (B, T_max, …) batch with a loss_mask."""
    if not games:
        raise ValueError("collate_games called with an empty batch")

    B = len(games)
    T_max = max(g["ply_count"] for g in games)

    piece_idx = torch.zeros(B, T_max, 64, dtype=torch.long)
    aux = torch.zeros(B, T_max, AUX_SIZE, dtype=torch.float32)
    move_idx = torch.zeros(B, T_max, dtype=torch.long)
    legal_mask = torch.zeros(B, T_max, ACTION_SIZE, dtype=torch.bool)
    value_target = torch.zeros(B, T_max, dtype=torch.float32)
    loss_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for b, g in enumerate(games):
        T = g["ply_count"]
        piece_idx[b, :T] = g["piece_idx"]
        aux[b, :T] = g["aux"]
        move_idx[b, :T] = g["move_idx"]
        legal_mask[b, :T] = g["legal_mask"]
        value_target[b, :T] = g["value_target"]
        loss_mask[b, :T] = True

    return {
        "piece_idx": piece_idx,
        "aux": aux,
        "move_idx": move_idx,
        "legal_mask": legal_mask,
        "value_target": value_target,
        "loss_mask": loss_mask,
    }
