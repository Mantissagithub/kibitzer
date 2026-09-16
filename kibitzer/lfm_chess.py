"""Adapter for mlabonne/LFM2.5-230M-Chess (a HF causal LM, not a UCI engine).

The model reads a position as a fixed 80-token prompt (board cells, side to
move, castling, en passant, halfmove bucket, repetition bucket, then an
8-ply move history), predicts its own win probability, then predicts one
move token masked to the legal moves. This exact protocol is not published
in the model card; it was reverse-engineered from the official browser demo
(https://hf.co/spaces/mlabonne/ChessLFM) with the model author's blessing
implied by the demo being the reference implementation. The demo also wraps
the raw policy in a shallow negamax alpha-beta search (depth 3, width 6,
root width 12) to get its reported ~2004 Elo; `search_depth=0` instead plays
the raw one-pass policy (~1500 Elo per the model card).
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

MAX_HIST = 8
PROMPT_LEN = 80
NUM_VALUE_BINS = 64

_CASTLE_BITS = (("K", 8), ("Q", 4), ("k", 2), ("q", 1))


def _castling_combo(bits: int) -> str:
    combo = "".join(c for c, bit in _CASTLE_BITS if bits & bit)
    return combo or "-"


def _board_cells(board: chess.Board) -> str:
    placement = board.board_fen()
    cells = []
    for rank in placement.split("/"):
        for ch in rank:
            cells.append("." * int(ch) if ch.isdigit() else ch)
    cells_str = "".join(cells)
    if len(cells_str) != 64:
        raise ValueError(f"board expands to {len(cells_str)} cells: {placement}")
    return cells_str


def build_prompt_tokens(
    board: chess.Board,
    history: list[str],
    *,
    repetition: int = 0,
) -> list[str]:
    """Reproduce the demo's `Gy(fen, repetition, history)` token sequence."""
    fen = board.fen()
    _placement, stm, castling, ep, halfmove, _fullmove = fen.split(" ")
    bits = 0
    if castling != "-":
        for ch in castling:
            bit = dict(_CASTLE_BITS)[ch]
            bits |= bit
    ep_file = ep[0] if ep != "-" else "-"
    halfmove_bucket = min(int(halfmove), 100) // 4
    rep_bucket = min(max(repetition, 0), 2)

    tokens = ["<|pos|>"]
    tokens.extend(f"<c:{ch}>" for ch in _board_cells(board))
    tokens.append(f"<stm:{stm}>")
    tokens.append(f"<cast:{_castling_combo(bits)}>")
    tokens.append(f"<ep:{ep_file}>")
    tokens.append(f"<hm:{halfmove_bucket}>")
    tokens.append(f"<rep:{rep_bucket}>")
    tokens.append("<|hist|>")
    recent = history[-MAX_HIST:]
    tokens.extend(["<m:0000>"] * (MAX_HIST - len(recent)))
    tokens.extend(f"<m:{move}>" for move in recent)
    tokens.append("<|eval|>")
    if len(tokens) != PROMPT_LEN:
        raise ValueError(f"prompt is {len(tokens)} tokens, expected {PROMPT_LEN}")
    return tokens


@dataclass(frozen=True)
class SearchSettings:
    depth: int = 3
    root_top_k: int = 12
    top_k: int = 6


class LFMChessPlayer:
    """Plays chess with mlabonne/LFM2.5-230M-Chess, optionally with lookahead search."""

    def __init__(
        self,
        model_id: str = "mlabonne/LFM2.5-230M-Chess",
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        # AutoTokenizer chokes on this repo's tokenizer_config.json (a
        # "TokenizersBackend" class this transformers version doesn't ship yet);
        # the raw tokenizers.json has every special token we need, so load it
        # directly and skip AutoTokenizer entirely.
        tokenizer_path = hf_hub_download(model_id, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype if device == "cuda" else torch.float32
        ).to(device).eval()

        self._value_ids = torch.tensor(
            [self._token_id(f"<v:{k}>") for k in range(NUM_VALUE_BINS)], device=device
        )
        self._bestmove_id = self._token_id("<|bestmove|>")
        # per-move transposition cache; cleared before each top-level search
        self._value_cache: dict[tuple, float] = {}
        self._board_cache: dict[tuple, tuple[float, list[tuple[str, float]]]] = {}

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"token {token!r} not found in tokenizer vocab")
        return token_id

    def _move_token_id(self, uci: str) -> int:
        return self._token_id(f"<m:{uci}>")

    def _cache_key(self, board: chess.Board, history: list[str]) -> tuple:
        fen_key = " ".join(board.fen().split(" ")[:4])
        return (fen_key, tuple(history[-MAX_HIST:]))

    @torch.inference_mode()
    def _forward_last_logits(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[0, -1, :]

    @torch.inference_mode()
    def _value_and_move_logits(self, prompt_tokens: list[str]) -> tuple[float, torch.Tensor]:
        # two independent full forward passes (no manual KV-cache splicing): LFM2's
        # hybrid conv+attention layers keep their own short conv cache, which is easy
        # to get subtly wrong when hand-rolling continuation; re-running the ~80-token
        # prefix is cheap for a 230M model, so correctness wins over the micro-optimization.
        token_ids = [self._token_id(t) for t in prompt_tokens]
        first_logits = self._forward_last_logits(token_ids)
        value_bin = int(torch.argmax(first_logits[self._value_ids]).item())
        winprob = (value_bin + 0.5) / NUM_VALUE_BINS

        value_token_id = int(self._value_ids[value_bin].item())
        move_logits = self._forward_last_logits(token_ids + [value_token_id, self._bestmove_id])
        return winprob, move_logits

    def _evaluate_value_only(
        self, board: chess.Board, history: list[str], repetition: int
    ) -> float:
        key = self._cache_key(board, history)
        if key in self._value_cache:
            return self._value_cache[key]
        prompt = build_prompt_tokens(board, history, repetition=repetition)
        logits = self._forward_last_logits([self._token_id(t) for t in prompt])
        value_bin = int(torch.argmax(logits[self._value_ids]).item())
        value = (value_bin + 0.5) / NUM_VALUE_BINS
        self._value_cache[key] = value
        return value

    def _evaluate_board(
        self, board: chess.Board, history: list[str], repetition: int
    ) -> tuple[float, list[tuple[str, float]]]:
        key = self._cache_key(board, history)
        if key in self._board_cache:
            return self._board_cache[key]
        prompt = build_prompt_tokens(board, history, repetition=repetition)
        winprob, move_logits = self._value_and_move_logits(prompt)

        legal_ucis = [move.uci() for move in board.legal_moves]
        move_ids = torch.tensor([self._move_token_id(uci) for uci in legal_ucis], device=self.device)
        legal_logits = move_logits[move_ids].float()
        probs = torch.softmax(legal_logits, dim=0).cpu().tolist()
        priors = sorted(zip(legal_ucis, probs, strict=True), key=lambda item: -item[1])

        self._board_cache[key] = (winprob, priors)
        self._value_cache[key] = winprob
        return winprob, priors

    def _negamax(
        self,
        board: chess.Board,
        history: list[str],
        depth: int,
        alpha: float,
        beta: float,
        settings: SearchSettings,
    ) -> float:
        if board.is_checkmate():
            return 0.0
        if board.is_game_over(claim_draw=True):
            return 0.5
        if depth <= 0:
            return self._evaluate_value_only(board, history, repetition=0)

        _root_value, priors = self._evaluate_board(board, history, repetition=0)
        best = 0.0
        for uci, _prob in priors[: settings.top_k]:
            move = chess.Move.from_uci(uci)
            board.push(move)
            history.append(uci)
            child_score = self._negamax(board, history, depth - 1, 1 - beta, 1 - alpha, settings)
            history.pop()
            board.pop()
            value = 1.0 - child_score
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def search_move(
        self,
        board: chess.Board,
        history: list[str],
        *,
        settings: SearchSettings = SearchSettings(),
        repetition: int = 0,
    ) -> chess.Move:
        """Pick a move for `board`, given its real preceding `history` (uci strings)."""
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search a terminal board")
        self._value_cache.clear()
        self._board_cache.clear()

        _root_value, root_priors = self._evaluate_board(board, history, repetition)
        if settings.depth <= 0:
            return chess.Move.from_uci(root_priors[0][0])

        best_uci = root_priors[0][0]
        best_score = -1.0
        alpha = 0.0
        for uci, _prob in root_priors[: settings.root_top_k]:
            move = chess.Move.from_uci(uci)
            board.push(move)
            history.append(uci)
            child_score = self._negamax(board, history, settings.depth - 1, 0.0, 1 - alpha, settings)
            history.pop()
            board.pop()
            value = 1.0 - child_score
            if value > best_score:
                best_score = value
                best_uci = uci
            alpha = max(alpha, value)
        return chess.Move.from_uci(best_uci)
