"""stateful inference engine for the kibitzer model.

keeps a board-history buffer and exposes the small play api used by scripts
and match runners. calls still re-forward the full history; that is fine for
short games and can be revisited if self-play throughput needs it.
"""

from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn.functional as F

from kibitzer.encoding import board_to_tensor, index_to_move, move_to_index
from kibitzer.masking import legal_move_mask
from kibitzer.model import Kibitzer


class KibitzerEngine:

    def __init__(
        self,
        model: Kibitzer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(self.device).to(self.dtype).eval()
        self.history: list[chess.Board] = [chess.Board()]

    def reset(self, board: chess.Board | None = None) -> None:
        start = chess.Board() if board is None else board.copy(stack=False)
        self.history = [start]

    def push_move(self, move: chess.Move) -> None:
        nb = self.history[-1].copy(stack=False)
        nb.push(move)
        self.history.append(nb)

    def _forward_last(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, float, chess.Board]:
        """run the model for the current board history."""
        # todo(perf): re-forwards the full history. kv-caching is awkward here
        # because position encoding is per ply while the causal trunk spans time.
        T = len(self.history)
        if T == 0:
            raise RuntimeError("history is empty; call reset() first")
        max_T = self.model.config.max_seq_len
        if T > max_T:
            raise RuntimeError(
                f"history length {T} exceeds model.max_seq_len={max_T}"
            )

        piece_idxs = []
        auxes = []
        for b in self.history:
            t = board_to_tensor(b)
            piece_idxs.append(t["piece_idx"])
            auxes.append(t["aux"])
        piece_idx = torch.stack(piece_idxs).unsqueeze(0).to(self.device)
        aux = torch.stack(auxes).unsqueeze(0).to(self.device)
        pad_mask = torch.zeros(1, T, dtype=torch.bool, device=self.device)

        amp_enabled = self.dtype in (torch.bfloat16, torch.float16)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=amp_enabled,
        ):
            policy_logits, value_pred = self.model(piece_idx, aux, pad_mask)

        logits = policy_logits[0, -1].float()           # (4672,)
        value = float(value_pred[0, -1, 0].float().item())

        current = self.history[-1]
        mask = legal_move_mask(current).to(self.device)  # (4672,) bool

        return logits, mask, value, current

    @torch.no_grad()
    def evaluate(self) -> dict:
        """return policy probs, value, legal moves, and ranked move probs.

        returns
        -------
        dict
            ``policy`` : ``np.ndarray`` of shape ``(4672,)``, float32, summing
                to 1.0 over the legal moves only (illegal entries are 0).
            ``value`` : ``float`` in ``[-1, 1]``.
            ``legal_moves`` : ``list[chess.move]`` for the current position.
            ``move_probs`` : ``list[(chess.move, float)]`` sorted descending.
        """
        logits, mask, value, current = self._forward_last()
        masked = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(masked, dim=-1)

        legal_moves = list(current.legal_moves)
        move_probs: list[tuple[chess.Move, float]] = []
        for m in legal_moves:
            idx = move_to_index(m, current)
            move_probs.append((m, float(probs[idx].item())))
        move_probs.sort(key=lambda mp: mp[1], reverse=True)

        policy_np = probs.cpu().numpy().astype(np.float32)
        return {
            "policy": policy_np,
            "value": value,
            "legal_moves": legal_moves,
            "move_probs": move_probs,
        }

    @torch.no_grad()
    def select_move(
        self, temperature: float = 0.0, top_k: int | None = None
    ) -> chess.Move:
        """pick a move from the current position.

        ``temperature == 0.0`` returns the legal-masked argmax (deterministic).
        ``temperature > 0`` samples from ``softmax(logits / t)`` over the
        legal moves; ``top_k`` (if given) restricts sampling to the top ``k``
        by logit before softmax.
        """
        logits, mask, _value, current = self._forward_last()

        if temperature == 0.0:
            masked = logits.masked_fill(~mask, float("-inf"))
            idx = int(torch.argmax(masked).item())
            return index_to_move(idx, current)

        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        scaled = logits.masked_fill(~mask, float("-inf")) / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be > 0, got {top_k}")
            # sample only among the top-k legal logits.
            n_legal = int(mask.sum().item())
            k = min(top_k, n_legal)
            topk_vals, topk_idx = torch.topk(scaled, k)
            kept = torch.full_like(scaled, float("-inf"))
            kept[topk_idx] = topk_vals
            scaled = kept

        probs = F.softmax(scaled, dim=-1)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        return index_to_move(idx, current)

    @torch.no_grad()
    def evaluate_at(
        self, board: chess.Board, temperature: float = 0.0
    ) -> chess.Move:
        """temporarily evaluate ``board`` and return a sampled move.

        if the board has a ``move_stack``, replay from the root so the causal
        trunk sees the game history. otherwise use it as a one-position query.
        the previous engine history is restored before returning.
        """
        saved = list(self.history)
        try:
            if board.move_stack:
                self.reset(board.root())
                for m in board.move_stack:
                    self.push_move(m)
            else:
                self.reset(board)
            return self.select_move(temperature=temperature)
        finally:
            self.history = saved

    @torch.no_grad()
    def get_principal_variation(self, depth: int = 5) -> list[chess.Move]:
        """greedy rollout for ``depth`` plies, restoring history afterward."""
        saved = list(self.history)
        pv: list[chess.Move] = []
        try:
            for _ in range(depth):
                if self.history[-1].is_game_over():
                    break
                move = self.select_move(temperature=0.0)
                pv.append(move)
                self.push_move(move)
        finally:
            self.history = saved
        return pv
