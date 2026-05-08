"""Stateful inference engine for the Kibitzer model.

Wraps :class:`kibitzer.model.Kibitzer` with a game-history buffer and a
small action-oriented API: ``reset``, ``push_move``, ``evaluate``,
``select_move``, ``get_principal_variation``.

Every call to :meth:`KibitzerEngine.evaluate` re-runs the full history
through the model — there is no KV cache yet (see TODO inside ``evaluate``).
For games up to ``max_seq_len`` plies on a single mid-range GPU with bf16
this is fast enough; revisit if self-play throughput becomes a bottleneck.
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
    """Stateful interface around a :class:`Kibitzer` for play and analysis."""

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

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, board: chess.Board | None = None) -> None:
        """Clear the game history; optionally start from ``board``."""
        start = chess.Board() if board is None else board.copy(stack=False)
        self.history = [start]

    def push_move(self, move: chess.Move) -> None:
        """Apply ``move`` to the current position and append the result."""
        nb = self.history[-1].copy(stack=False)
        nb.push(move)
        self.history.append(nb)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_last(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, float, chess.Board]:
        """Run the model and return ``(logits, mask, value, current_board)``.

        ``logits`` and ``mask`` are 1-D tensors of length ``ACTION_SIZE``
        (4672) on ``self.device``; illegal entries in ``logits`` are NOT
        masked yet (callers do that). ``value`` is a Python float. The
        ``current_board`` is ``history[-1]`` for downstream lookups.
        """
        # TODO(perf): re-forwards the entire history every call. For games up
        # to ``max_seq_len`` plies on a 4060 with bf16 this is fast enough;
        # KV-caching is awkward here because the per-position encoder runs
        # independently per ply while the causal trunk runs over time.
        # Revisit if self-play throughput becomes a bottleneck.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Return policy probs, value, legal moves, and ranked move probs.

        Returns
        -------
        dict
            ``policy`` : ``np.ndarray`` of shape ``(4672,)``, float32, summing
                to 1.0 over the legal moves only (illegal entries are 0).
            ``value`` : ``float`` in ``[-1, 1]``.
            ``legal_moves`` : ``list[chess.Move]`` for the current position.
            ``move_probs`` : ``list[(chess.Move, float)]`` sorted descending
                by probability.
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
        """Pick a move from the current position.

        ``temperature == 0.0`` returns the legal-masked argmax (deterministic).
        ``temperature > 0`` samples from ``softmax(logits / T)`` over the
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
            # Restrict to top-k of the legal entries.
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
        """Reposition the engine to ``board`` and return a sampled move.

        Wraps the engine as a ``Callable[[chess.Board], chess.Move]`` for use
        with :func:`kibitzer.match.play_match`. Restores the engine's previous
        history before returning.

        If ``board`` has a populated ``move_stack``, the engine replays from
        the root so the causal trunk sees the full game history. If
        ``move_stack`` is empty (e.g. the board was built directly from a
        FEN), the engine treats the board as a single-ply history — fine for
        position analysis, less ideal for mid-game queries.
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
        """Greedy rollout: argmax-push for ``depth`` plies (or until game end).

        Restores the original history before returning.
        """
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
