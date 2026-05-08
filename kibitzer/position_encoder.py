"""Chess position encoder — turns board tensors into a fixed-size vector.

Consumes the dict produced by :func:`kibitzer.encoding.board_to_tensor`
(``piece_idx (B, 64)`` long + ``aux (B, 7)`` float) and produces a single
``(B, d_model)`` representation that downstream heads (policy / value / move)
can share.

Architecture: a learned per-square positional embedding plus a per-square piece
embedding, with global aux features projected and broadcast onto every square.
The 64 tokens go through ``n_layers`` of bidirectional pre-norm encoder blocks
(no RoPE — square ordering is arbitrary), then RMSNorm and mean pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kibitzer.transformer import EncoderBlock, RMSNorm


class PositionEncoder(nn.Module):
    """Encode a batch of chess positions into ``(B, d_model)`` vectors.

    Parameters
    ----------
    d_model : int
        Hidden width.
    n_heads : int
        Attention heads per encoder block; must divide ``d_model``.
    n_layers : int
        Number of stacked :class:`EncoderBlock` layers.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.piece_emb = nn.Embedding(13, d_model)
        self.square_emb = nn.Embedding(64, d_model)
        self.aux_proj = nn.Linear(7, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self, piece_idx: torch.Tensor, aux: torch.Tensor
    ) -> torch.Tensor:
        """Encode ``(piece_idx, aux)`` to ``(B, d_model)``.

        Parameters
        ----------
        piece_idx : torch.Tensor
            ``LongTensor`` of shape ``(B, 64)`` with values in ``[0, 12]``.
        aux : torch.Tensor
            ``FloatTensor`` of shape ``(B, 7)``.

        Returns
        -------
        torch.Tensor
            ``(B, d_model)`` mean-pooled board representation.
        """
        squares = torch.arange(64, device=piece_idx.device)
        x = self.piece_emb(piece_idx) + self.square_emb(squares)  # (B, 64, D)
        x = x + self.aux_proj(aux).unsqueeze(1)                    # (B, 64, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x.mean(dim=1)
