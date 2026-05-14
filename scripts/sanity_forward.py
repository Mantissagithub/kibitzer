"""end-to-end smoke test: pgn → encoding → mask → forward → softmax.

drives the (untrained) kibitzer on a known game (morphy's opera game, paris
1858). the point is integration: shape/dtype/device agreement and that
mask + softmax produce a valid distribution supported only on legal moves.
the actual policy values are garbage by design (untrained weights).
"""

from __future__ import annotations

import io

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from kibitzer.encoding import board_to_tensor, move_to_index
from kibitzer.masking import legal_move_mask
from kibitzer.model import Kibitzer


PGN = """[Event "Paris"]
[Site "Paris FRA"]
[Date "1858.??.??"]
[White "Morphy, Paul"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6
7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7
12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7
16. Qb8+ Nxb8 17. Rd8# 1-0
"""


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Kibitzer().to(device).eval()

    n = model.num_params()
    print(f"params: {n:_}")
    print(f"  FP32 weights: {n * 4 / 1e6:7.1f} MB  ({n * 4 / 1e9:.2f} GB)")
    print(f"  BF16 weights: {n * 2 / 1e6:7.1f} MB  ({n * 2 / 1e9:.2f} GB)")

    game = chess.pgn.read_game(io.StringIO(PGN))
    if game is None:
        raise RuntimeError("PGN failed to parse")

    board = game.board()
    piece_idxs: list[torch.Tensor] = []
    auxes: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    move_indices: list[int] = []

    for move in game.mainline_moves():
        t = board_to_tensor(board)
        piece_idxs.append(t["piece_idx"])
        auxes.append(t["aux"])
        masks.append(legal_move_mask(board))
        move_indices.append(move_to_index(move, board))
        board.push(move)

    T = len(move_indices)
    print(f"plies: {T}")

    piece_idx = torch.stack(piece_idxs).unsqueeze(0).to(device)   # (1, T, 64)
    aux = torch.stack(auxes).unsqueeze(0).to(device)              # (1, T, 7)
    pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)

    with torch.no_grad():
        policy_logits, value_pred = model(piece_idx, aux, pad_mask)

    for t in range(T):
        m = masks[t].to(device)                       # (4672,) bool
        logits = policy_logits[0, t]
        masked = logits.masked_fill(~m, float("-inf"))
        probs = F.softmax(masked, dim=-1)

        assert not torch.isnan(probs).any(), f"ply {t}: NaN in probs"
        s = probs.sum().item()
        assert abs(s - 1.0) < 1e-5, f"ply {t}: probs sum = {s}"
        illegal_mass = probs[~m].sum().item()
        assert illegal_mass < 1e-6, f"ply {t}: illegal mass = {illegal_mass}"
        played = move_indices[t]
        assert probs[played].item() > 0.0, f"ply {t}: played move has zero prob"

    print("all per-ply policy checks passed")

    print("value predictions:")
    for t in range(0, T, 10):
        print(f"  ply {t:2d}: {value_pred[0, t, 0].item():+.4f}")
    if (T - 1) % 10 != 0:
        print(f"  ply {T - 1:2d}: {value_pred[0, T - 1, 0].item():+.4f}")


if __name__ == "__main__":
    main()
