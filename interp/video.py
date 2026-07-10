# side-by-side video per game: the board playing on the left, the model's spatial
# attention (layer-3 encoder, mean over heads, attention received per square) on the
# right, synced to each of the model's moves. layer 3 is the encoder layer where the
# heads specialize, so its focus map is the most legible "what the net is looking at".

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter

from kibitzer.encoding import board_to_tensor
from kibitzer.model import Kibitzer

GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
         chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
LIGHT, DARK = "#efe6d4", "#b6885a"


def load_model(path, device):
    p = torch.load(path, map_location=device, weights_only=False)
    m = Kibitzer(p["config"]).to(device).eval()
    m.load_state_dict(p["model"])
    return m


@torch.no_grad()
def shaw_attn_received(block, x):
    # full shaw attention for one encoder block, reduced to attention received per key.
    h = block.attn_norm(x)
    b, s, _ = h.shape
    nh, hd = block.n_heads, block.head_dim
    q = block.q_proj(h).view(b, s, nh, hd).transpose(1, 2)
    k = block.k_proj(h).view(b, s, nh, hd).transpose(1, 2)
    bk = block.rel_buckets[:s, :s]
    a_q = block.rel_q(bk).view(s, s, nh, hd)
    a_k = block.rel_k(bk).view(s, s, nh, hd)
    qk = torch.einsum("bhid,bhjd->bhij", q, k)
    qk = qk + torch.einsum("bhid,ijhd->bhij", q, a_k)
    qk = qk + torch.einsum("bhjd,ijhd->bhij", k, a_q)
    qk = qk + torch.einsum("ijhd,ijhd->hij", a_q, a_k).unsqueeze(0)
    w = torch.softmax(qk / (hd**0.5), dim=-1)[0]  # [heads,64,64]
    return w.mean(0).mean(0).cpu().numpy()  # mean over heads then over queries -> [64]


@torch.no_grad()
def collect(model, game, model_white, device):
    grab = []
    h = model.position_encoder.blocks[2].register_forward_pre_hook(lambda m, a: grab.append(a[0].detach()))
    frames = []
    board = game.board()
    for move in game.mainline_moves():
        if board.turn == model_white:
            grab.clear()
            enc = board_to_tensor(board)
            piece = enc["piece_idx"].view(1, 1, -1).to(device)
            aux = enc["aux"].view(1, 1, -1).to(device)
            _, value = model(piece, aux)
            attn = shaw_attn_received(model.position_encoder.blocks[2], grab[0])
            frames.append((board.fen(), move.uci(), float(value[0, -1, 0]), attn))
        board.push(move)
    h.remove()
    return frames


def draw_board(ax, board: chess.Board, move: chess.Move) -> None:
    ax.clear()
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
    for sq in range(64):
        f, r = sq % 8, sq // 8
        ax.add_patch(plt.Rectangle((f, r), 1, 1, color=LIGHT if (f + r) % 2 else DARK))
    for sq in (move.from_square, move.to_square):
        f, r = sq % 8, sq // 8
        ax.add_patch(plt.Rectangle((f, r), 1, 1, color="#e6d84a", alpha=0.55))
    for sq, pc in board.piece_map().items():
        f, r = sq % 8, sq // 8
        col = "white" if pc.color == chess.WHITE else "#111111"
        stroke = "#111111" if pc.color == chess.WHITE else "#dddddd"
        ax.text(f + 0.5, r + 0.5, GLYPH[pc.piece_type], fontsize=22, ha="center", va="center",
                color=col, path_effects=[pe.withStroke(linewidth=1.4, foreground=stroke)])
    ax.set_title("board", fontsize=11)


def render(frames, title, out_path):
    fig, (axb, axh) = plt.subplots(1, 2, figsize=(10, 5.2))
    vmax = max(f[3].max() for f in frames)
    heat = axh.imshow(np.zeros((8, 8)), cmap="magma", vmin=0, vmax=vmax, origin="lower")
    fig.colorbar(heat, ax=axh, fraction=0.046, pad=0.04, label="attention received")
    axh.set_xticks(range(8)); axh.set_xticklabels(list("abcdefgh"))
    axh.set_yticks(range(8)); axh.set_yticklabels(range(1, 9))

    def update(i):
        fen, uci, val, attn = frames[i]
        b = chess.Board(fen); mv = chess.Move.from_uci(uci)
        draw_board(axb, b, mv)
        heat.set_data(attn.reshape(8, 8))
        axh.set_title(f"layer-3 attention", fontsize=11)
        fig.suptitle(f"{title}   |   ply {i+1}/{len(frames)}   value={val:+.2f}   move {uci}", fontsize=12)
        return heat,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
    anim.save(str(out_path), writer=FFMpegWriter(fps=2, bitrate=1800))
    plt.close(fig)
    print(f"[video] wrote {out_path} ({len(frames)} frames)", flush=True)


def pick_games(pgn_path, jsonl_path):
    recs = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    want = {1.0: "win", 0.5: "draw", 0.0: "loss"}
    chosen = {}
    with open(pgn_path) as fh:
        for rec in recs:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            lab = want.get(rec["score"])
            if lab and lab not in chosen:
                chosen[lab] = (g, bool(rec["network_white"]), rec["result"])
            if len(chosen) == 3:
                break
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/tactical/tactical_repair.pt")
    p.add_argument("--pgn", default="reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.pgn")
    p.add_argument("--jsonl", default="reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.jsonl")
    p.add_argument("--out-dir", default="interp/figures")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    for label, (game, mw, result) in pick_games(args.pgn, args.jsonl).items():
        frames = collect(model, game, mw, args.device)
        render(frames, f"{label.upper()} vs Leela-2700 ({result})", out / f"game_{label}.mp4")


if __name__ == "__main__":
    main()
