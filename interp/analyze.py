# mechanistic-interp pass over the base model (tactical_repair.pt). we do NOT make
# it play -- we replay existing leela-2700 gate games (one win, one draw, one loss,
# same opponent) and, at every position where it is the model's move, push the board
# through the net with hooks to record: how the board enters, the 64-square encoder
# attention (recomputed from each block's params -- SDPA discards weights), the
# layer-by-layer activation magnitudes, the value-head read, and behavioral features
# of what it wants to play (captures/checks/king-pressure = aggression).
#
# the model is single-position (ctx=1), so the 10 trunk layers see a length-1
# sequence and are near-idle: the real chess reasoning is the 3-layer encoder + the
# mean-pool over the 64 squares. that is the thing this dump is built to expose.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from kibitzer.encoding import board_to_tensor, move_to_index
from kibitzer.model import Kibitzer

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def load_model(path: str, device: str) -> Kibitzer:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Kibitzer(payload["config"]).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


# recompute the full shaw attention distribution for one RelativeEncoderBlock given
# its input x [1,64,d]. mirrors blocks.py: logits = (q.k^T + q.a_k + k.a_q + a_q.a_k)
# / sqrt(head_dim), then softmax over keys. returns [heads, 64, 64].
@torch.no_grad()
def shaw_attention(block, x: torch.Tensor) -> np.ndarray:
    h = block.attn_norm(x)
    b, s, _ = h.shape
    nh, hd = block.n_heads, block.head_dim
    q = block.q_proj(h).view(b, s, nh, hd).transpose(1, 2)
    k = block.k_proj(h).view(b, s, nh, hd).transpose(1, 2)
    buckets = block.rel_buckets[:s, :s]
    a_q = block.rel_q(buckets).view(s, s, nh, hd)
    a_k = block.rel_k(buckets).view(s, s, nh, hd)
    qk = torch.einsum("bhid,bhjd->bhij", q, k)
    qk = qk + torch.einsum("bhid,ijhd->bhij", q, a_k)
    qk = qk + torch.einsum("bhjd,ijhd->bhij", k, a_q)
    qk = qk + torch.einsum("ijhd,ijhd->hij", a_q, a_k).unsqueeze(0)
    logits = qk / (hd**0.5)
    return torch.softmax(logits, dim=-1)[0].cpu().numpy()  # [heads,64,64]


# behavioral features of the model's preferred move + policy shape at one position.
# "aggression" = how much policy mass sits on captures/checks and how close the top
# move lands to the enemy king.
@torch.no_grad()
def position_features(model, board: chess.Board, device: str) -> dict:
    enc = board_to_tensor(board)
    piece = enc["piece_idx"].view(1, 1, -1).to(device)
    aux = enc["aux"].view(1, 1, -1).to(device)
    logits, value = model(piece, aux)
    logits = logits[0, -1]
    legal = list(board.legal_moves)
    idx = torch.tensor([move_to_index(m, board) for m in legal], device=device)
    probs = torch.softmax(logits[idx].float(), dim=0).cpu().numpy()
    order = probs.argsort()[::-1]
    top = legal[int(order[0])]
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    cap_mass = float(sum(p for m, p in zip(legal, probs) if board.is_capture(m)))
    chk_mass = float(sum(p for m, p in zip(legal, probs) if board.gives_check(m)))
    ek = board.king(not board.turn)
    kd = None
    if ek is not None:
        kd = chess.square_distance(top.to_square, ek)
    mat = 0
    for sq, pc in board.piece_map().items():
        v = PIECE_VALUE[pc.piece_type]
        mat += v if pc.color == board.turn else -v
    return {
        "value": float(value[0, -1, 0]),
        "entropy": ent,
        "top_prob": float(probs[order[0]]),
        "top_move": top.uci(),
        "top_is_capture": bool(board.is_capture(top)),
        "top_gives_check": bool(board.gives_check(top)),
        "capture_mass": cap_mass,
        "check_mass": chk_mass,
        "king_dist_top": kd,
        "material": mat,
        "n_legal": len(legal),
    }


def analyze_game(model, game: chess.pgn.Game, model_white: bool, device: str) -> dict:
    # capture each encoder block's input via pre-hooks, and activation norms per stage.
    enc_inputs: list[torch.Tensor] = []
    act_norms: dict[str, float] = {}
    handles = []
    for i, blk in enumerate(model.position_encoder.blocks):
        handles.append(blk.register_forward_pre_hook(lambda m, a, i=i: enc_inputs.append(a[0].detach())))
        handles.append(blk.register_forward_hook(
            lambda m, a, o, i=i: act_norms.__setitem__(f"enc{i+1}", float(o.detach().norm(dim=-1).mean()))))
    handles.append(model.position_encoder.register_forward_hook(
        lambda m, a, o: act_norms.__setitem__("pool", float(o.detach().norm(dim=-1).mean()))))
    for i, blk in enumerate(model.trunk):
        handles.append(blk.register_forward_hook(
            lambda m, a, o, i=i: act_norms.__setitem__(f"trunk{i+1}", float(o.detach().norm(dim=-1).mean()))))

    board = game.board()
    n_layers, n_heads = len(model.position_encoder.blocks), model.position_encoder.blocks[0].n_heads
    attn_sum = np.zeros((n_layers, n_heads, 64, 64), dtype=np.float64)
    per_ply, act_trace, attn_samples = [], [], {}
    ply = 0
    for move in game.mainline_moves():
        if board.turn == model_white:
            enc_inputs.clear(); act_norms.clear()
            feat = position_features(model, board, device)
            attn = np.stack([shaw_attention(model.position_encoder.blocks[i], enc_inputs[i])
                             for i in range(n_layers)])  # [L,H,64,64]
            attn_sum += attn
            feat["ply"] = ply
            feat["fen"] = board.fen()
            feat["played_move"] = move.uci()
            feat["played_is_top"] = (move.uci() == feat["top_move"])
            per_ply.append(feat)
            act_trace.append(dict(act_norms))
            if ply in (2, 12, 24):  # early / middlegame / late samples
                attn_samples[str(ply)] = attn.astype(np.float32)
            ply += 1
        board.push(move)
    for h in handles:
        h.remove()
    attn_mean = (attn_sum / max(1, len(per_ply))).astype(np.float32)
    return {"per_ply": per_ply, "act_trace": act_trace,
            "attn_mean": attn_mean, "attn_samples": attn_samples,
            "n_layers": n_layers, "n_heads": n_heads}


def pick_games(pgn_path: str, jsonl_path: str) -> dict:
    records = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    wanted = {1.0: "win", 0.5: "draw", 0.0: "loss"}
    chosen: dict[str, tuple] = {}
    with open(pgn_path) as fh:
        for rec in records:
            g = chess.pgn.read_game(fh)
            if g is None:
                break
            label = wanted.get(rec["score"])
            if label and label not in chosen:
                chosen[label] = (g, bool(rec["network_white"]), rec["result"])
            if len(chosen) == 3:
                break
    return chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/tactical/tactical_repair.pt")
    p.add_argument("--pgn", default="reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.pgn")
    p.add_argument("--jsonl", default="reports/tactical_repair/tactical_repair_r1_vs2700_s128_g80_seed23.jsonl")
    p.add_argument("--out-dir", default="interp/data")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    chosen = pick_games(args.pgn, args.jsonl)
    summary = {}
    for label, (game, mw, result) in chosen.items():
        print(f"[interp] {label}: model_white={mw} result={result}", flush=True)
        data = analyze_game(model, game, mw, args.device)
        np.savez_compressed(out / f"{label}.npz",
                            attn_mean=data["attn_mean"],
                            **{f"attn_sample_{k}": v for k, v in data["attn_samples"].items()})
        (out / f"{label}_plies.json").write_text(json.dumps(data["per_ply"], indent=2))
        (out / f"{label}_acts.json").write_text(json.dumps(data["act_trace"], indent=2))
        vals = [f["value"] for f in data["per_ply"]]
        summary[label] = {
            "model_white": mw, "result": result, "n_model_plies": len(data["per_ply"]),
            "mean_value": float(np.mean(vals)) if vals else None,
            "final_value": vals[-1] if vals else None,
            "mean_capture_mass": float(np.mean([f["capture_mass"] for f in data["per_ply"]])),
            "mean_check_mass": float(np.mean([f["check_mass"] for f in data["per_ply"]])),
            "mean_entropy": float(np.mean([f["entropy"] for f in data["per_ply"]])),
            "top_capture_rate": float(np.mean([f["top_is_capture"] for f in data["per_ply"]])),
            "played_top_agree": float(np.mean([f["played_is_top"] for f in data["per_ply"]])),
            "n_layers": data["n_layers"], "n_heads": data["n_heads"],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[interp] wrote", out / "summary.json", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
