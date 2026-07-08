# enlarge the value head and retrain it (only) on the stockfish depth-14 label
# cache, freezing the trunk/encoder/policy. the 33k-param head is the known weak
# link (D50 alpha-beta collapse); this tests whether more value capacity + a fresh
# fit gives a leaf evaluator good enough to help search. run later.
#
# train:
#   uv run python scripts/train_value_head_big.py \
#     --init runs/scaling_shaw_data/checkpoints/S2_shaw_100M.pt \
#     --label-cache data/stockfish/joint_d14_mpv8_250000.pt \
#     --out runs/value_big/value_big.pt --value-hidden 256 --value-layers 3
#
# re-test search (does the better value head rescue alpha-beta?), from search_lab/:
#   uv run python compare.py --checkpoint ../runs/value_big/value_big.pt \
#     --variant alphabeta --baseline baseline_puct --budget 128 --games 20 \
#     --seed 7 --out results/alphabeta_bigvalue.json
#
# re-test play vs SF-1900:
#   uv run python scripts/eval_search_vs_stockfish.py \
#     --checkpoint runs/value_big/value_big.pt --out reports/value_big_sf1900.json \
#     --games 20 --simulations 64 --stockfish-elo 1900

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kibitzer.encoding import board_to_tensor
from kibitzer.model import Kibitzer, build_value_head


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--label-cache", type=Path, default=Path("data/stockfish/joint_d14_mpv8_250000.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--value-hidden", type=int, default=256)
    p.add_argument("--value-layers", type=int, default=3)
    p.add_argument("--unfreeze-final-norm", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# value target only; the cache label is (action_scores, value).
class ValueDataset(Dataset):
    def __init__(self, fens: list[str], values: list[float]) -> None:
        self.fens = fens
        self.values = values

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = board_to_tensor(chess.Board(self.fens[idx]))
        return {
            "piece_idx": enc["piece_idx"],
            "aux": enc["aux"],
            "value": torch.tensor(self.values[idx], dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "piece_idx": torch.stack([b["piece_idx"] for b in batch]),
        "aux": torch.stack([b["aux"] for b in batch]),
        "value": torch.stack([b["value"] for b in batch]),
    }


# game-disjoint split so held-out value error isn't leaked by same-game positions.
def split_by_game(samples: list, values: list[float], eval_fraction: float, seed: int):
    game_ids = sorted({s.game_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(game_ids)
    n_eval = max(1, round(len(game_ids) * eval_fraction))
    eval_ids = set(game_ids[:n_eval])
    tr_f, tr_v, ev_f, ev_v = [], [], [], []
    for s, v in zip(samples, values):
        if s.game_id in eval_ids:
            ev_f.append(s.fen); ev_v.append(v)
        else:
            tr_f.append(s.fen); tr_v.append(v)
    return tr_f, tr_v, ev_f, ev_v


@torch.no_grad()
def value_metrics(model: Kibitzer, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        piece = batch["piece_idx"].unsqueeze(1).to(device)
        aux = batch["aux"].unsqueeze(1).to(device)
        _, value = model(piece, aux)
        preds.append(value[:, -1, 0].cpu())
        tgts.append(batch["value"])
    pred = torch.cat(preds).float()
    tgt = torch.cat(tgts).float()
    err = pred - tgt
    mse = float(err.square().mean())
    mae = float(err.abs().mean())
    pc = pred - pred.mean()
    tc = tgt - tgt.mean()
    pearson = float((pc * tc).sum() / (pc.square().sum().sqrt() * tc.square().sum().sqrt()).clamp_min(1e-12))
    nz = tgt != 0
    sign = float((pred[nz].sign() == tgt[nz].sign()).float().mean()) if nz.any() else float("nan")
    return {"mse": mse, "mae": mae, "pearson": pearson, "sign": sign}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = torch.load(args.label_cache, map_location="cpu", weights_only=False)
    samples = cache["samples"]
    values = [float(v) for _, v in cache["labels"]]
    tr_f, tr_v, ev_f, ev_v = split_by_game(samples, values, args.eval_fraction, args.seed)
    print(f"cache {args.label_cache}: {len(samples)} positions -> train {len(tr_f)} / eval {len(ev_f)}")

    train_loader = DataLoader(ValueDataset(tr_f, tr_v), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(ValueDataset(ev_f, ev_v), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    payload = torch.load(args.init, map_location=args.device, weights_only=False)
    config = payload["config"]
    model = Kibitzer(config)
    model.load_state_dict(payload["model"])

    # baseline: the legacy head's held-out value error, for a fair before/after.
    model.to(args.device)
    base = value_metrics(model, eval_loader, args.device)
    print(f"legacy value head: mse={base['mse']:.4f} mae={base['mae']:.4f} "
          f"pearson={base['pearson']:.4f} sign={100*base['sign']:.2f}%")

    # enlarge + fresh-init the value head; record the arch in the config so every
    # loader (ModelEvaluator, search_lab) rebuilds it from the saved checkpoint.
    config.value_hidden = args.value_hidden
    config.value_layers = args.value_layers
    model.value_head = build_value_head(config.d_model, args.value_hidden, args.value_layers)
    model.config = config
    model.to(args.device)
    new_params = sum(p.numel() for p in model.value_head.parameters())
    print(f"enlarged value head: {new_params:,} params (hidden={args.value_hidden}, layers={args.value_layers})")

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(model.value_head.parameters())
    if args.unfreeze_final_norm:
        trainable += list(model.norm.parameters())
        for p in model.norm.parameters():
            p.requires_grad_(True)
    for p in model.value_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_mse = float("inf")
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            piece = batch["piece_idx"].unsqueeze(1).to(args.device)
            aux = batch["aux"].unsqueeze(1).to(args.device)
            target = batch["value"].to(args.device)
            _, value = model(piece, aux)
            loss = F.mse_loss(value[:, -1, 0], target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += float(loss.item()); n += 1
        m = value_metrics(model, eval_loader, args.device)
        print(f"epoch {epoch+1}/{args.epochs}: train_mse={total/max(1,n):.4f}  "
              f"eval mse={m['mse']:.4f} mae={m['mae']:.4f} pearson={m['pearson']:.4f} "
              f"sign={100*m['sign']:.2f}%  (legacy mse {base['mse']:.4f})")
        if m["mse"] < best_mse:
            best_mse = m["mse"]
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "config": model.config,
                        "training_objective": "value_head_big", "eval_metrics": m,
                        "legacy_value_metrics": base}, args.out)
            print(f"  saved new best -> {args.out}")

    print(f"done. best epoch {best_epoch}, eval mse {best_mse:.4f} vs legacy {base['mse']:.4f} "
          f"({'better' if best_mse < base['mse'] else 'not better'}); checkpoint {args.out}")


if __name__ == "__main__":
    main()
