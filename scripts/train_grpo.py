# grpo + exact-divergence dppo on an external verifiable reward. this is the D55
# bet: critic-free policy gradient (group baseline, no value head) on game
# outcomes vs a strength-capped stockfish ladder, fenced by an exact-tv dppo
# trust region over legal moves plus a weak kl to the frozen tactical base. it
# removes the two ingredients behind every prior failure -- self-generated
# targets and unfenced drift (see LOGBOOK.md, memory rl_grpo_dppo_plan_d55).
#
# gen:   sample groups of games vs a fixed stockfish elo, write per-ply records.
# train: one grpo+dppo pass over a fresh buffer, heads+norm only.
# probe: greedy-search score vs a held-out elo (cheap health check).
# loop:  gen -> train -> adaptive-ladder step, warm-starting each iteration.

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index
from kibitzer.grpo import dppo_mask, exact_tv, group_zscore
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer
from kibitzer.rollout import game_results, generate, informative_group_fraction, open_stockfish

# curated openings so both colors start from balanced, real positions. kept
# disjoint from the 20-line maia_gauntlet gate book on purpose.
OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6",
    "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 d2d4 g8f6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]


# build one spec per game: `groups` openings, G games each, alternating the
# model's color per group so wins aren't confounded with a color edge.
def build_specs(groups: int, group_size: int, rng: random.Random) -> list[tuple[str, bool, int, int]]:
    specs = []
    game_id = 0
    for gid in range(groups):
        opening = rng.choice(OPENING_BOOK)
        model_white = gid % 2 == 0
        for _ in range(group_size):
            specs.append((opening, model_white, gid, game_id))
            game_id += 1
    return specs


def gen(args: argparse.Namespace) -> tuple[int, int, int]:
    rng = random.Random(args.seed)
    random.seed(args.seed)  # puct_search dirichlet noise uses the module rng
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    engine = open_stockfish(args.stockfish, args.elo)
    try:
        specs = build_specs(args.groups, args.group_size, rng)
        records = generate(evaluator, engine, specs, sims=args.sims,
                           max_plies=args.max_plies, rng=rng, temp=args.temp,
                           temp_plies=args.temp_plies, temp_late=args.temp_late,
                           dirichlet_alpha=args.dirichlet_alpha, dirichlet_epsilon=args.dirichlet_epsilon,
                           engine_time=args.engine_time, log_prefix=f"gen elo{args.elo} s{args.sims}")
    finally:
        engine.quit()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")
    w, d, l = game_results(records)
    total = max(1, w + d + l)
    info = informative_group_fraction(records)
    print(f"[gen] elo {args.elo}: {w}W/{d}D/{l}L  score={(w + 0.5 * d) / total:.3f}  "
          f"{len(records)} pos  informative_groups={info:.0%} -> {args.out_jsonl}", flush=True)
    return w, d, l


# advantage is a per-game group z-score broadcast to that game's plies. we build
# it once from the buffer and hand each position its scalar advantage.
def advantages_by_game(records: list[dict]) -> dict[int, float]:
    reward, group = {}, {}
    for r in records:
        reward[r["game_id"]] = r["reward"]
        group[r["game_id"]] = r["group_id"]
    ids = sorted(reward)
    adv = group_zscore(torch.tensor([reward[i] for i in ids]),
                       torch.tensor([group[i] for i in ids]))
    return {i: float(a) for i, a in zip(ids, adv.tolist())}


class GRPODataset(Dataset):
    def __init__(self, records: list[dict], adv_by_game: dict[int, float]) -> None:
        self.records = records
        self.adv = adv_by_game

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        board = chess.Board(r["fen"])
        enc = board_to_tensor(board)
        # dense mu over the action space (softmax mass lives only on legal moves).
        mu = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        for uci, p in r["mu"].items():
            mu[move_to_index(chess.Move.from_uci(uci), board)] = p
        return {
            "piece_idx": enc["piece_idx"],
            "aux": enc["aux"],
            "legal_mask": legal_move_mask(board),
            "mu": mu,
            "action_idx": move_to_index(chess.Move.from_uci(r["action"]), board),
            "advantage": self.adv[r["game_id"]],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "piece_idx": torch.stack([x["piece_idx"] for x in batch]),
        "aux": torch.stack([x["aux"] for x in batch]),
        "legal_mask": torch.stack([x["legal_mask"] for x in batch]),
        "mu": torch.stack([x["mu"] for x in batch]),
        "action_idx": torch.tensor([x["action_idx"] for x in batch], dtype=torch.long),
        "advantage": torch.tensor([x["advantage"] for x in batch], dtype=torch.float32),
    }


# heads scope trains only the policy head + final norm -- the scope behind every
# past lift, and it keeps the frozen value head (and thus 128-sim gate search)
# anchored. full is an ablation.
def set_scope(model: Kibitzer, scope: str) -> list[torch.nn.Parameter]:
    if scope == "full":
        return list(model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.policy_head.parameters():
        p.requires_grad_(True)
    for p in model.norm.parameters():
        p.requires_grad_(True)
    return list(model.policy_head.parameters()) + list(model.norm.parameters())


def train(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    # frozen global anchor: tactical base, held fixed across all iterations so the
    # weak per-rollout anchor can't let drift accumulate iteration over iteration.
    anchor_payload = torch.load(args.anchor, map_location=args.device, weights_only=False)
    base = Kibitzer(anchor_payload["config"]).to(args.device)
    base.load_state_dict(anchor_payload["model"])
    base.eval().requires_grad_(False)

    records = []
    for f in sorted(glob.glob(args.data)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                records.append(json.loads(line))
    adv_by_game = advantages_by_game(records)
    trainable = set_scope(model, args.scope)
    tp = sum(p.numel() for p in trainable)
    print(f"[train] {len(records)} positions, dppo-tv delta={args.delta} beta={args.beta} "
          f"scope={args.scope} ({tp:,} params)", flush=True)

    loader = DataLoader(GRPODataset(records, adv_by_game), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        tpg = tkl = tmask = ttv = tbase = 0.0
        n = 0
        for batch in loader:
            piece = batch["piece_idx"].unsqueeze(1).to(args.device)
            aux = batch["aux"].unsqueeze(1).to(args.device)
            legal = batch["legal_mask"].to(args.device)
            mu = batch["mu"].to(args.device)
            action = batch["action_idx"].to(args.device)
            adv = batch["advantage"].to(args.device)

            logits, _ = model(piece, aux)
            logits = logits[:, -1, :].masked_fill(~legal, -1e9)
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            # importance ratio on the sampled action, relative to the rollout policy mu.
            pi_a = p.gather(1, action.unsqueeze(1)).squeeze(1)
            mu_a = mu.gather(1, action.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            ratio = pi_a / mu_a
            div = exact_tv(p, mu, legal)
            mask = dppo_mask(adv, ratio, div, args.delta)
            pg = -(mask * ratio * adv).sum() / mask.sum().clamp_min(1.0)
            # base-kl anchor: KL(pi || base) over legal moves keeps the policy from
            # wandering off the tactical base while it chases outcome reward.
            with torch.no_grad():
                blogits, _ = base(piece, aux)
                blogits = blogits[:, -1, :].masked_fill(~legal, -1e9)
                blogp = F.log_softmax(blogits, dim=-1)
            kl = ((p * (logp - blogp)) * legal).sum(dim=-1).mean()
            loss = pg + args.beta * kl
            # tv_to_base is the actual drift kill-switch metric (plan: watch p95);
            # tracked separately from tv_to_mu (per-step trust region) since base
            # stays frozen while mu is last iteration's rollout policy.
            tv_base = exact_tv(p.detach(), blogp.exp(), legal).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tpg += float(pg.item()); tkl += float(kl.item())
            tmask += float(mask.mean().item()); ttv += float(div.mean().item())
            tbase += float(tv_base.item())
            n += 1
        n = max(1, n)
        print(f"[train] epoch {epoch+1}: policy_grad {tpg/n:+.4f}  keep_rate {tmask/n:.0%}  "
              f"tv_to_mu {ttv/n:.4f}  tv_to_base {tbase/n:.4f}  base_kl {tkl/n:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config,
                "training_objective": "grpo_dppo_external"}, args.out)
    print(f"[train] saved {args.out}", flush=True)


def probe(args: argparse.Namespace) -> float:
    rng = random.Random(args.seed)
    random.seed(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    engine = open_stockfish(args.stockfish, args.elo)
    try:
        # greedy search, no dirichlet: a deterministic strength read at gate-like sims.
        specs = build_specs(args.games, 1, rng)
        records = generate(evaluator, engine, specs, sims=args.sims, max_plies=args.max_plies,
                           rng=rng, temp=0.0, temp_plies=0, temp_late=0.0,
                           dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
                           engine_time=args.engine_time, log_prefix=f"probe elo{args.elo} s{args.sims}")
    finally:
        engine.quit()
    w, d, l = game_results(records)
    total = max(1, w + d + l)
    score = (w + 0.5 * d) / total
    print(f"[probe] vs held-out elo {args.elo}: {w}W/{d}D/{l}L score={score:.3f}", flush=True)
    return score


# adaptive ladder: nudge the opponent toward the elo where the model scores ~50%,
# so groups stay informative (a group where all G games share a result has zero
# advantage). elo is clamped to the ladder bounds.
def next_elo(elo: int, score: float, lo: int, hi: int, step: int = 100) -> int:
    if score > 0.55:
        elo += step
    elif score < 0.45:
        elo -= step
    return max(lo, min(hi, elo))


def loop(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
    checkpoint = args.checkpoint  # warm-started each iteration
    elo = args.elo
    for it in range(1, args.iterations + 1):
        print("=" * 60, flush=True)
        print(f" iteration {it}/{args.iterations}   opponent elo {elo}   "
              f"model {checkpoint.name if isinstance(checkpoint, Path) else checkpoint}", flush=True)
        print("=" * 60, flush=True)
        data = out_dir / f"grpo_data_{it}.jsonl"
        ckpt = out_dir / f"grpo_v{it}.pt"
        gen_args = argparse.Namespace(
            checkpoint=checkpoint, stockfish=args.stockfish, elo=elo, groups=args.groups,
            group_size=args.group_size, sims=args.sims, temp=args.temp, temp_plies=args.temp_plies,
            temp_late=args.temp_late, dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon, max_plies=args.max_plies,
            engine_time=args.engine_time, seed=args.seed + it, out_jsonl=data, device=args.device)
        w, d, l = gen(gen_args)
        total = max(1, w + d + l)
        score = (w + 0.5 * d) / total
        train_args = argparse.Namespace(
            checkpoint=checkpoint, anchor=args.anchor, data=str(data), delta=args.delta,
            beta=args.beta, scope=args.scope, batch_size=args.batch_size, lr=args.lr,
            epochs=args.epochs, out=ckpt, device=args.device)
        train(train_args)
        checkpoint = ckpt
        probe_score = None
        if it % args.probe_every == 0:
            probe_args = argparse.Namespace(
                checkpoint=ckpt, stockfish=args.stockfish, elo=args.probe_elo, games=40,
                sims=args.probe_sims, max_plies=args.max_plies, engine_time=args.engine_time,
                seed=args.seed, device=args.device)
            probe_score = probe(probe_args)
        # move the ladder toward ~50% and say why, so the trend is readable live.
        # a clamp at a bound is called out separately from a genuine near-50% hold.
        new_elo = next_elo(elo, score, args.elo_lo, args.elo_hi)
        if new_elo > elo:
            reason = "raising (winning too easily)"
        elif new_elo < elo:
            reason = "lowering (opponent too strong)"
        elif score > 0.55:
            reason = "at ceiling (still winning)"
        elif score < 0.45:
            reason = "at floor (still losing)"
        else:
            reason = "holding (near 50%)"
        probe_txt = f"  probe@{args.probe_elo}={probe_score:.3f}" if probe_score is not None else ""
        print(f"[iter {it}] score={score:.3f}  ladder {elo}->{new_elo} {reason}{probe_txt}  "
              f"-> {ckpt.name}", flush=True)
        metrics.write(json.dumps({"iter": it, "elo": elo, "w": w, "d": d, "l": l,
                                 "score": score, "probe": probe_score}) + "\n")
        metrics.flush()
        elo = new_elo
    metrics.close()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = sub.add_parser("gen")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--stockfish", default="stockfish")
    g.add_argument("--elo", type=int, default=1900)
    g.add_argument("--groups", type=int, default=10)
    g.add_argument("--group-size", type=int, default=8)
    g.add_argument("--sims", type=int, default=128, help="puct sims per move; 0 = searchless raw")
    g.add_argument("--temp", type=float, default=0.8, help="visit-count temperature for the opening")
    g.add_argument("--temp-plies", type=int, default=16, help="plies played at --temp before going near-greedy")
    g.add_argument("--temp-late", type=float, default=0.0, help="visit-count temperature after --temp-plies")
    g.add_argument("--dirichlet-alpha", type=float, default=0.3)
    g.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    g.add_argument("--max-plies", type=int, default=200)
    g.add_argument("--engine-time", type=float, default=0.01)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out-jsonl", type=Path, required=True)
    g.add_argument("--device", default=dev)

    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--anchor", type=Path, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--delta", type=float, default=0.2)
    t.add_argument("--beta", type=float, default=0.05)
    t.add_argument("--scope", choices=["heads", "full"], default="heads")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default=dev)

    pr = sub.add_parser("probe")
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--stockfish", default="stockfish")
    pr.add_argument("--elo", type=int, default=2000)
    pr.add_argument("--games", type=int, default=40)
    pr.add_argument("--sims", type=int, default=128, help="gate-like search strength read")
    pr.add_argument("--max-plies", type=int, default=200)
    pr.add_argument("--engine-time", type=float, default=0.01)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--device", default=dev)

    lp = sub.add_parser("loop")
    lp.add_argument("--checkpoint", type=Path, required=True)
    lp.add_argument("--anchor", type=Path, required=True)
    lp.add_argument("--out-dir", type=Path, required=True)
    lp.add_argument("--iterations", type=int, default=30)
    lp.add_argument("--stockfish", default="stockfish")
    lp.add_argument("--elo", type=int, default=1900)
    lp.add_argument("--elo-lo", type=int, default=1600)
    lp.add_argument("--elo-hi", type=int, default=2600)
    lp.add_argument("--probe-elo", type=int, default=2000)
    lp.add_argument("--probe-every", type=int, default=5)
    lp.add_argument("--probe-sims", type=int, default=128)
    lp.add_argument("--groups", type=int, default=10)
    lp.add_argument("--group-size", type=int, default=8)
    lp.add_argument("--sims", type=int, default=128, help="puct sims per move in rollouts")
    lp.add_argument("--temp", type=float, default=0.8)
    lp.add_argument("--temp-plies", type=int, default=16)
    lp.add_argument("--temp-late", type=float, default=0.0)
    lp.add_argument("--dirichlet-alpha", type=float, default=0.3)
    lp.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    lp.add_argument("--max-plies", type=int, default=200)
    lp.add_argument("--engine-time", type=float, default=0.01)
    lp.add_argument("--delta", type=float, default=0.2)
    lp.add_argument("--beta", type=float, default=0.05)
    lp.add_argument("--scope", choices=["heads", "full"], default="heads")
    lp.add_argument("--batch-size", type=int, default=256)
    lp.add_argument("--lr", type=float, default=1e-5)
    lp.add_argument("--epochs", type=int, default=1)
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--device", default=dev)

    args = p.parse_args()
    {"gen": gen, "train": train, "probe": probe, "loop": loop}[args.mode](args)


if __name__ == "__main__":
    main()
