from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import random
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
import chess.pgn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kibitzer.data import collate_positions, dense_policy_from_scores
from kibitzer.encoding import board_to_tensor, legal_move_mask, move_to_index
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer
from kibitzer.stockfish import analyze_actions


_worker_engine: chess.engine.SimpleEngine | None = None
_worker_depth = 0
_worker_multipv = 1


def _start_stockfish_worker(path: str, depth: int, multipv: int) -> None:
    global _worker_engine, _worker_depth, _worker_multipv
    _worker_engine = chess.engine.SimpleEngine.popen_uci(path)
    _worker_depth = depth
    _worker_multipv = multipv


def _label_fen(fen: str) -> tuple[dict[int, float], float]:
    if _worker_engine is None:
        raise RuntimeError("Stockfish worker was not initialized")
    return analyze_actions(
        _worker_engine,
        chess.Board(fen),
        depth=_worker_depth,
        multipv=_worker_multipv,
    )


def iter_az_positions(paths: list[Path], *, max_positions: int | None) -> Iterator[dict[str, object]]:
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                # az data already has the final self-play outcome; keep it so we can
                # catch positions where the local game result disagrees with sf.
                yield {
                    "fen": payload["fen"],
                    "source": str(path),
                    "outcome_value": payload.get("value"),
                }
                seen += 1
                if max_positions is not None and seen >= max_positions:
                    return


def iter_pgn_positions(
    paths: list[Path],
    *,
    min_ply: int,
    stride: int,
    max_positions: int | None,
) -> Iterator[dict[str, object]]:
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while game := chess.pgn.read_game(handle):
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= min_ply and (ply - min_ply) % stride == 0:
                        yield {"fen": board.fen(), "source": str(path), "outcome_value": None}
                        seen += 1
                        if max_positions is not None and seen >= max_positions:
                            return
                    board.push(move)


def unique_records(records: Iterator[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    seen: set[str] = set()
    for record in records:
        fen = str(record["fen"])
        if fen in seen:
            continue
        seen.add(fen)
        out.append(record)
    return out


def label_fens(
    fens: list[str],
    *,
    stockfish_path: str,
    depth: int,
    multipv: int,
    workers: int,
) -> list[tuple[dict[int, float], float]]:
    if workers == 1:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            return [
                analyze_actions(engine, chess.Board(fen), depth=depth, multipv=multipv)
                for fen in tqdm(fens, desc="label stockfish")
            ]

    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_start_stockfish_worker,
        initargs=(stockfish_path, depth, multipv),
    ) as pool:
        labels = pool.imap(_label_fen, fens, chunksize=4)
        return list(tqdm(labels, total=len(fens), desc="label stockfish"))


def policy_top_move(evaluator: ModelEvaluator, fen: str) -> tuple[int, str]:
    board = chess.Board(fen)
    priors = evaluator.evaluate(board).priors
    move = max(priors, key=priors.get)
    return move_to_index(move, board), move.uci()


def labeled_regret(
    action_scores: dict[int, float],
    chosen_index: int | None,
) -> float:
    if not action_scores:
        return 0.0
    best = max(action_scores.values())
    floor = min(action_scores.values())
    # multipv only labels the teacher's top moves. if our policy picked something
    # outside that set, treat it as at least as bad as the worst labeled move.
    chosen = action_scores.get(chosen_index, floor) if chosen_index is not None else floor
    return max(0.0, best - chosen)


def summarize_values(values: list[float]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)

    def pct(q: float) -> float:
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[index]

    mean = sum(ordered) / len(ordered)
    return (
        f"n={len(ordered):,} mean={mean:.3f} "
        f"p50={pct(0.50):.3f} p90={pct(0.90):.3f} max={ordered[-1]:.3f}"
    )


def command_label(args: argparse.Namespace) -> None:
    print("[1/3] REGRET BUFFER CONFIG", flush=True)
    print(f"  az jsonl:        {len(args.az_jsonl or [])}")
    print(f"  pgn files:       {len(args.pgn or [])}")
    print(f"  checkpoint:      {args.checkpoint or 'none; outcome-gap filter only'}")
    print(f"  stockfish:       {args.stockfish_path} depth={args.depth} multipv={args.multipv}")
    print(f"  workers:         {args.stockfish_workers}")
    print(f"  max positions:   {args.max_positions or 'all'}")
    print(f"  filters:         regret>={args.min_regret:g} or outcome_gap>={args.min_outcome_gap:g}")
    sources: list[dict[str, object]] = []
    if args.az_jsonl:
        sources.extend(unique_records(iter_az_positions(args.az_jsonl, max_positions=args.max_positions)))
    if args.pgn:
        remaining = None if args.max_positions is None else max(0, args.max_positions - len(sources))
        sources.extend(
            unique_records(
                iter_pgn_positions(
                    args.pgn,
                    min_ply=args.min_ply,
                    stride=args.position_stride,
                    max_positions=remaining,
                )
            )
        )
    if not sources:
        raise SystemExit("no candidate positions found")
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(sources)
    if args.max_positions is not None:
        sources = sources[: args.max_positions]

    print("\n[2/3] STOCKFISH LABELING", flush=True)
    print(f"  candidate fens:  {len(sources):,}")
    fens = [str(item["fen"]) for item in sources]
    labels = label_fens(
        fens,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        multipv=args.multipv,
        workers=args.stockfish_workers,
    )
    evaluator = (
        ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
        if args.checkpoint is not None
        else None
    )

    print("\n[3/3] FILTER HIGH-REGRET POSITIONS", flush=True)
    kept = []
    regret_values = []
    outcome_gaps = []
    kept_regret = 0
    kept_outcome = 0
    kept_both = 0
    unlabeled = 0
    for source, (action_scores, teacher_value) in zip(sources, labels, strict=True):
        if not action_scores:
            unlabeled += 1
            continue
        chosen_index = None
        chosen_uci = None
        if evaluator is not None:
            chosen_index, chosen_uci = policy_top_move(evaluator, str(source["fen"]))
        regret = labeled_regret(action_scores, chosen_index)
        regret_values.append(regret)
        outcome_value = source.get("outcome_value")
        outcome_gap = (
            abs(float(teacher_value) - float(outcome_value))
            if outcome_value is not None
            else 0.0
        )
        outcome_gaps.append(outcome_gap)
        # keep only positions that can actually teach us something: either the
        # model's preferred move is bad, or az's outcome target disagrees with sf.
        regret_hit = regret >= args.min_regret
        outcome_hit = outcome_gap >= args.min_outcome_gap
        if not regret_hit and not outcome_hit:
            continue
        kept_regret += int(regret_hit)
        kept_outcome += int(outcome_hit)
        kept_both += int(regret_hit and outcome_hit)
        kept.append(
            {
                "fen": source["fen"],
                "source": source["source"],
                "teacher_value": teacher_value,
                "action_scores": {str(index): value for index, value in action_scores.items()},
                "policy_move": chosen_uci,
                "policy_regret": regret,
                "outcome_value": outcome_value,
                "outcome_gap": outcome_gap,
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")
    print(f"  unlabeled:       {unlabeled:,}")
    print(f"  policy regret:   {summarize_values(regret_values)}")
    print(f"  outcome gap:     {summarize_values(outcome_gaps)}")
    print(f"  kept:            {len(kept):,}/{len(sources):,}")
    print(f"  kept by regret:  {kept_regret:,}")
    print(f"  kept by outcome: {kept_outcome:,}")
    print(f"  kept by both:    {kept_both:,}")
    print(f"  output:          {args.out_jsonl}")


class RegretDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[dict[str, object]], *, temperature: float) -> None:
        self.records = records
        self.temperature = temperature

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        board = chess.Board(str(record["fen"]))
        encoded = board_to_tensor(board)
        action_scores = {
            int(index): float(value)
            for index, value in dict(record["action_scores"]).items()
        }
        return {
            "piece_idx": encoded["piece_idx"],
            "aux": encoded["aux"],
            "policy_target": dense_policy_from_scores(action_scores, self.temperature),
            "value_target": torch.tensor(float(record["teacher_value"]), dtype=torch.float32),
            "legal_mask": legal_move_mask(board),
        }


def configure_trainable(model: Kibitzer, *, unfreeze_last_trunk_blocks: int) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # default repair is intentionally small: let the heads and final norm adapt
    # before risking trunk drift. unfreeze trunk blocks only for a second pass.
    modules: list[torch.nn.Module] = [model.policy_head, model.norm, model.value_head]
    if unfreeze_last_trunk_blocks:
        if unfreeze_last_trunk_blocks > len(model.trunk):
            raise SystemExit(
                f"cannot unfreeze {unfreeze_last_trunk_blocks} trunk blocks; "
                f"model has {len(model.trunk)}"
            )
        modules.extend(model.trunk[-unfreeze_last_trunk_blocks:])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def anchored_loss(
    model: Kibitzer,
    reference: Kibitzer,
    batch: dict[str, torch.Tensor],
    *,
    value_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, value = model(batch["piece_idx"], batch["aux"])
    legal = batch["legal_mask"]
    logits = logits.masked_fill(~legal, -1e9)
    target = batch["policy_target"]
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = -(target * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(value.squeeze(-1), batch["value_target"])
    with torch.no_grad():
        reference_logits, _ = reference(batch["piece_idx"], batch["aux"])
        reference_logits = reference_logits.masked_fill(~legal, -1e9)
        reference_logp = F.log_softmax(reference_logits, dim=-1)
        reference_p = reference_logp.exp()
    # base anchor keeps the repair from becoming another sibling-overfit model:
    # it can move toward sf top-k, but pays for forgetting the original policy.
    anchor_kl = (reference_p * (reference_logp - logp)).sum(dim=-1).mean()
    loss = policy_loss + value_weight * value_loss + anchor_weight * anchor_kl
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "anchor_kl": anchor_kl.detach(),
    }


@torch.no_grad()
def evaluate(
    model: Kibitzer,
    reference: Kibitzer,
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "anchor_kl": 0.0}
    top1 = 0
    positions = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss, metrics = anchored_loss(
            model,
            reference,
            batch,
            value_weight=1.0,
            anchor_weight=0.0,
        )
        totals["loss"] += float(loss.item())
        for key in ("policy_loss", "value_loss", "anchor_kl"):
            totals[key] += float(metrics[key].item())
        logits, _ = model(batch["piece_idx"], batch["aux"])
        logits = logits.masked_fill(~batch["legal_mask"], -1e9)
        top1 += int((logits.argmax(dim=-1) == batch["policy_target"].argmax(dim=-1)).sum().item())
        positions += batch["policy_target"].shape[0] * batch["policy_target"].shape[1]
    n = max(1, len(loader))
    return {
        "loss": totals["loss"] / n,
        "policy_loss": totals["policy_loss"] / n,
        "value_loss": totals["value_loss"] / n,
        "anchor_kl": totals["anchor_kl"] / n,
        "policy_top1": top1 / max(1, positions),
    }


def load_records(pattern: str) -> list[dict[str, object]]:
    records = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no regret records matched {pattern!r}")
    return records


def command_train(args: argparse.Namespace) -> None:
    print("[1/4] LOAD BASE CHECKPOINT", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    reference = Kibitzer(payload["config"]).to(args.device)
    reference.load_state_dict(payload["model"])
    reference.eval().requires_grad_(False)
    trainable = configure_trainable(
        model,
        unfreeze_last_trunk_blocks=args.unfreeze_last_trunk_blocks,
    )

    print("\n[2/4] LOAD REGRET BUFFER", flush=True)
    records = load_records(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    split = max(1, int(len(records) * (1.0 - args.eval_fraction)))
    train_records = records[:split]
    eval_records = records[split:] or records[: min(len(records), args.batch_size)]
    print(f"  data pattern:    {args.data}")
    print(f"  records:         {len(records):,}")
    print(f"  train/eval:      {len(train_records):,}/{len(eval_records):,}")
    print(f"  teacher value:   {summarize_values([float(r['teacher_value']) for r in records])}")
    print(f"  policy regret:   {summarize_values([float(r.get('policy_regret', 0.0)) for r in records])}")
    print(f"  outcome gap:     {summarize_values([float(r.get('outcome_gap', 0.0)) for r in records])}")
    train_loader = DataLoader(
        RegretDataset(train_records, temperature=args.temperature),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_positions,
    )
    eval_loader = DataLoader(
        RegretDataset(eval_records, temperature=args.temperature),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_positions,
    )
    print("\n[3/4] TRAINING CONFIG", flush=True)
    print(f"  trainable params:{sum(p.numel() for p in trainable):,}")
    print(f"  unfreeze trunk:  {args.unfreeze_last_trunk_blocks}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch size:      {args.batch_size}")
    print(f"  lr:              {args.lr:g}")
    print(f"  temperature:     {args.temperature:g}")
    print(f"  weights:         policy=1 value={args.value_weight:g} anchor={args.anchor_weight:g}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")
    print("\n[4/4] TRAIN", flush=True)
    for epoch in range(args.epochs):
        model.train()
        sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "anchor_kl": 0.0}
        for batch in train_loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss, metrics = anchored_loss(
                model,
                reference,
                batch,
                value_weight=args.value_weight,
                anchor_weight=args.anchor_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            for key in sums:
                sums[key] += float(metrics[key].item())
        n = max(1, len(train_loader))
        eval_metrics = evaluate(model, reference, eval_loader, device=args.device)
        score = eval_metrics["policy_loss"] + args.value_weight * eval_metrics["value_loss"]
        improved = score < best_score
        if score < best_score:
            best_score = score
            best_metrics = eval_metrics
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "eval_metrics": eval_metrics,
                    "training_objective": "regret_guided_teacher_repair",
                    "best_epoch": epoch + 1,
                },
                args.out,
            )
        print(
            f"epoch {epoch+1}: train policy {sums['policy_loss']/n:.4f} "
            f"value {sums['value_loss']/n:.4f} anchor {sums['anchor_kl']/n:.4f} | "
            f"eval policy {eval_metrics['policy_loss']:.4f} "
            f"value {eval_metrics['value_loss']:.4f} top1 {eval_metrics['policy_top1']:.3f} "
            f"score {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
    print(f"best checkpoint:   {args.out}")
    print(f"best metrics:      {best_metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regret-guided teacher repair for Kibitzer.")
    sub = parser.add_subparsers(dest="mode", required=True)

    label = sub.add_parser("label")
    label.add_argument("--az-jsonl", type=Path, action="append")
    label.add_argument("--pgn", type=Path, action="append")
    label.add_argument("--checkpoint", type=Path, help="Base checkpoint used to measure policy regret.")
    label.add_argument("--out-jsonl", type=Path, required=True)
    label.add_argument("--stockfish-path", default="stockfish")
    label.add_argument("--stockfish-workers", type=int, default=min(8, multiprocessing.cpu_count()))
    label.add_argument("--depth", type=int, default=12)
    label.add_argument("--multipv", type=int, default=8)
    label.add_argument("--max-positions", type=int)
    label.add_argument("--min-regret", type=float, default=0.20)
    label.add_argument("--min-outcome-gap", type=float, default=0.75)
    label.add_argument("--min-ply", type=int, default=8)
    label.add_argument("--position-stride", type=int, default=4)
    label.add_argument("--shuffle", action="store_true")
    label.add_argument("--seed", type=int, default=0)
    label.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    train = sub.add_parser("train")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--eval-fraction", type=float, default=0.1)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--value-weight", type=float, default=0.25)
    train.add_argument("--anchor-weight", type=float, default=0.5)
    train.add_argument("--unfreeze-last-trunk-blocks", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "label":
        command_label(args)
    elif args.mode == "train":
        command_train(args)


if __name__ == "__main__":
    main()
