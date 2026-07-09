from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kibitzer.data import result_to_value
from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer
from kibitzer.search import puct_search


def sample_move(visits: dict[chess.Move, int], rng: random.Random, temperature: float) -> chess.Move:
    moves = list(visits)
    counts = [visits[move] for move in moves]
    if temperature <= 1e-3 or sum(counts) == 0:
        return moves[max(range(len(moves)), key=lambda idx: counts[idx])]
    weights = [count ** (1.0 / temperature) for count in counts]
    total = sum(weights) or 1.0
    threshold = rng.random() * total
    running = 0.0
    for move, weight in zip(moves, weights):
        running += weight
        if running >= threshold:
            return move
    return moves[-1]


def load_start_records(pattern: str, *, max_starts: int | None, min_regret: float) -> list[dict[str, object]]:
    rows = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if float(row.get("policy_regret", 0.0)) < min_regret:
                    continue
                rows.append(row)
                if max_starts is not None and len(rows) >= max_starts:
                    return rows
    if not rows:
        raise SystemExit(f"no start records matched {pattern!r} with regret >= {min_regret:g}")
    return rows


def command_gen(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    starts = load_start_records(
        args.starts,
        max_starts=args.max_starts,
        min_regret=args.min_regret,
    )
    if args.shuffle:
        rng.shuffle(starts)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("[1/2] REGRET-START SELF-PLAY CONFIG", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  starts:          {len(starts):,} from {args.starts}")
    print(f"  sims:            {args.sims}")
    print(f"  continuation:    {args.plies} plies")
    print(f"  temp plies:      {args.temp_plies}")
    print(f"  min regret:      {args.min_regret:g}")
    print(f"  value target:    {args.value_target}")
    print(f"  output:          {args.out_jsonl}")

    total_positions = 0
    terminal_count = 0
    start_time = time.monotonic()
    with args.out_jsonl.open("w", encoding="utf-8") as output:
        print("\n[2/2] GENERATE CONTINUATIONS", flush=True)
        for idx, start in enumerate(starts):
            board = chess.Board(str(start["fen"]))
            start_turn = board.turn
            records = []
            root_values = []
            plies = 0
            while not board.is_game_over(claim_draw=True) and plies < args.plies:
                result = puct_search(
                    board,
                    evaluator,
                    simulations=args.sims,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_epsilon=args.dirichlet_epsilon,
                )
                records.append(
                    {
                        "fen": board.fen(),
                        "turn": board.turn,
                        "visits": {move.uci(): count for move, count in result.visits.items()},
                        "root_value": result.root_value,
                    }
                )
                root_values.append(result.root_value)
                temperature = args.temperature if plies < args.temp_plies else 0.0
                board.push(sample_move(result.visits, rng, temperature))
                plies += 1

            outcome = board.outcome(claim_draw=True)
            result_text = outcome.result() if outcome is not None else "1/2-1/2"
            terminal = outcome is not None
            terminal_count += int(terminal)
            for record in records:
                outcome_value = result_to_value(result_text, bool(record["turn"]))
                root_value = float(record["root_value"])
                if args.value_target == "outcome":
                    value = outcome_value
                elif args.value_target == "mixed":
                    value = outcome_value if terminal else root_value
                else:
                    value = root_value
                output.write(
                    json.dumps(
                        {
                            "fen": record["fen"],
                            "visits": record["visits"],
                            "value": value,
                            "root_value": root_value,
                            "outcome_value": outcome_value,
                            "result": result_text,
                            "start_fen": start["fen"],
                            "start_policy_regret": start.get("policy_regret"),
                            "start_teacher_value": start.get("teacher_value"),
                        }
                    )
                    + "\n"
                )
            output.flush()
            total_positions += len(records)
            elapsed = time.monotonic() - start_time
            eta = (elapsed / (idx + 1)) * (len(starts) - idx - 1)
            print(
                f"[gen {idx+1}/{len(starts)}] {plies} plies  {len(records)} pos  "
                f"total {total_positions} pos  terminal={terminal} result={result_text}  "
                f"{elapsed/60:.1f}m elapsed  ~{eta/60:.1f}m left",
                flush=True,
            )
    elapsed = time.monotonic() - start_time
    print(
        f"[gen] done: {len(starts):,} starts, {total_positions:,} positions, "
        f"{terminal_count:,} terminal continuations, {elapsed/60:.1f} min",
        flush=True,
    )


class VisitDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        board = chess.Board(str(row["fen"]))
        encoded = board_to_tensor(board)
        total = sum(dict(row["visits"]).values()) or 1
        move_indices = []
        move_probs = []
        for uci, count in dict(row["visits"]).items():
            move_indices.append(move_to_index(chess.Move.from_uci(uci), board))
            move_probs.append(float(count) / total)
        return {
            "piece_idx": encoded["piece_idx"],
            "aux": encoded["aux"],
            "legal_mask": legal_move_mask(board),
            "target_idx": torch.tensor(move_indices, dtype=torch.long),
            "target_prob": torch.tensor(move_probs, dtype=torch.float32),
            "value": torch.tensor(float(row["value"]), dtype=torch.float32),
        }


def collate_visits(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    target = torch.zeros(batch_size, ACTION_SIZE, dtype=torch.float32)
    for idx, item in enumerate(batch):
        target[idx, item["target_idx"]] = item["target_prob"]
    return {
        "piece_idx": torch.stack([item["piece_idx"] for item in batch]).unsqueeze(1),
        "aux": torch.stack([item["aux"] for item in batch]).unsqueeze(1),
        "legal_mask": torch.stack([item["legal_mask"] for item in batch]),
        "target_policy": target,
        "value": torch.stack([item["value"] for item in batch]),
    }


def configure_trainable(model: Kibitzer, *, unfreeze_last_trunk_blocks: int) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # keep this conservative: targeted self-play can still overfit its own local
    # search quirks, so only widen the trunk after a gate proves useful.
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


def anchored_visit_loss(
    model: Kibitzer,
    reference: Kibitzer,
    batch: dict[str, torch.Tensor],
    *,
    value_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, value = model(batch["piece_idx"], batch["aux"])
    legal = batch["legal_mask"]
    logits = logits[:, -1, :].masked_fill(~legal, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = -(batch["target_policy"] * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(value[:, -1, 0], batch["value"])
    with torch.no_grad():
        reference_logits, _ = reference(batch["piece_idx"], batch["aux"])
        reference_logits = reference_logits[:, -1, :].masked_fill(~legal, -1e9)
        reference_logp = F.log_softmax(reference_logits, dim=-1)
        reference_prob = reference_logp.exp()
    # same anti-forgetting anchor as regret repair, but now against search visits.
    anchor_kl = (reference_prob * (reference_logp - logp)).sum(dim=-1).mean()
    loss = policy_loss + value_weight * value_loss + anchor_weight * anchor_kl
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "anchor_kl": anchor_kl.detach(),
    }


def load_visit_rows(pattern: str) -> list[dict[str, object]]:
    rows = []
    for path_text in sorted(glob.glob(pattern)):
        with open(path_text, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no continuation rows matched {pattern!r}")
    return rows


@torch.no_grad()
def evaluate(
    model: Kibitzer,
    reference: Kibitzer,
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "anchor_kl": 0.0}
    top1 = 0
    positions = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        _, metrics = anchored_visit_loss(
            model,
            reference,
            batch,
            value_weight=1.0,
            anchor_weight=0.0,
        )
        for key in totals:
            totals[key] += float(metrics[key].item())
        logits, _ = model(batch["piece_idx"], batch["aux"])
        logits = logits[:, -1, :].masked_fill(~batch["legal_mask"], -1e9)
        top1 += int((logits.argmax(dim=-1) == batch["target_policy"].argmax(dim=-1)).sum().item())
        positions += logits.shape[0]
    n = max(1, len(loader))
    return {
        "policy_loss": totals["policy_loss"] / n,
        "value_loss": totals["value_loss"] / n,
        "anchor_kl": totals["anchor_kl"] / n,
        "policy_top1": top1 / max(1, positions),
    }


def command_train(args: argparse.Namespace) -> None:
    rows = load_visit_rows(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split = max(1, int(len(rows) * (1.0 - args.eval_fraction)))
    train_rows = rows[:split]
    eval_rows = rows[split:] or rows[: min(len(rows), args.batch_size)]

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

    print("[1/2] REGRET-START TRAIN CONFIG", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  data:            {args.data}")
    print(f"  train/eval:      {len(train_rows):,}/{len(eval_rows):,}")
    print(f"  trainable params:{sum(parameter.numel() for parameter in trainable):,}")
    print(f"  epochs:          {args.epochs}")
    print(f"  weights:         policy=1 value={args.value_weight:g} anchor={args.anchor_weight:g}")
    print(f"  output:          {args.out}")

    train_loader = DataLoader(
        VisitDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_visits,
    )
    eval_loader = DataLoader(
        VisitDataset(eval_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_visits,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_metrics: dict[str, float] | None = None

    print("\n[2/2] TRAIN", flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "anchor_kl": 0.0}
        for batch in train_loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss, metrics = anchored_visit_loss(
                model,
                reference,
                batch,
                value_weight=args.value_weight,
                anchor_weight=args.anchor_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(metrics[key].item())
        n = max(1, len(train_loader))
        eval_metrics = evaluate(model, reference, eval_loader, device=args.device)
        score = eval_metrics["policy_loss"] + args.value_weight * eval_metrics["value_loss"]
        improved = score < best_score
        if improved:
            best_score = score
            best_metrics = eval_metrics
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "eval_metrics": eval_metrics,
                    "training_objective": "regret_start_selfplay",
                    "best_epoch": epoch + 1,
                },
                args.out,
            )
        print(
            f"epoch {epoch+1}: train policy {totals['policy_loss']/n:.4f} "
            f"value {totals['value_loss']/n:.4f} anchor {totals['anchor_kl']/n:.4f} | "
            f"eval policy {eval_metrics['policy_loss']:.4f} value {eval_metrics['value_loss']:.4f} "
            f"top1 {eval_metrics['policy_top1']:.3f} score {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
    print(f"best checkpoint:   {args.out}")
    print(f"best metrics:      {best_metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted AZ continuations from regret-buffer FENs.")
    sub = parser.add_subparsers(dest="mode", required=True)

    gen = sub.add_parser("gen")
    gen.add_argument("--checkpoint", type=Path, required=True)
    gen.add_argument("--starts", required=True)
    gen.add_argument("--out-jsonl", type=Path, required=True)
    gen.add_argument("--max-starts", type=int, default=1000)
    gen.add_argument("--min-regret", type=float, default=0.05)
    gen.add_argument("--sims", type=int, default=128)
    gen.add_argument("--plies", type=int, default=32)
    gen.add_argument("--temp-plies", type=int, default=8)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--dirichlet-alpha", type=float, default=0.3)
    gen.add_argument("--dirichlet-epsilon", type=float, default=0.15)
    gen.add_argument("--value-target", choices=("root", "outcome", "mixed"), default="root")
    gen.add_argument("--shuffle", action="store_true")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    train = sub.add_parser("train")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=3e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--value-weight", type=float, default=0.0)
    train.add_argument("--anchor-weight", type=float, default=0.75)
    train.add_argument("--eval-fraction", type=float, default=0.1)
    train.add_argument("--unfreeze-last-trunk-blocks", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "gen":
        command_gen(args)
    elif args.mode == "train":
        command_train(args)


if __name__ == "__main__":
    main()
