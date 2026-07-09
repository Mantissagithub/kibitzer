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

from kibitzer.data import dense_policy_from_scores
from kibitzer.encoding import board_to_tensor, index_to_move, legal_move_mask, move_to_index
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


def iter_jsonl_positions(paths: list[Path], *, max_positions: int | None) -> Iterator[dict[str, object]]:
    seen = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                fen = payload.get("fen")
                if fen is None:
                    continue
                yield {"fen": fen, "source": str(path)}
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
                        yield {"fen": board.fen(), "source": str(path)}
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


def policy_ranked_indices(evaluator: ModelEvaluator, fen: str) -> list[tuple[int, str, float]]:
    board = chess.Board(fen)
    priors = evaluator.evaluate(board).priors
    ranked = sorted(priors.items(), key=lambda item: item[1], reverse=True)
    return [(move_to_index(move, board), move.uci(), float(prob)) for move, prob in ranked]


def choose_preference_pair(
    fen: str,
    action_scores: dict[int, float],
    policy_moves: list[tuple[int, str, float]],
    *,
    min_margin: float,
) -> dict[str, object] | None:
    if len(action_scores) < 2:
        return None
    board = chess.Board(fen)
    good_index, good_score = max(action_scores.items(), key=lambda item: item[1])
    floor = min(action_scores.values())
    bad_index = None
    bad_uci = None
    bad_score = floor
    bad_policy_prob = 0.0
    bad_score_is_floor = False

    # prefer the model's own tempting mistake; this is the policy-improvement signal.
    for index, uci, prob in policy_moves:
        if index == good_index:
            continue
        score = action_scores.get(index, floor)
        if good_score - score >= min_margin:
            bad_index = index
            bad_uci = uci
            bad_score = score
            bad_policy_prob = prob
            bad_score_is_floor = index not in action_scores
            break

    if bad_index is None:
        bad_index, bad_score = min(action_scores.items(), key=lambda item: item[1])
        if good_index == bad_index or good_score - bad_score < min_margin:
            return None
        bad_uci = index_to_move(bad_index, board).uci()

    good_uci = index_to_move(good_index, board).uci()
    return {
        "fen": fen,
        "good_index": good_index,
        "good_move": good_uci,
        "good_score": good_score,
        "bad_index": bad_index,
        "bad_move": bad_uci,
        "bad_score": bad_score,
        "teacher_margin": good_score - bad_score,
        "bad_policy_prob": bad_policy_prob,
        "bad_score_is_floor": bad_score_is_floor,
        "action_scores": {str(index): value for index, value in action_scores.items()},
    }


def command_label(args: argparse.Namespace) -> None:
    print("[1/4] PREFERENCE LABEL CONFIG", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  stockfish:       {args.stockfish_path} depth={args.depth} multipv={args.multipv}")
    print(f"  workers:         {args.stockfish_workers}")
    print(f"  jsonl sources:   {len(args.jsonl or [])}")
    print(f"  pgn sources:     {len(args.pgn or [])}")
    print(f"  max positions:   {args.max_positions or 'all'}")
    print(f"  min margin:      {args.min_margin:g}")

    sources: list[dict[str, object]] = []
    if args.jsonl:
        sources.extend(unique_records(iter_jsonl_positions(args.jsonl, max_positions=args.max_positions)))
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

    print("\n[2/4] STOCKFISH TEACHER LABELS", flush=True)
    print(f"  candidate fens:  {len(sources):,}")
    labels = label_fens(
        [str(item["fen"]) for item in sources],
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        multipv=args.multipv,
        workers=args.stockfish_workers,
    )

    print("\n[3/4] MODEL POLICY MISTAKES", flush=True)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    kept = []
    margins = []
    floor_pairs = 0
    skipped = 0
    for source, (action_scores, teacher_value) in tqdm(
        list(zip(sources, labels, strict=True)),
        desc="build pairs",
    ):
        fen = str(source["fen"])
        if not action_scores:
            skipped += 1
            continue
        try:
            policy_moves = policy_ranked_indices(evaluator, fen)
            pair = choose_preference_pair(
                fen,
                action_scores,
                policy_moves,
                min_margin=args.min_margin,
            )
        except (ValueError, AssertionError):
            skipped += 1
            continue
        if pair is None:
            skipped += 1
            continue
        pair["source"] = source["source"]
        pair["teacher_value"] = teacher_value
        pair["policy_top_move"] = policy_moves[0][1] if policy_moves else None
        pair["policy_top_prob"] = policy_moves[0][2] if policy_moves else 0.0
        kept.append(pair)
        margins.append(float(pair["teacher_margin"]))
        floor_pairs += int(bool(pair["bad_score_is_floor"]))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")

    print("\n[4/4] PREFERENCE BUFFER SUMMARY", flush=True)
    print(f"  kept pairs:      {len(kept):,}/{len(sources):,}")
    print(f"  skipped:         {skipped:,}")
    print(f"  floor bad score: {floor_pairs:,}")
    print(f"  margins:         {summarize_values(margins)}")
    print(f"  output:          {args.out_jsonl}")


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
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
            "good_index": torch.tensor(int(record["good_index"]), dtype=torch.long),
            "bad_index": torch.tensor(int(record["bad_index"]), dtype=torch.long),
            "teacher_margin": torch.tensor(float(record["teacher_margin"]), dtype=torch.float32),
            "policy_target": dense_policy_from_scores(action_scores, self.temperature),
            "legal_mask": legal_move_mask(board),
        }


def collate_preferences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "piece_idx": torch.stack([item["piece_idx"] for item in batch]).unsqueeze(1),
        "aux": torch.stack([item["aux"] for item in batch]).unsqueeze(1),
        "good_index": torch.stack([item["good_index"] for item in batch]).unsqueeze(1),
        "bad_index": torch.stack([item["bad_index"] for item in batch]).unsqueeze(1),
        "teacher_margin": torch.stack([item["teacher_margin"] for item in batch]).unsqueeze(1),
        "policy_target": torch.stack([item["policy_target"] for item in batch]).unsqueeze(1),
        "legal_mask": torch.stack([item["legal_mask"] for item in batch]).unsqueeze(1),
    }


def configure_trainable(model: Kibitzer, *, unfreeze_last_trunk_blocks: int) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[torch.nn.Module] = [model.policy_head, model.norm]
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


def preference_loss(
    model: Kibitzer,
    reference: Kibitzer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float,
    ce_weight: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, _ = model(batch["piece_idx"], batch["aux"])
    legal = batch["legal_mask"]
    logits = logits.masked_fill(~legal, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        reference_logits, _ = reference(batch["piece_idx"], batch["aux"])
        reference_logits = reference_logits.masked_fill(~legal, -1e9)
        reference_logp = F.log_softmax(reference_logits, dim=-1)
        reference_p = reference_logp.exp()

    gather_good = batch["good_index"].unsqueeze(-1)
    gather_bad = batch["bad_index"].unsqueeze(-1)
    good_logp = logp.gather(dim=-1, index=gather_good).squeeze(-1)
    bad_logp = logp.gather(dim=-1, index=gather_bad).squeeze(-1)
    ref_good_logp = reference_logp.gather(dim=-1, index=gather_good).squeeze(-1)
    ref_bad_logp = reference_logp.gather(dim=-1, index=gather_bad).squeeze(-1)

    # dpo compares how much the candidate shifts good-vs-bad odds over the frozen base.
    model_pair = good_logp - bad_logp
    reference_pair = ref_good_logp - ref_bad_logp
    dpo_logits = beta * (model_pair - reference_pair)
    dpo_loss = -F.logsigmoid(dpo_logits).mean()
    ce_loss = -(batch["policy_target"] * logp).sum(dim=-1).mean()
    anchor_kl = (reference_p * (reference_logp - logp)).sum(dim=-1).mean()
    loss = dpo_loss + ce_weight * ce_loss + anchor_weight * anchor_kl
    pair_acc = (model_pair > 0).float().mean()
    pair_margin = model_pair.mean()
    return loss, {
        "loss": loss.detach(),
        "dpo_loss": dpo_loss.detach(),
        "ce_loss": ce_loss.detach(),
        "anchor_kl": anchor_kl.detach(),
        "pair_acc": pair_acc.detach(),
        "pair_margin": pair_margin.detach(),
    }


@torch.no_grad()
def evaluate(
    model: Kibitzer,
    reference: Kibitzer,
    loader: DataLoader,
    *,
    device: str,
    beta: float,
    ce_weight: float,
    anchor_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "dpo_loss": 0.0,
        "ce_loss": 0.0,
        "anchor_kl": 0.0,
        "pair_acc": 0.0,
        "pair_margin": 0.0,
    }
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        _, metrics = preference_loss(
            model,
            reference,
            batch,
            beta=beta,
            ce_weight=ce_weight,
            anchor_weight=anchor_weight,
        )
        for key in totals:
            totals[key] += float(metrics[key].item())
    n = max(1, len(loader))
    return {key: value / n for key, value in totals.items()}


def load_records(pattern: str) -> list[dict[str, object]]:
    records = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no preference records matched {pattern!r}")
    return records


def command_train(args: argparse.Namespace) -> None:
    print("[1/5] LOAD REFERENCE CHECKPOINT", flush=True)
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

    print("\n[2/5] LOAD PREFERENCE BUFFER", flush=True)
    records = load_records(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    split = max(1, int(len(records) * (1.0 - args.eval_fraction)))
    train_records = records[:split]
    eval_records = records[split:] or records[: min(len(records), args.batch_size)]
    print(f"  data pattern:    {args.data}")
    print(f"  records:         {len(records):,}")
    print(f"  train/eval:      {len(train_records):,}/{len(eval_records):,}")
    print(f"  margins:         {summarize_values([float(r['teacher_margin']) for r in records])}")
    print(f"  floor bad score: {sum(1 for r in records if r.get('bad_score_is_floor')):,}")

    train_loader = DataLoader(
        PreferenceDataset(train_records, temperature=args.temperature),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_preferences,
    )
    eval_loader = DataLoader(
        PreferenceDataset(eval_records, temperature=args.temperature),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_preferences,
    )

    print("\n[3/5] TRAINING CONFIG", flush=True)
    print(f"  objective:       DPO pair loss + teacher CE + reference KL")
    print(f"  trainable params:{sum(p.numel() for p in trainable):,}")
    print(f"  unfreeze trunk:  {args.unfreeze_last_trunk_blocks}")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch size:      {args.batch_size}")
    print(f"  lr:              {args.lr:g}")
    print(f"  temperature:     {args.temperature:g}")
    print(f"  beta:            {args.beta:g}")
    print(f"  weights:         dpo=1 ce={args.ce_weight:g} anchor={args.anchor_weight:g}")

    print("\n[4/5] BASELINE HELD-OUT PAIRS", flush=True)
    baseline = evaluate(
        model,
        reference,
        eval_loader,
        device=args.device,
        beta=args.beta,
        ce_weight=args.ce_weight,
        anchor_weight=args.anchor_weight,
    )
    print(
        f"  pair_acc={baseline['pair_acc']:.3f} "
        f"pair_margin={baseline['pair_margin']:.3f} "
        f"ce={baseline['ce_loss']:.4f}",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")
    print("\n[5/5] TRAIN", flush=True)
    for epoch in range(args.epochs):
        model.train()
        sums = {
            "loss": 0.0,
            "dpo_loss": 0.0,
            "ce_loss": 0.0,
            "anchor_kl": 0.0,
            "pair_acc": 0.0,
            "pair_margin": 0.0,
        }
        for batch in train_loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss, metrics = preference_loss(
                model,
                reference,
                batch,
                beta=args.beta,
                ce_weight=args.ce_weight,
                anchor_weight=args.anchor_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            for key in sums:
                sums[key] += float(metrics[key].item())

        n = max(1, len(train_loader))
        eval_metrics = evaluate(
            model,
            reference,
            eval_loader,
            device=args.device,
            beta=args.beta,
            ce_weight=args.ce_weight,
            anchor_weight=args.anchor_weight,
        )
        score = eval_metrics["dpo_loss"] + args.ce_weight * eval_metrics["ce_loss"]
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
                    "training_objective": "teacher_preference_dpo_repair",
                    "best_epoch": epoch + 1,
                    "reference_checkpoint": str(args.checkpoint),
                },
                args.out,
            )
        print(
            f"epoch {epoch+1}: "
            f"train dpo {sums['dpo_loss']/n:.4f} ce {sums['ce_loss']/n:.4f} "
            f"kl {sums['anchor_kl']/n:.5f} acc {sums['pair_acc']/n:.3f} "
            f"margin {sums['pair_margin']/n:.3f} | "
            f"eval dpo {eval_metrics['dpo_loss']:.4f} ce {eval_metrics['ce_loss']:.4f} "
            f"kl {eval_metrics['anchor_kl']:.5f} acc {eval_metrics['pair_acc']:.3f} "
            f"margin {eval_metrics['pair_margin']:.3f} "
            f"score {score:.4f}{' *' if improved else ''}",
            flush=True,
        )

    print(f"best checkpoint:   {args.out}")
    print(f"best metrics:      {best_metrics}")
    print("next external gate:")
    print(
        "  CANDIDATE_NAME=preference_repair "
        "CANDIDATE_CHECKPOINT=runs/preference/preference_repair.pt "
        "CANDIDATE_REPORT_DIR=reports/preference_repair "
        "SEED=31 bash scripts/run_repair_eval_gate.sh"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-preference policy repair for Kibitzer.")
    sub = parser.add_subparsers(dest="mode", required=True)

    label = sub.add_parser("label")
    label.add_argument("--jsonl", type=Path, action="append")
    label.add_argument("--pgn", type=Path, action="append")
    label.add_argument("--checkpoint", type=Path, required=True)
    label.add_argument("--out-jsonl", type=Path, required=True)
    label.add_argument("--stockfish-path", default="stockfish")
    label.add_argument("--stockfish-workers", type=int, default=min(8, multiprocessing.cpu_count()))
    label.add_argument("--depth", type=int, default=12)
    label.add_argument("--multipv", type=int, default=8)
    label.add_argument("--max-positions", type=int)
    label.add_argument("--min-margin", type=float, default=0.08)
    label.add_argument("--min-ply", type=int, default=8)
    label.add_argument("--position-stride", type=int, default=6)
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
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--ce-weight", type=float, default=0.25)
    train.add_argument("--anchor-weight", type=float, default=0.05)
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
