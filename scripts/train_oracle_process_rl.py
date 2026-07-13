# critic-free on-policy rl with dense stockfish process rewards. unlike regret
# repair, this updates the sampled move with a signed return-to-go instead of
# copying the teacher's best move. the real game result stays in every return.

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import random
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer
from kibitzer.oracle_rl import (
    add_returns_and_advantages,
    clipped_process_reward,
    configure_policy_scope,
    filter_training_records,
    oracle_policy_loss,
    signed_outcome,
)
from kibitzer.rollout import game_results, generate, open_stockfish
from kibitzer.stockfish import pov_score_to_value
from scripts.train_grpo import build_specs


_worker_engine: chess.engine.SimpleEngine | None = None
_worker_nodes = 0
_worker_multipv = 1
_worker_clip = 0.5


def _start_teacher_worker(path: str, nodes: int, multipv: int, clip: float) -> None:
    global _worker_engine, _worker_nodes, _worker_multipv, _worker_clip
    _worker_engine = chess.engine.SimpleEngine.popen_uci(path)
    _worker_nodes = nodes
    _worker_multipv = multipv
    _worker_clip = clip


def teacher_label(
    engine: chess.engine.SimpleEngine,
    record: dict,
    *,
    nodes: int,
    multipv: int,
    clip: float,
) -> dict:
    board = chess.Board(str(record["fen"]))
    chosen_move = chess.Move.from_uci(str(record["action"]))
    requested = min(multipv, board.legal_moves.count())
    infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=requested)
    if isinstance(infos, dict):
        infos = [infos]
    if not infos:
        raise RuntimeError(f"stockfish returned no analysis for {board.fen()}")

    action_scores: dict[int, float] = {}
    best_move: chess.Move | None = None
    best_value: float | None = None
    chosen_value: float | None = None
    for rank, info in enumerate(infos):
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        move = pv[0]
        value = pov_score_to_value(score, board.turn)
        action_scores[move_to_index(move, board)] = value
        if rank == 0:
            best_move = move
            best_value = value
        if move == chosen_move:
            chosen_value = value

    if best_move is None or best_value is None:
        raise RuntimeError(f"stockfish returned no scored pv for {board.fen()}")
    if chosen_value is None:
        chosen = engine.analyse(
            board,
            chess.engine.Limit(nodes=nodes),
            root_moves=[chosen_move],
        )
        score = chosen.get("score")
        if score is None:
            raise RuntimeError(f"stockfish returned no chosen-move score for {board.fen()}")
        chosen_value = pov_score_to_value(score, board.turn)
        action_scores[move_to_index(chosen_move, board)] = chosen_value

    labeled = dict(record)
    labeled.update(
        {
            "teacher_best_move": best_move.uci(),
            "teacher_best_value": best_value,
            "teacher_chosen_value": chosen_value,
            "teacher_action_scores": {str(index): value for index, value in action_scores.items()},
            "regret": max(0.0, best_value - chosen_value),
            "process_reward": clipped_process_reward(best_value, chosen_value, clip),
        }
    )
    return labeled


def _label_worker(record: dict) -> dict:
    if _worker_engine is None:
        raise RuntimeError("stockfish teacher worker was not initialized")
    return teacher_label(
        _worker_engine,
        record,
        nodes=_worker_nodes,
        multipv=_worker_multipv,
        clip=_worker_clip,
    )


def label_records(
    records: list[dict],
    *,
    stockfish: str,
    nodes: int,
    multipv: int,
    clip: float,
    workers: int,
) -> list[dict]:
    if workers == 1:
        with chess.engine.SimpleEngine.popen_uci(stockfish) as engine:
            return [
                teacher_label(engine, record, nodes=nodes, multipv=multipv, clip=clip)
                for record in tqdm(records, desc="stockfish process labels")
            ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_start_teacher_worker,
        initargs=(stockfish, nodes, multipv, clip),
    ) as pool:
        labels = pool.imap(_label_worker, records, chunksize=2)
        return list(tqdm(labels, total=len(records), desc="stockfish process labels"))


def summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)

    def percentile(q: float) -> float:
        index = round(q * (len(ordered) - 1))
        return ordered[max(0, min(len(ordered) - 1, index))]

    return (
        f"n={len(ordered):,} mean={sum(ordered) / len(ordered):+.3f} "
        f"p50={percentile(0.5):+.3f} p90={percentile(0.9):+.3f} "
        f"min={ordered[0]:+.3f} max={ordered[-1]:+.3f}"
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_generate(args: argparse.Namespace) -> None:
    print("[1/2] ON-POLICY ROLLOUT CONFIG", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  opponent:        Stockfish UCI_Elo {args.opponent_elo}")
    print(f"  groups/games:    {args.groups} x {args.group_size} = {args.groups * args.group_size}")
    print(f"  model search:    {args.sims} sims")
    print(f"  sampling:        temp {args.temp} through ply {args.temp_plies}, then {args.temp_late}")
    print(f"  max plies:       {args.max_plies}")
    print(f"  output:          {args.out_jsonl}")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    engine = open_stockfish(args.stockfish, args.opponent_elo)
    try:
        specs = build_specs(args.groups, args.group_size, rng)
        records = generate(
            evaluator,
            engine,
            specs,
            sims=args.sims,
            max_plies=args.max_plies,
            rng=rng,
            temp=args.temp,
            temp_plies=args.temp_plies,
            temp_late=args.temp_late,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            engine_time=args.engine_time,
            log_prefix=f"oracle rollout s{args.sims}",
        )
    finally:
        engine.quit()

    by_game: dict[int, int] = defaultdict(int)
    for record in records:
        game_id = int(record["game_id"])
        record["model_ply"] = by_game[game_id]
        record["terminal_reward"] = signed_outcome(float(record["reward"]))
        by_game[game_id] += 1
    write_jsonl(args.out_jsonl, records)
    wins, draws, losses = game_results(records)
    total = max(1, wins + draws + losses)
    print("\n[2/2] ROLLOUT SUMMARY", flush=True)
    print(f"  result:          {wins}W/{draws}D/{losses}L score={(wins + 0.5 * draws) / total:.3f}")
    print(f"  model positions: {len(records):,}")
    print(f"  output:          {args.out_jsonl}")


def command_label(args: argparse.Namespace) -> None:
    print("[1/3] PROCESS REWARD CONFIG", flush=True)
    print(f"  rollout:         {args.input_jsonl}")
    print(f"  teacher:         {args.stockfish} nodes={args.teacher_nodes:,} multipv={args.multipv}")
    print(f"  workers:         {args.workers}")
    print(f"  reward clip:     {args.reward_clip:g} value units")
    print(f"  return:          {args.process_weight:g}*process + {args.terminal_weight:g}*outcome, gamma={args.gamma:g}")

    records = read_jsonl(args.input_jsonl)
    if not records:
        raise SystemExit("on-policy rollout contains no model positions")
    print(f"\n[2/3] STOCKFISH LABELING ({len(records):,} POSITIONS)", flush=True)
    labeled = label_records(
        records,
        stockfish=args.stockfish,
        nodes=args.teacher_nodes,
        multipv=args.multipv,
        clip=args.reward_clip,
        workers=args.workers,
    )
    labeled = add_returns_and_advantages(
        labeled,
        gamma=args.gamma,
        process_weight=args.process_weight,
        terminal_weight=args.terminal_weight,
    )
    write_jsonl(args.out_jsonl, labeled)

    regrets = [float(record["regret"]) for record in labeled]
    process = [float(record["process_reward"]) for record in labeled]
    returns = [float(record["return"]) for record in labeled]
    advantages = [float(record["advantage"]) for record in labeled]
    print("\n[3/3] LABELED BUFFER SUMMARY", flush=True)
    print(f"  regret:          {summarize(regrets)}")
    print(f"  process reward:  {summarize(process)}")
    print(f"  return-to-go:    {summarize(returns)}")
    print(f"  advantages:      {summarize(advantages)}")
    print(f"  signed signal:   {sum(v > 0 for v in advantages):,} positive / {sum(v < 0 for v in advantages):,} negative")
    print(f"  output:          {args.out_jsonl}")


class OracleDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        board = chess.Board(str(record["fen"]))
        encoded = board_to_tensor(board)
        rollout_policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        for uci, probability in dict(record["mu"]).items():
            move = chess.Move.from_uci(str(uci))
            rollout_policy[move_to_index(move, board)] = float(probability)

        scores = {int(key): float(value) for key, value in dict(record["teacher_action_scores"]).items()}
        floor = min(scores.values())
        teacher_scores = torch.full((ACTION_SIZE,), floor, dtype=torch.float32)
        for action_index, value in scores.items():
            teacher_scores[action_index] = value
        return {
            "piece_idx": encoded["piece_idx"],
            "aux": encoded["aux"],
            "legal": legal_move_mask(board),
            "rollout_policy": rollout_policy,
            "action": torch.tensor(move_to_index(chess.Move.from_uci(str(record["action"])), board)),
            "advantage": torch.tensor(float(record["advantage"]), dtype=torch.float32),
            "teacher_scores": teacher_scores,
            "teacher_best": torch.tensor(float(record["teacher_best_value"]), dtype=torch.float32),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in batch]) for key in batch[0]}


def split_by_group(records: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = sorted({int(record["group_id"]) for record in records})
    if len(groups) < 2:
        raise SystemExit("oracle training needs at least two surviving rollout groups")
    random.Random(seed).shuffle(groups)
    eval_count = max(1, round(len(groups) * eval_fraction))
    eval_groups = set(groups[:eval_count])
    train = [record for record in records if int(record["group_id"]) not in eval_groups]
    evaluate = [record for record in records if int(record["group_id"]) in eval_groups]
    if not train or not evaluate:
        raise SystemExit("group-disjoint train/eval split produced an empty side")
    return train, evaluate


@torch.no_grad()
def evaluate_model(
    model: Kibitzer,
    anchor: Kibitzer,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        piece = batch["piece_idx"].unsqueeze(1)
        aux = batch["aux"].unsqueeze(1)
        logits, _ = model(piece, aux)
        base_logits, _ = anchor(piece, aux)
        logits = logits[:, -1, :].masked_fill(~batch["legal"], -1e9)
        base_logits = base_logits[:, -1, :].masked_fill(~batch["legal"], -1e9)
        logp = F.log_softmax(logits, dim=-1)
        base_logp = F.log_softmax(base_logits, dim=-1)
        policy = logp.exp()
        top = policy.argmax(dim=-1)
        top_score = batch["teacher_scores"].gather(1, top.unsqueeze(1)).squeeze(1)
        regret = (batch["teacher_best"] - top_score).clamp_min(0.0)
        sampled_prob = policy.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        sampled_logp = logp.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        expected_score = (policy * batch["teacher_scores"]).sum(dim=-1)
        expected_regret = (batch["teacher_best"] - expected_score).clamp_min(0.0)
        kl = (policy * (logp - base_logp)).sum(dim=-1)
        tv = 0.5 * (policy - base_logp.exp()).abs().sum(dim=-1)
        batch_size = piece.shape[0]
        totals["regret"] += float(regret.sum().item())
        totals["expected_regret"] += float(expected_regret.sum().item())
        totals["near_best"] += float((regret <= 0.05).float().sum().item())
        totals["sampled_prob"] += float(sampled_prob.sum().item())
        totals["signed_logprob"] += float((batch["advantage"] * sampled_logp).sum().item())
        totals["anchor_kl"] += float(kl.sum().item())
        totals["tv_base"] += float(tv.sum().item())
        count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def command_train(args: argparse.Namespace) -> None:
    print("[1/5] LOAD AND FILTER ORACLE BUFFER", flush=True)
    records = read_jsonl(args.data)
    kept = filter_training_records(
        records,
        min_regret=args.min_regret,
        min_abs_advantage=args.min_abs_advantage,
    )
    positive = sum(float(record["advantage"]) > 0 for record in kept)
    negative = sum(float(record["advantage"]) < 0 for record in kept)
    print(f"  labeled:         {len(records):,}")
    print(f"  kept:            {len(kept):,} with regret>={args.min_regret:g} and |adv|>={args.min_abs_advantage:g}")
    print(f"  signed signal:   {positive:,} positive / {negative:,} negative")
    if not kept:
        raise SystemExit("no meaningful-regret records survived filtering")
    if positive == 0 or negative == 0:
        raise SystemExit("filtered buffer needs both positive and negative advantages")
    train_records, eval_records = split_by_group(kept, args.eval_fraction, args.seed)
    print(f"  train/eval:      {len(train_records):,}/{len(eval_records):,} positions, group-disjoint")

    print("\n[2/5] LOAD MODEL AND FROZEN ANCHOR", flush=True)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    anchor_payload = torch.load(args.anchor, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    anchor = Kibitzer(anchor_payload["config"]).to(args.device)
    anchor.load_state_dict(anchor_payload["model"])
    anchor.eval().requires_grad_(False)
    trainable = configure_policy_scope(model)
    print(f"  init:            {args.checkpoint}")
    print(f"  anchor:          {args.anchor}")
    print(f"  trainable:       policy head + final norm ({sum(p.numel() for p in trainable):,} params)")
    print("  frozen:          position encoder + trunk + value head")

    train_loader = DataLoader(
        OracleDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        OracleDataset(eval_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    print("\n[3/5] TRAINING CONFIG", flush=True)
    print(f"  objective:       signed return policy gradient + exact base KL")
    print(f"  trust region:    DPPO exact-TV delta={args.delta:g}")
    print(f"  anchor:          beta={args.beta:g}, reject epoch above mean TV={args.max_tv_base:g}")
    print(f"  optimizer:       AdamW lr={args.lr:g} batch={args.batch_size} epochs={args.epochs}")

    baseline = evaluate_model(model, anchor, eval_loader, args.device)
    print("\n[4/5] HELD-OUT BASELINE", flush=True)
    print(
        f"  top1_regret={baseline['regret']:.4f} near_best={baseline['near_best']:.3f} "
        f"expected_regret={baseline['expected_regret']:.4f} "
        f"signed_logprob={baseline['signed_logprob']:+.4f} tv_base={baseline['tv_base']:.4f}",
        flush=True,
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = baseline
    best_epoch = 0
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    with args.metrics.open("w", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps({"epoch": 0, "split": "eval", **baseline}) + "\n")
        print("\n[5/5] POLICY UPDATE", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            sums = defaultdict(float)
            batches = 0
            for batch in train_loader:
                batch = {key: value.to(args.device) for key, value in batch.items()}
                piece = batch["piece_idx"].unsqueeze(1)
                aux = batch["aux"].unsqueeze(1)
                logits, _ = model(piece, aux)
                with torch.no_grad():
                    base_logits, _ = anchor(piece, aux)
                loss, batch_metrics = oracle_policy_loss(
                    logits[:, -1, :],
                    base_logits[:, -1, :],
                    batch["legal"],
                    batch["rollout_policy"],
                    batch["action"],
                    batch["advantage"],
                    delta=args.delta,
                    beta=args.beta,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                for key, value in batch_metrics.items():
                    sums[key] += float(value.item())
                batches += 1

            train_metrics = {key: value / max(1, batches) for key, value in sums.items()}
            eval_metrics = evaluate_model(model, anchor, eval_loader, args.device)
            # top-1 regret is discrete and often stays flat under a small KL-fenced
            # update. held-out A*log pi(a) is the smooth objective this run trains.
            accepted = (
                eval_metrics["signed_logprob"] > best_metrics["signed_logprob"]
                and eval_metrics["tv_base"] <= args.max_tv_base
            )
            epoch_path = args.out.with_name(f"{args.out.stem}_epoch{epoch}{args.out.suffix}")
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "training_objective": "oracle_process_rl",
                    "epoch": epoch,
                    "eval_metrics": eval_metrics,
                },
                epoch_path,
            )
            if accepted:
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = eval_metrics
                best_epoch = epoch
            print(
                f"  epoch {epoch}: pg={train_metrics['policy_gradient']:+.4f} "
                f"kl={train_metrics['anchor_kl']:.4f} tv_base={eval_metrics['tv_base']:.4f} "
                f"keep={train_metrics['keep_rate']:.0%} entropy={train_metrics['entropy']:.3f} | "
                f"top1_regret={eval_metrics['regret']:.4f} "
                f"expected_regret={eval_metrics['expected_regret']:.4f} "
                f"heldout_Alogp={eval_metrics['signed_logprob']:+.4f} "
                f"{'BEST' if accepted else 'reject'}",
                flush=True,
            )
            metrics_file.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "split": "train_eval",
                        "accepted": accepted,
                        "train": train_metrics,
                        "eval": eval_metrics,
                    }
                )
                + "\n"
            )
            metrics_file.flush()

    model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "training_objective": "oracle_process_rl",
            "base_checkpoint": str(args.checkpoint),
            "anchor_checkpoint": str(args.anchor),
            "best_epoch": best_epoch,
            "baseline_metrics": baseline,
            "best_metrics": best_metrics,
        },
        args.out,
    )
    verdict = "IMPROVED_OFFLINE" if best_epoch > 0 else "NO_OFFLINE_LIFT_BASE_RESTORED"
    print("\nORACLE RL RESULT", flush=True)
    print(f"  best epoch:      {best_epoch}")
    print(f"  regret:          {baseline['regret']:.4f} -> {best_metrics['regret']:.4f}")
    print(f"  expected regret: {baseline['expected_regret']:.4f} -> {best_metrics['expected_regret']:.4f}")
    print(f"  heldout A*logp:  {baseline['signed_logprob']:+.4f} -> {best_metrics['signed_logprob']:+.4f}")
    print(f"  near-best:       {baseline['near_best']:.3f} -> {best_metrics['near_best']:.3f}")
    print(f"  verdict:         {verdict}")
    print(f"  checkpoint:      {args.out}")
    print(f"  metrics:         {args.metrics}")


def command_run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "on_policy_raw.jsonl"
    labeled = args.out_dir / "oracle_labeled.jsonl"
    checkpoint = args.out_dir / "oracle_process_rl.pt"
    metrics = args.report_dir / "training_metrics.jsonl"

    print("=" * 64)
    print(" KIBITZER ORACLE-SHAPED ON-POLICY RL")
    print("=" * 64)
    print(f"base/anchor:       {args.checkpoint}")
    print(f"rollout:           {args.groups * args.group_size} games @ {args.sims} sims vs SF-{args.opponent_elo}")
    print(f"teacher:           Stockfish nodes={args.teacher_nodes:,} multipv={args.multipv}")
    print(f"reward:            process={args.process_weight:g} terminal={args.terminal_weight:g} gamma={args.gamma:g}")
    print(f"filter:            regret>={args.min_regret:g}, |adv|>={args.min_abs_advantage:g}")
    print(f"update:            policy+norm only, beta={args.beta:g}, lr={args.lr:g} x{args.epochs}")
    print(f"outputs:           {args.out_dir} and {args.report_dir}")
    print()

    if args.reuse_rollout:
        if not raw.is_file():
            raise SystemExit(f"cannot resume; rollout buffer is missing: {raw}")
        print(f"[1/3] REUSE COMPLETED ON-POLICY ROLLOUT\n  input:           {raw}", flush=True)
        print(f"  positions:       {sum(1 for line in raw.open(encoding='utf-8') if line.strip()):,}", flush=True)
    else:
        gen_args = argparse.Namespace(**vars(args), out_jsonl=raw)
        command_generate(gen_args)
    print("\n" + "-" * 64 + "\n")
    label_args = argparse.Namespace(**vars(args), input_jsonl=raw, out_jsonl=labeled)
    command_label(label_args)
    print("\n" + "-" * 64 + "\n")
    train_args = argparse.Namespace(
        data=labeled,
        checkpoint=args.checkpoint,
        anchor=args.checkpoint,
        min_regret=args.min_regret,
        min_abs_advantage=args.min_abs_advantage,
        eval_fraction=args.eval_fraction,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        delta=args.delta,
        beta=args.beta,
        max_tv_base=args.max_tv_base,
        seed=args.seed,
        device=args.device,
        out=checkpoint,
        metrics=metrics,
    )
    command_train(train_args)
    summary = {
        "checkpoint": str(checkpoint),
        "raw_rollout": str(raw),
        "labeled_buffer": str(labeled),
        "metrics": str(metrics),
        "next": "run paired external gates at 128 and 512 simulations",
    }
    with (args.report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("\nORACLE_PROCESS_RL_DONE", flush=True)


def add_shared_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--opponent-elo", type=int, default=2300)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--temp-plies", type=int, default=20)
    parser.add_argument("--temp-late", type=float, default=0.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--engine-time", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_label_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teacher-nodes", type=int, default=10_000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--reward-clip", type=float, default=0.5)
    parser.add_argument("--process-weight", type=float, default=0.25)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    generate_parser = sub.add_parser("generate")
    add_shared_generation_args(generate_parser)
    generate_parser.add_argument("--out-jsonl", type=Path, required=True)

    label_parser = sub.add_parser("label")
    label_parser.add_argument("--input-jsonl", type=Path, required=True)
    label_parser.add_argument("--out-jsonl", type=Path, required=True)
    label_parser.add_argument("--stockfish", default="stockfish")
    add_label_args(label_parser)
    label_parser.add_argument("--workers", type=int, default=4)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--anchor", type=Path, required=True)
    train_parser.add_argument("--min-regret", type=float, default=0.05)
    train_parser.add_argument("--min-abs-advantage", type=float, default=0.1)
    train_parser.add_argument("--eval-fraction", type=float, default=0.2)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--delta", type=float, default=0.1)
    train_parser.add_argument("--beta", type=float, default=0.1)
    train_parser.add_argument("--max-tv-base", type=float, default=0.08)
    train_parser.add_argument("--seed", type=int, default=31)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--metrics", type=Path, required=True)

    run_parser = sub.add_parser("run")
    add_shared_generation_args(run_parser)
    add_label_args(run_parser)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--min-regret", type=float, default=0.05)
    run_parser.add_argument("--min-abs-advantage", type=float, default=0.1)
    run_parser.add_argument("--eval-fraction", type=float, default=0.2)
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--lr", type=float, default=1e-5)
    run_parser.add_argument("--weight-decay", type=float, default=0.01)
    run_parser.add_argument("--epochs", type=int, default=2)
    run_parser.add_argument("--delta", type=float, default=0.1)
    run_parser.add_argument("--beta", type=float, default=0.1)
    run_parser.add_argument("--max-tv-base", type=float, default=0.08)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--report-dir", type=Path, required=True)
    run_parser.add_argument("--reuse-rollout", action="store_true")

    args = parser.parse_args()
    {"generate": command_generate, "label": command_label, "train": command_train, "run": command_run}[args.mode](args)


if __name__ == "__main__":
    main()
