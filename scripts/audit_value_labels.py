from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from kibitzer.diagnostics import VALUE_BINS, bounded_value, value_bin
from scripts.diagnose_search import StockfishPool


AUDIT_VERSION = 1


def summarize_pairs(
    cached_values: Sequence[float],
    oracle_values: Sequence[float],
) -> dict[str, float | int]:
    if len(cached_values) != len(oracle_values):
        raise ValueError("cached and oracle values must have equal length")
    if not cached_values:
        return {
            "count": 0,
            "mae": float("nan"),
            "sign_disagreement": float("nan"),
            "direction_flip": float("nan"),
        }

    errors = [abs(cached - oracle) for cached, oracle in zip(cached_values, oracle_values, strict=True)]
    nonzero = [
        (cached, oracle)
        for cached, oracle in zip(cached_values, oracle_values, strict=True)
        if oracle != 0.0
    ]
    sign_disagreement = (
        sum((cached >= 0.0) != (oracle >= 0.0) for cached, oracle in nonzero)
        / len(nonzero)
        if nonzero
        else float("nan")
    )
    return {
        "count": len(cached_values),
        "mae": sum(errors) / len(errors),
        "sign_disagreement": sign_disagreement,
        "direction_flip": sum(cached * oracle < 0.0 for cached, oracle in zip(cached_values, oracle_values, strict=True))
        / len(cached_values),
    }


def bin_name(value: float) -> str:
    return value_bin(round(value * 1000.0))


def summarize_by_bin(
    cached_values: Sequence[float],
    oracle_values: Sequence[float],
    *,
    bin_source: str,
) -> dict[str, dict[str, float | int]]:
    if bin_source not in {"cached", "oracle"}:
        raise ValueError("bin_source must be cached or oracle")
    source = cached_values if bin_source == "cached" else oracle_values
    result: dict[str, dict[str, float | int]] = {}
    for name in VALUE_BINS:
        indices = [index for index, value in enumerate(source) if bin_name(value) == name]
        result[name] = summarize_pairs(
            [cached_values[index] for index in indices],
            [oracle_values[index] for index in indices],
        )
    return result


def census(values: Sequence[float]) -> dict[str, int]:
    return {name: sum(bin_name(value) == name for value in values) for name in VALUE_BINS}


def source_signature(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "source_signature": payload.get("signature"),
        "sample_count": len(payload["samples"]),
    }


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_indices(total: int, positions: int, seed: int) -> list[int]:
    if positions > total:
        raise ValueError(f"requested {positions} positions from a cache containing {total}")
    return sorted(random.Random(seed).sample(range(total), positions))


def route_result(overall_mae: float, won_sign_disagreement: float) -> str:
    if overall_mae >= 0.1653 - 0.02:
        return "label_limited"
    if overall_mae < 0.10 and won_sign_disagreement <= 0.08:
        return "cached_labels_have_headroom"
    return "borderline_head_only_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit depth-14 cached value labels against depth 20.")
    parser.add_argument("--label-cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish-path", required=True)
    parser.add_argument("--stockfish-workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--positions", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.positions < 1 or args.batch_size < 1 or args.stockfish_workers < 1:
        raise SystemExit("positions, batch size, and workers must all be positive")
    if not args.label_cache.is_file():
        raise SystemExit(f"missing joint label cache: {args.label_cache}")

    payload = torch.load(args.label_cache, map_location="cpu", weights_only=False)
    signature = {
        "version": AUDIT_VERSION,
        "source": source_signature(args.label_cache, payload),
        "positions": args.positions,
        "seed": args.seed,
        "depth": args.depth,
    }
    indices = choose_indices(len(payload["samples"]), args.positions, args.seed)
    cached_values = [float(payload["labels"][index][1]) for index in indices]

    completed: dict[int, int] = {}
    if args.audit_cache.is_file():
        audit_payload = torch.load(args.audit_cache, map_location="cpu", weights_only=False)
        if audit_payload.get("signature") != signature:
            raise SystemExit(
                f"audit cache settings do not match: {args.audit_cache}; use a different path"
            )
        completed = {int(index): int(cp) for index, cp in audit_payload["depth20_cp"].items()}

    pending = [index for index in indices if index not in completed]
    print("[1/3] LABEL-CEILING AUDIT CONFIGURATION", flush=True)
    print(f"  source cache:     {args.label_cache}")
    print(f"  cached teacher:   depth={payload.get('signature', {}).get('depth')} MultiPV={payload.get('signature', {}).get('multipv')}")
    print(f"  audit teacher:    depth={args.depth} MultiPV=1")
    print(f"  sampled positions:{len(indices):,} deterministic (seed={args.seed})")
    print(f"  already cached:   {len(completed):,}")
    print("  locked test split:not read or modified", flush=True)

    if pending:
        print("\n[2/3] STOCKFISH DEPTH-20 RELABELING", flush=True)
        with StockfishPool(
            path=args.stockfish_path,
            depth=args.depth,
            workers=args.stockfish_workers,
        ) as pool:
            for start in range(0, len(pending), args.batch_size):
                batch_indices = pending[start : start + args.batch_size]
                tasks = [(payload["samples"][index].fen, None) for index in batch_indices]
                results = pool.analyse(tasks, description="audit depth20")
                completed.update(
                    {index: cp for index, (cp, _) in zip(batch_indices, results, strict=True)}
                )
                save_atomic(
                    {"signature": signature, "depth20_cp": completed},
                    args.audit_cache,
                )
                print(f"  progress cached:  {len(completed):,}/{len(indices):,}", flush=True)
    else:
        print("\n[2/3] STOCKFISH DEPTH-20 RELABELING")
        print("  cache complete: no Stockfish work needed")

    oracle_values = [bounded_value(completed[index]) for index in indices]
    overall = summarize_pairs(cached_values, oracle_values)
    cached_bins = summarize_by_bin(cached_values, oracle_values, bin_source="cached")
    oracle_bins = summarize_by_bin(cached_values, oracle_values, bin_source="oracle")
    decision = route_result(
        float(overall["mae"]),
        float(oracle_bins["won"]["sign_disagreement"]),
    )
    report = {
        "contract": signature,
        "full_cached_label_census": census([float(label[1]) for label in payload["labels"]]),
        "sample_cached_label_census": census(cached_values),
        "sample_depth20_label_census": census(oracle_values),
        "overall": overall,
        "by_cached_label_bin": cached_bins,
        "by_depth20_label_bin": oracle_bins,
        "decision": decision,
        "thresholds": {
            "label_limited_mae_at_least": 0.1453,
            "headroom_mae_below": 0.10,
            "headroom_won_sign_disagreement_at_most": 0.08,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)

    print("\n[3/3] AUDIT RESULT", flush=True)
    print(f"  overall MAE:              {float(overall['mae']):.4f}")
    print(f"  overall sign disagreement:{100.0 * float(overall['sign_disagreement']):.2f}%")
    print(f"  won-bin disagreement:     {100.0 * float(oracle_bins['won']['sign_disagreement']):.2f}% (depth-20 bins)")
    print(f"  routing decision:         {decision}")
    print(f"  report:                   {args.output}")


if __name__ == "__main__":
    main()
