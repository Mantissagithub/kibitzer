# read the per-game jsonl from analyze_failures.py and roll it up into the numbers the
# report and the gpt consult actually need: score + loss-shape split per opponent, acpl by
# phase, blunder concentration by phase, and the search-axis view (does more search cut the
# per-move error, and where). also dumps a handful of representative losing positions.

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def simcount(src: str) -> str | None:
    m = re.search(r"_s(\d+)_", src)
    return m.group(1) if m else None


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def score_of(g: dict) -> float:
    return {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(g["result"], 0.0)


def phase_acpl(games: list[dict]) -> dict:
    out = {}
    for ph in ("opening", "middlegame", "endgame"):
        vals = [g["acpl_by_phase"][ph] for g in games if g["acpl_by_phase"].get(ph) is not None]
        out[ph] = round(mean(vals), 1) if vals else None
    return out


def summarize(games: list[dict], label: str) -> dict:
    valid = [g for g in games if g["valid"] and g["result"] in ("win", "draw", "loss")]
    losses = [g for g in valid if g["result"] == "loss"]
    shapes = defaultdict(int)
    for g in losses:
        shapes[g["loss_shape"]] += 1
    # blunders per 100 contested moves, and where they land
    bl = [b for g in valid for b in g["blunders"]]
    bl_phase = defaultdict(int)
    for b in bl:
        bl_phase[b["phase"]] += 1
    return {
        "label": label,
        "games": len(valid),
        "score": round(mean(score_of(g) for g in valid), 3) if valid else None,
        "losses": len(losses),
        "loss_shapes": dict(shapes),
        "acpl_overall": round(mean(g["acpl"] for g in valid if g["acpl"]), 1) if valid else None,
        "acpl_by_phase": phase_acpl(valid),
        "blunders_total": len(bl),
        "blunders_by_phase": dict(bl_phase),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    games = load(args.jsonl)

    by_opp = defaultdict(list)
    by_sim = defaultdict(list)
    for g in games:
        by_opp[g["opponent"]].append(g)
        sc = simcount(g["src"])
        if sc:
            by_sim[sc].append(g)

    report = {"overall": summarize(games, "ALL"),
              "by_opponent": {k: summarize(v, k) for k, v in sorted(by_opp.items())},
              "by_simcount": {k: summarize(v, f"s{k}") for k, v in sorted(by_sim.items(), key=lambda x: int(x[0]))}}

    # representative losses: rarest-shape and worst-acpl examples with the collapse position
    valid_losses = [g for g in games if g["valid"] and g["result"] == "loss"]
    reps = []
    for g in sorted(valid_losses, key=lambda x: -x["acpl"])[:8]:
        pos = None
        for b in g["blunders"]:
            pos = b
            break
        reps.append({"src": Path(g["src"]).name, "idx": g["idx"], "opp": g["opponent"],
                     "shape": g["loss_shape"], "acpl": g["acpl"], "collapse_ply": g["collapse_ply"],
                     "decisive_ply": g["decisive_ply"], "decisive_cpl": g["decisive_cpl"],
                     "example_blunder": pos})
    report["representative_losses"] = reps

    txt = json.dumps(report, indent=2)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)


if __name__ == "__main__":
    main()
