# offline failure analysis: re-score every model move with stockfish to find where the
# model actually loses ground. the model emits no eval of its own, so we reconstruct a
# per-move centipawn-loss (cpl) curve the way a lichess computer review does: eval the
# position (best), eval after the played move, difference = how much the move threw away.
# from that we classify each loss as sudden (a tactical blunder) vs gradual (slow
# positional drift into a lost game) vs mate-net, and bucket cpl by game phase.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn


MATE_CP = 10000          # a forced mate is scored as this minus distance
BLUNDER = 200            # single-move cpl (in cp) that counts as a blunder
LOSING = 150             # model-pov eval (cp) below which we call the position lost
DECIDED = 600            # |eval| above this = position already decided; move quality here is noise
CPL_CAP = 1000           # clamp per-move cpl so one mate doesn't dominate an average


def score_cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    # signed centipawns from `pov`'s point of view; mates mapped onto a large ramp so
    # "mate in 2" is worse-for-the-loser than "mate in 8" but still comparable to cp.
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        return (MATE_CP - abs(m) * 10) * (1 if m > 0 else -1)
    return s.score()


def phase_of(ply: int) -> str:
    if ply <= 20:
        return "opening"
    if ply <= 60:
        return "middlegame"
    return "endgame"


@dataclass
class MoveRec:
    ply: int
    phase: str
    san: str
    fen: str
    eval_before: int     # model pov, best play
    eval_after: int      # model pov, after the played move
    cpl: int


@dataclass
class GameRec:
    src: str
    idx: int
    model_white: bool
    opponent: str
    result: str          # model pov: win/loss/draw/void
    termination: str
    valid: bool = True   # false for time forfeits / unterminated (harness artifacts, not chess)
    n_model_moves: int = 0
    acpl: float = 0.0
    acpl_by_phase: dict = field(default_factory=dict)
    blunders: list = field(default_factory=list)   # plies with cpl>=BLUNDER
    collapse_ply: int | None = None                # first ply eval stays below -LOSING to the end
    decisive_ply: int | None = None                # ply of the biggest throw from a still-playable pos
    decisive_cpl: int = 0
    max_drop: int = 0                              # biggest single-move cpl (any position)
    max_drop_ply: int | None = None
    loss_shape: str | None = None                  # sudden / gradual / n-a


def model_result(headers, model_white: bool) -> str:
    r = headers.get("Result", "*")
    if r == "*":
        return "void"
    if r == "1/2-1/2":
        return "draw"
    return "win" if (r == "1-0") == model_white else "loss"


def analyse(engine, board, limit) -> chess.engine.PovScore:
    info = engine.analyse(board, limit)
    return info["score"]


def score_game(engine, game, src, idx, limit) -> GameRec:
    h = game.headers
    white = h.get("White", "")
    model_white = white.startswith("Kibitzer")
    opp = h.get("Black" if model_white else "White", "?")
    res = model_result(h, model_white)
    term = h.get("Termination", "normal")

    board = game.board()
    moves = [n.move for n in game.mainline()]
    recs: list[MoveRec] = []
    pov = chess.WHITE if model_white else chess.BLACK

    for mv in moves:
        if board.turn == pov and not board.is_game_over():
            before = score_cp(analyse(engine, board, limit), pov)
            san = board.san(mv)
            ply = board.ply() + 1
            fen = board.fen()
            board.push(mv)
            after = score_cp(analyse(engine, board, limit), pov)
            cpl = max(0, min(before - after, CPL_CAP))
            recs.append(MoveRec(ply, phase_of(ply), san, fen, before, after, cpl))
        else:
            board.push(mv)

    valid = res != "void" and term not in ("time forfeit",)
    g = GameRec(src=src, idx=idx, model_white=model_white, opponent=opp,
                result=res, termination=term, valid=valid, n_model_moves=len(recs))
    if not recs:
        g.loss_shape = "n-a"
        return g

    # only judge moves made from a still-contested position (|eval_before|<=DECIDED);
    # accuracy while already winning/losing is technique noise, not why games are decided.
    contested = [r for r in recs if abs(r.eval_before) <= DECIDED]
    if contested:
        g.acpl = round(sum(r.cpl for r in contested) / len(contested), 1)
        for ph in ("opening", "middlegame", "endgame"):
            sub = [r.cpl for r in contested if r.phase == ph]
            if sub:
                g.acpl_by_phase[ph] = round(sum(sub) / len(sub), 1)
    g.blunders = [{"ply": r.ply, "phase": r.phase, "san": r.san, "cpl": r.cpl,
                   "eval_before": r.eval_before, "eval_after": r.eval_after, "fen": r.fen}
                  for r in contested if r.cpl >= BLUNDER]
    worst = max(recs, key=lambda r: r.cpl)
    g.max_drop, g.max_drop_ply = worst.cpl, worst.ply

    # sustained collapse: earliest ply after which the model eval never climbs back to playable
    for i, r in enumerate(recs):
        if r.eval_after < -LOSING and all(rr.eval_after < -LOSING for rr in recs[i:]):
            g.collapse_ply = r.ply
            break

    # decisive throw = biggest cpl among moves played from a still-playable position
    # (eval_before > -LOSING). the final mating move is excluded because by then the
    # position is already lost, so it can't masquerade as the blunder that lost the game.
    throws = [r for r in recs if r.eval_before > -LOSING]
    if throws:
        d = max(throws, key=lambda r: r.cpl)
        g.decisive_ply, g.decisive_cpl = d.ply, d.cpl

    if res == "loss" and valid:
        reached_lost = any(r.eval_after < -LOSING for r in recs)
        if g.decisive_cpl >= 300:
            g.loss_shape = "sudden"        # one move took a playable position to lost
        elif reached_lost:
            g.loss_shape = "gradual"       # rotted into a lost game with no single big throw
        else:
            g.loss_shape = "gradual"       # ground down / resigned without a sampled collapse
    else:
        g.loss_shape = "n-a"
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--movetime", type=float, default=0.0, help="sec/eval; overrides depth if >0")
    ap.add_argument("--max-games", type=int, default=0, help="cap games per file (0=all)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hash", type=int, default=256)
    args = ap.parse_args()

    limit = (chess.engine.Limit(time=args.movetime) if args.movetime > 0
             else chess.engine.Limit(depth=args.depth))
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Threads": args.threads, "Hash": args.hash})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[GameRec] = []
    for src in args.pgns:
        with open(src) as f:
            i = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if args.max_games and i >= args.max_games:
                    break
                g = score_game(engine, game, src, i, limit)
                results.append(g)
                print(f"{Path(src).name}[{i:>2}] {'W' if g.model_white else 'B'} vs {g.opponent:9} "
                      f"{g.result:5} shape={str(g.loss_shape):8} acpl={g.acpl:6} "
                      f"collapse={g.collapse_ply} maxdrop={g.max_drop}@{g.max_drop_ply}", flush=True)
                i += 1
    engine.quit()

    with open(out, "w") as f:
        for g in results:
            f.write(json.dumps(asdict(g)) + "\n")
    print(f"\nwrote {len(results)} game records -> {out}")


if __name__ == "__main__":
    main()
