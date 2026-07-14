#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# salvage a rating from a tournament PGN that contains a few contaminated games
# (anchor time-forfeits, illegal moves, unterminated). drops ONLY those games and runs
# ordo on every real game, so a run does not have to be repeated just because the strict
# validator (validate_tournament_pgn.py) fails closed on a handful of clock artifacts.
#
#   bash scripts/rate_pgn.sh                                  # default official_elo pgn
#   bash scripts/rate_pgn.sh reports/official_elo/official_elo_s512_gpp40.pgn

PGN="${1:-reports/official_elo/official_elo_s512_gpp40.pgn}"
ANCHOR_REF="${ANCHOR_REF:-2500}"
ORDO_SAMPLES="${ORDO_SAMPLES:-1000}"
ORDO="${ORDO:-$ROOT/scripts/ordo}"
CLEAN="${CLEAN:-${PGN%.pgn}_clean.pgn}"
OUT="${OUT:-$(dirname "$PGN")/ratings_clean.txt}"

[[ -s "$PGN" ]] || { echo "error: missing pgn $PGN" >&2; exit 1; }
[[ -x "$ORDO" ]] || { echo "error: ordo not executable: $ORDO" >&2; exit 1; }

python3 - "$PGN" "$CLEAN" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
parts = [p for p in re.split(r'(?=\[Event )', text) if p.strip()]
bad = ("time forfeit", "illegal move", "unterminated")
kept, dropped, noresult = [], 0, 0
for p in parts:
    if not re.search(r'\[Result "(1-0|0-1|1/2-1/2)"', p):
        noresult += 1
        continue
    m = re.search(r'\[Termination "([^"]+)"', p)
    term = m.group(1).lower() if m else ""
    if any(b in term for b in bad):
        dropped += 1
        continue
    kept.append(p if p.endswith("\n") else p + "\n")
open(dst, "w").write("\n".join(kept))
print(f"kept {len(kept)} rated games | dropped {dropped} contaminated | skipped {noresult} without a result")
PY

echo "clean pgn: $CLEAN"
echo "------------------------------------------------------------"
"$ORDO" -Q -D -a "$ANCHOR_REF" -A "SF-$ANCHOR_REF" -s "$ORDO_SAMPLES" -p "$CLEAN" -o "$OUT"
echo
cat "$OUT"
