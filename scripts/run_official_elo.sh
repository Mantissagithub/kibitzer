#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

# proper tournament-based Elo for the model wrapped in 512-sim PUCT search. plays a
# cutechess-cli gauntlet vs a calibrated Stockfish UCI_Elo ladder (both colors, real
# opening book, adjudication), then Ordo computes the model's rating with a
# confidence interval, anchored to the Stockfish ladder. this is the CCRL-style
# method, not the old single-opponent score->Elo transform.
#
#   bash scripts/run_official_elo.sh                 # ~overnight local run
#   GPP=100 bash scripts/run_official_elo.sh         # more games = tighter CI
#   SMOKE=1 bash scripts/run_official_elo.sh         # 2-game sanity check

CHECKPOINT="${CHECKPOINT:-runs/tactical/tactical_repair.pt}"
SIMS="${SIMS:-512}"
DEVICE="${DEVICE:-cuda}"
GPP="${GPP:-40}"                       # games per opponent (both colors)
CONCURRENCY="${CONCURRENCY:-1}"        # 1 = model gets the whole GPU (avoid contention)
ST="${ST:-30}"                         # seconds/move ceiling (model ignores clock, does fixed sims)
MAXMOVES="${MAXMOVES:-0}"              # 0 = no cap; smoke uses a cap so plumbing finishes quickly
ORDO_SAMPLES="${ORDO_SAMPLES:-1000}"
ANCHORS="${ANCHORS:-2200 2500 2700 2900 3100}"   # stockfish UCI_Elo ladder
ANCHOR_REF="${ANCHOR_REF:-2500}"       # the ladder rung Ordo pins the scale to
BOOK="${BOOK:-resources/books/8moves_v3.pgn}"
OPENING_PLIES="${OPENING_PLIES:-8}"
OUT="${OUT:-reports/official_elo}"
STOCKFISH="${STOCKFISH:-$(command -v stockfish || true)}"
CUTECHESS="${CUTECHESS:-$(command -v cutechess-cli || true)}"
ORDO="${ORDO:-$ROOT/scripts/ordo}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  SIMS="${SMOKE_SIMS:-1}"
  GPP=2
  ANCHORS="${SMOKE_ANCHORS:-2500}"
  ANCHOR_REF=2500
  ST="${SMOKE_ST:-10}"
  MAXMOVES="${SMOKE_MAXMOVES:-1}"
  OPENING_PLIES="${SMOKE_OPENING_PLIES:-1}"
  ORDO_SAMPLES="${SMOKE_ORDO_SAMPLES:-100}"
  DEVICE="${SMOKE_DEVICE:-cpu}"
  OUT="${SMOKE_OUT:-reports/official_elo_smoke}"
fi

for x in "$CHECKPOINT" "$BOOK" "$ORDO"; do [[ -s "$x" ]] || { echo "error: missing $x" >&2; exit 1; }; done
[[ -x "$CUTECHESS" ]] || { echo "error: cutechess-cli not found" >&2; exit 1; }
[[ -x "$STOCKFISH" ]] || { echo "error: stockfish not found" >&2; exit 1; }
mkdir -p "$OUT"
ROUNDS=$(( GPP / 2 )); (( ROUNDS < 1 )) && ROUNDS=1
EXPECTED_GAMES=$(( ROUNDS * 2 * $(wc -w <<< "$ANCHORS") ))
PGN="$OUT/official_elo_s${SIMS}_gpp${GPP}.pgn"
CUTECHESS_LOG="$OUT/cutechess_s${SIMS}_gpp${GPP}.log"

# interrupted cutechess runs leave partial PGNs. a new rating run must never append
# fresh games to that stale evidence, because Ordo would silently rate the mixture.
rm -f "$PGN" "$CUTECHESS_LOG" "$OUT/ratings.txt" "$OUT/validation.json"

echo "============================================================"
echo " OFFICIAL ELO TOURNAMENT  (model @ ${SIMS} sims vs SF ladder)"
echo "============================================================"
echo "checkpoint:  $CHECKPOINT"
echo "model:       Kibitzer-s${SIMS}  device=$DEVICE"
echo "opponents:   SF UCI_Elo { $ANCHORS }   (Ordo anchor = SF-$ANCHOR_REF)"
echo "games:       $GPP per opponent (both colors), concurrency $CONCURRENCY"
if (( MAXMOVES > 0 )); then
  echo "move cap:    $MAXMOVES full moves (smoke/plumbing mode)"
else
  echo "move cap:    none"
fi
echo "book:        $BOOK  plies=$OPENING_PLIES  |  pgn -> $PGN"
echo "cutechess:   full stdout/stderr -> $CUTECHESS_LOG"
echo "ordo:        ratings -> $OUT/ratings.txt"
echo "validation:  expected_games=$EXPECTED_GAMES; time/illegal/unterminated games rejected"
echo "read:        smoke checks plumbing only; real Elo needs GPP >= 40"
echo

# engine list: model first (gauntlet seed), then the stockfish anchors
engines=( -engine "name=Kibitzer-s${SIMS}" "cmd=$ROOT/scripts/kibitzer_uci.sh"
          "arg=--checkpoint" "arg=$CHECKPOINT" "arg=--sims" "arg=$SIMS"
          "arg=--device" "arg=$DEVICE" proto=uci )
for e in $ANCHORS; do
  engines+=( -engine "name=SF-$e" "cmd=$STOCKFISH" proto=uci
             "option.UCI_LimitStrength=true" "option.UCI_Elo=$e" )
done

cutechess_extra=()
if (( MAXMOVES > 0 )); then
  cutechess_extra+=( -maxmoves "$MAXMOVES" )
fi

set +e
"$CUTECHESS" "${engines[@]}" \
  -each proto=uci "st=$ST" \
  -tournament gauntlet -concurrency "$CONCURRENCY" \
  -rounds "$ROUNDS" -games 2 -repeat \
  -openings "file=$BOOK" format=pgn order=random "plies=$OPENING_PLIES" \
  -resign movecount=3 score=700 -draw movenumber=40 movecount=8 score=10 \
  "${cutechess_extra[@]}" \
  -pgnout "$PGN" -ratinginterval 10 -recover 2>&1 | tee "$CUTECHESS_LOG"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  echo "error: cutechess-cli failed with exit code $rc"
  echo "full log: $CUTECHESS_LOG"
  exit "$rc"
fi

finished_games=0
if [[ -s "$PGN" ]]; then
  finished_games="$(grep -c '^\[Result ' "$PGN" || true)"
fi
if (( finished_games < EXPECTED_GAMES )); then
  echo "error: cutechess stopped before the tournament completed"
  echo "finished_games: $finished_games / $EXPECTED_GAMES"
  echo "full log: $CUTECHESS_LOG"
  echo "pgn: $PGN"
  exit 130
fi

echo
echo "cutechess completed:"
echo "  log: $CUTECHESS_LOG"
echo "  pgn: $PGN"

validation_args=(
  --pgn "$PGN"
  --expected-games "$EXPECTED_GAMES"
  --out "$OUT/validation.json"
)
if (( MAXMOVES > 0 )); then
  validation_args+=( --allow-unterminated )
fi
uv run python scripts/validate_tournament_pgn.py "${validation_args[@]}"

if (( MAXMOVES > 0 )); then
  echo
  echo "SMOKE_DONE -> protocol and PGN plumbing are valid; no rating was calculated"
  exit 0
fi

echo
echo "============================================================"
echo " ANCHORED RATING (Ordo, with confidence interval)"
echo "============================================================"
"$ORDO" -Q -D -a "$ANCHOR_REF" -A "SF-$ANCHOR_REF" -s "$ORDO_SAMPLES" -p "$PGN" -o "$OUT/ratings.txt"
echo
cat "$OUT/ratings.txt"
echo
echo "read: 'Kibitzer-s${SIMS}' RATING column = the model's tournament Elo; the +/- is the"
echo "95% error bar. The SF-* rows should land near their UCI_Elo labels (calibration check)."
echo "OFFICIAL_ELO_DONE -> $OUT/ratings.txt"
