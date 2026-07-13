#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIMS_LIST="${SIMS_LIST:-256 512 1024}"
GPP="${GPP:-40}"
ST="${ST:-60}"
BASE_OUT="${BASE_OUT:-reports/official_elo_frontier}"

echo "============================================================"
echo " FIXED-MODEL OFFICIAL ELO FRONTIER"
echo "============================================================"
echo "sims:       $SIMS_LIST"
echo "games:      $GPP per Stockfish anchor"
echo "move limit: $ST seconds"
echo "outputs:    $BASE_OUT"
echo "read:       every point is validated before Ordo sees its PGN"
echo

for sims in $SIMS_LIST; do
  echo "------------------------------------------------------------"
  echo " FIXED SEARCH: $sims SIMULATIONS"
  echo "------------------------------------------------------------"
  SIMS="$sims" GPP="$GPP" ST="$ST" OUT="$BASE_OUT/s${sims}" \
    bash scripts/run_official_elo.sh
done

echo
echo "OFFICIAL_ELO_FRONTIER_DONE -> $BASE_OUT"
