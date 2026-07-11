#!/usr/bin/env bash
# thin wrapper so cutechess-cli can launch the model as a uci engine.
# usage (cutechess): cmd=scripts/kibitzer_uci.sh arg=--sims arg=512 ...
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/kibitzer-uv-cache}"
exec uv run python scripts/uci_engine.py "$@"
