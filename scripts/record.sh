#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: records a .mkv via the autosplitter's --record flag.
# The autosplitter is the sole UDP consumer — no second socket needed.
#
# Usage:
#   ./scripts/record.sh [PATH]
#
# PATH defaults to recordings/<timestamp>.mkv

OUTDIR="${RECORD_DIR:-$(pwd)/recordings}"
mkdir -p "$OUTDIR"
OUTFILE="${1:-$OUTDIR/$(date +%Y%m%d_%H%M%S).mkv}"

echo "Starting autosplitter with recording -> $OUTFILE"
echo "Press Ctrl-C to stop."

exec uv run python -m autosplitter.main \
  --config configs/start-powerless.yaml \
  --live --livesplit --loop \
  --record "$OUTFILE"
