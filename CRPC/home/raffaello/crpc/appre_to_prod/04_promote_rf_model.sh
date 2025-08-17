#!/usr/bin/env bash
set -euo pipefail
STAGING="${1:-/home/raffaello/apprendimento/models/rfscan_staging.pkl}"
SERVED_DIR="${2:-/home/raffaello/apprendimento/models/served}"
TARGET="$SERVED_DIR/rfscan.pkl"
mkdir -p "$SERVED_DIR"
cp -f "$STAGING" "$TARGET"
echo "Promoted RF model to: $TARGET"
