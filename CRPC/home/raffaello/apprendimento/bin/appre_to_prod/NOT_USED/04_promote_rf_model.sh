#!/usr/bin/env bash
set -euo pipefail
STAGING="${1:-/home/raffaello/apprendimento/models/rfscan_staging.pkl}"
SERVED_DIR="${2:-/home/raffaello/apprendimento/models/served}"
ts=$(date +%Y%m%d-%H%M%S)
mkdir -p "$SERVED_DIR"

# copia versionata
TARGET_VER="$SERVED_DIR/rfscan_${ts}.pkl"
cp -f "$STAGING" "$TARGET_VER"

# aggiorna symlink atomico
ln -sfn "$(basename "$TARGET_VER")" "$SERVED_DIR/rfscan.pkl"

echo "Promoted to: $TARGET_VER"
echo "Symlink updated: $SERVED_DIR/rfscan.pkl -> $(basename "$TARGET_VER")"
