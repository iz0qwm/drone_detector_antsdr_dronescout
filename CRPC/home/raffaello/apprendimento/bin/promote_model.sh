#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/raffaello/apprendimento/models"
ARCH="$ROOT/archive"; mkdir -p "$ARCH" "$ROOT/served"
TS=$(date -u +%Y%m%d-%H%M%SZ)

test -f "$ROOT/rfscan_staging.pkl" || { echo "staging mancante"; exit 1; }

# salva uno snapshot
cp -v "$ROOT/rfscan_staging.pkl" "$ARCH/rfscan_${TS}.pkl"

# aggiorna 'current' e un symlink standard per i servizi
cp -v "$ROOT/rfscan_staging.pkl" "$ROOT/rfscan_current.pkl"
ln -sfn "$ROOT/rfscan_current.pkl" "$ROOT/served/rfscan.pkl"

echo "OK: current aggiornato e snapshot salvato in $ARCH"
