#!/usr/bin/env bash
set -euo pipefail

PORT="/dev/ttyUSB0"         # o /dev/rfexplorer se hai creato l'alias udev
BIN="rfexplorerDetailedScan" # già nel PATH
OUTDIR="/tmp/rfe/scans"
SLEEP_BETWEEN=1              # secondi tra uno sweep e l'altro (per respirare)
LOGTAG="[RFE-DUAL]"

# Bande ISM tipiche droni
B24_START=2400
B24_END=2485
B58_START=5725
B58_END=5875

mkdir -p "$OUTDIR"

echo "$LOGTAG Start dual sweep su $PORT (2.4GHz <-> 5.8GHz)"
while true; do
  TS=$(date +%Y%m%d-%H%M%S)

  # 2.4 GHz
  OUT24="$OUTDIR/scan-24_$TS.csv"
  echo "$LOGTAG Sweep 2.4 GHz → $OUT24"
  $BIN -p "$PORT" -s $B24_START -e $B24_END "$OUT24" -v || echo "$LOGTAG WARN: sweep 2.4 fallito"
  ln -sfn "$OUT24" "$OUTDIR/latest_24.csv"

  sleep $SLEEP_BETWEEN

  # 5.8 GHz
  OUT58="$OUTDIR/scan-58_$TS.csv"
  echo "$LOGTAG Sweep 5.8 GHz → $OUT58"
  $BIN -p "$PORT" -s $B58_START -e $B58_END "$OUT58" -v || echo "$LOGTAG WARN: sweep 5.8 fallito"
  ln -sfn "$OUT58" "$OUTDIR/latest_58.csv"

  sleep $SLEEP_BETWEEN
done

