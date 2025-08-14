#!/usr/bin/env bash
set -euo pipefail

PORT="/dev/ttyUSB0"
OUTDIR="/tmp/rfe/scan"
RING=10                 # quanti CSV tenere per banda (0..9)
SLEEP_BETWEEN=1         # secondi tra gli sweep
LOGTAG="[RFE-TRIPLE]"

# Bande
B24_START=2400
B24_END=2485
B52_START=5170          
B52_END=5250
B58_START=5725
B58_END=5875

# rfe_scan.py accanto a questo script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYSCAN="${SCRIPT_DIR}/rfe_scan.py"

mkdir -p "$OUTDIR"

echo "$LOGTAG Start dual sweep (Python) su $PORT → $OUTDIR (ring=$RING)"
i=0
while true; do
  idx=$(( i % RING ))

  # 2.4 GHz
  OUT24="$OUTDIR/24_${idx}.csv"
  echo "$LOGTAG [24] sweep ${B24_START}-${B24_END} → $OUT24"
  python3 "$PYSCAN" --port "$PORT" --band 24 -s $B24_START -e $B24_END --out "$OUT24" -v \
    || echo "$LOGTAG WARN: sweep 24 fallito"
  ln -sfn "$OUT24" "$OUTDIR/latest_24.csv"

  sleep "$SLEEP_BETWEEN"

  # 5.2 GHz  ⬅️ NUOVO BLOCCO
  OUT52="$OUTDIR/52_${idx}.csv"
  echo "$LOGTAG [52] sweep ${B52_START}-${B52_END} → $OUT52"
  python3 "$PYSCAN" --port "$PORT" --band 52 -s $B52_START -e $B52_END --out "$OUT52" -v \
    || echo "$LOGTAG WARN: sweep 52 fallito"
  ln -sfn "$OUT52" "$OUTDIR/latest_52.csv"

  sleep "$SLEEP_BETWEEN"
  
  # 5.8 GHz
  OUT58="$OUTDIR/58_${idx}.csv"
  echo "$LOGTAG [58] sweep ${B58_START}-${B58_END} → $OUT58"
  python3 "$PYSCAN" --port "$PORT" --band 58 -s $B58_START -e $B58_END --out "$OUT58" -v \
    || echo "$LOGTAG WARN: sweep 58 fallito"
  ln -sfn "$OUT58" "$OUTDIR/latest_58.csv"

  sleep "$SLEEP_BETWEEN"
  i=$((i + 1))
done

