#!/usr/bin/env bash
set -euo pipefail
SCRIPT="${1:-/home/raffaello/crpc/iq_to_tiles_cmap_arg.py}"
PY="${PY:-/usr/bin/python3}"
$PY "$SCRIPT" \
  --fs 10e6 --fs-view 1.2e6 \
  --nfft 65536 --hop-div 8 \
  --mode EWMA --alpha 0.12 \
  --norm bgsub --bg-alpha 0.015 \
  --fast-settle-cols 120 --fast-settle-alpha 0.20 \
  --delta-floor -6 --delta-ceil 10 \
  --gamma 1.35 --cmap orange_red \
  --dc-notch
