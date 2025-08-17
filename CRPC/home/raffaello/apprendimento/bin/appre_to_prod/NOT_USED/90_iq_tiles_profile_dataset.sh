#!/usr/bin/env bash
set -euo pipefail
SCRIPT="${1:-/home/raffaello/crpc/iq_to_tiles_cmap_arg.py}"
PY="${PY:-/usr/bin/python3}"
$PY "$SCRIPT" \
  --fs 10e6 --fs-view 1.2e6 \
  --nfft 65536 --hop-div 8 \
  --mode EWMA --alpha 0.20 \
  --norm bgsub --bg-alpha 0.02 \
  --fast-settle-cols 120 --fast-settle-alpha 0.25 \
  --delta-floor -10 --delta-ceil 10 \
  --gamma 1.3 \
  --cmap turbo \
  --dc-notch
