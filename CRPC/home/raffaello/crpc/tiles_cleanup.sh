#!/usr/bin/env bash
set -euo pipefail
# Tieni solo gli ultimi 2000 PNG (regola a piacere)
find /tmp/tiles_done -type f -name '*.png' -printf '%T@ %p\n' \
 | sort -n \
 | awk 'NR<=KEEP{next} {print $2}' KEEP=2000 \
 | xargs -r rm -f

