#!/usr/bin/env bash
set -euo pipefail
# evita ripartenza se lock presente
if [ -f /run/apprendimento.lock ]; then
  echo "Lock presente: non riavvio i live."; exit 0
fi
SVC=(crpc-rfscan crpc-tiles crpc-tracker crpc-yolo hackrf-controller rfe-csv-bridge rfe-dual-scan rfe-trigger)
for s in "${SVC[@]}"; do
  sudo systemctl start "$s" || true
done

