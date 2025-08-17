#!/usr/bin/env bash
set -euo pipefail
SVC=(crpc-rfscan crpc-tiles crpc-tracker crpc-yolo hackrf-controller rfe-csv-bridge rfe-dual-scan rfe-trigger)
for s in "${SVC[@]}"; do
  if systemctl is-active --quiet "$s"; then
    sudo systemctl stop "$s"
  fi
done
# piccola attesa per liberare le USB/SDR
sleep 1
# opzionale: rilascia lock file del tuo pipeline live se presente

