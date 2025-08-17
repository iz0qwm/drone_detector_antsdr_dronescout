#!/bin/bash
# Avvia il servizio RF Explorer dual scan
set -e
systemctl --user start rfe-dual-scan.service
echo "RF Explorer dual scan avviato"
