#!/bin/bash

# Termina tutti i processi rilevanti avviati da start_all.sh
echo "🛑 Arresto di tutti i servizi..."

# Ferma servizi su AntSDR
SCRIPT_DIR="/home/pi"
ANTSDR_CONTROL="$SCRIPT_DIR/trackers/service_controller.sh"
bash "$ANTSDR_CONTROL" stop

# Uccide gli script Python avviati manualmente o via nohup
pkill -f dji_receiver.py
pkill -f bridge_uploader.py
pkill -f main.py

sleep 1
echo "✅ Tutti i processi sono stati terminati."
