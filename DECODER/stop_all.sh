#!/bin/bash

# Termina tutti i processi rilevanti avviati da start_all.sh
echo "🛑 Arresto di tutti i servizi..."

# Ferma servizi su AntSDR
#SCRIPT_DIR="/home/pi"
#ANTSDR_CONTROL="$SCRIPT_DIR/trackers/service_controller.sh"
#bash "$ANTSDR_CONTROL" stop

# Uccide gli script Python avviati manualmente o via nohup
pkill -f ogn-decode
pkill -f ogn-rf
pkill -f aprs_local.py
pkill -f dji_receiver.py
pkill -f bridge_core.py
pkill -f bridge_web.py
pkill -f main.py
#pkill -f mm2_aircraft_server.py
pkill -f socat

sleep 1
echo "✅ Tutti i processi sono stati terminati."
