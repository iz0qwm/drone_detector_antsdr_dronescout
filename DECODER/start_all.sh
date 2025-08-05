#!/bin/bash

# Percorsi assoluti (modifica se necessario)
SCRIPT_DIR="/home/pi"
DJI_SCRIPT="$SCRIPT_DIR/trackers/dji_receiver.py"
REMOTE_SCRIPT="$SCRIPT_DIR/remotetrack/main.py"
BRIDGE_SCRIPT="$SCRIPT_DIR/bridge_uploader.py"
ANTSDR_CONTROL="$SCRIPT_DIR/trackers/service_controller.sh"

# Funzione per avvio con log minimale
start_background() {
  CMD="$1"
  echo "▶️ Avvio: $CMD"
  nohup $CMD > /dev/null 2>&1 &
}

# Avvia AntSDR (controlla se start funziona via SSH)
echo "🛰️ Avvio servizi AntSDR via SSH..."
bash "$ANTSDR_CONTROL" start
sleep 2

# Avvia ricezione DJI
start_background "python3 $DJI_SCRIPT --debug"

# Avvia ricezione Remote ID
start_background "python3 $REMOTE_SCRIPT"

# Avvia bridge uploader unificato
nohup ~/.pyenv/versions/3.10.14/bin/python $BRIDGE_SCRIPT > /home/pi/bridge.log 2>&1 &

#start_background "python3 $BRIDGE_SCRIPT"

echo "✅ Tutti i servizi sono stati avviati."

