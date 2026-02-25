#!/bin/bash

# Percorsi assoluti
SCRIPT_DIR="/home/pi"
DJI_SCRIPT="$SCRIPT_DIR/trackers/dji_receiver.py"
REMOTE_SCRIPT="$SCRIPT_DIR/remotetrack/main.py"
BRIDGECORE_SCRIPT="$SCRIPT_DIR/bridge/bridge_core.py"
BRIDGEWEB_SCRIPT="$SCRIPT_DIR/bridge/bridge_web.py"
ANTSDR_CONTROL="$SCRIPT_DIR/trackers/service_controller.sh"
OGN_DIR="$SCRIPT_DIR/ogn/rtlsdr-ogn"
OGN_RF="$OGN_DIR/ogn-rf"
OGN_DECODE="$OGN_DIR/ogn-decode"
OGN_CONF="$OGN_DIR/DSCNODE.conf"
APRS_LOCAL="$SCRIPT_DIR/bridge/aprs_local.py"

# Funzione per avvio con log minimale
start_background() {
  CMD="$1"
  NAME="$2"
  echo "▶️ Avvio: $CMD"
  nohup bash -c "$CMD" > "/home/pi/bridge/logs/log_${NAME}.log" 2>&1 &
  #nohup bash -c "$CMD" > "/home/pi/bridge/logs/log_general.log" 2>&1 &
}

# Cancelliamo i file di log
rm /home/pi/bridge/logs/aprs_raw.log
rm /home/pi/bridge/logs/ogn_rf.log
rm /home/pi/bridge/logs/ogn_decode.log
rm /home/pi/bridge/logs/bridge.log

# Avvia AntSDR
echo "🛰️ Avvio servizi AntSDR via SSH..."
bash "$ANTSDR_CONTROL" start
sleep 2

# Avvia ricezione DJI
start_background "python3 $DJI_SCRIPT --debug --mode legacy" "dji"

# Avvia ricezione Remote ID
cd /home/pi/remotetrack
/usr/bin/python3 -u main.py > /home/pi/bridge/logs/log_remote.log 2>&1 &
cd /home/pi
#start_background "cd /home/pi/remotetrack && /usr/bin/python3 main.py >> /home/pi/bridge/logs/log_remote.log"

# Avvia APRS server locale (per OGN → bridge)
echo "📡 Avvio APRS locale..."
start_background "python3 $APRS_LOCAL -log -logfile /home/pi/bridge/logs/aprs_raw.log" "aprs"
sleep 1

# Avvia ricezione OGN / FLARM
echo "🪂 Avvio OGN RF..."
#start_background "$OGN_RF $OGN_CONF" "ogn_rf"
screen -dmS ogn-rf bash -c "cd /home/pi/ogn/rtlsdr-ogn && ./ogn-rf DSCNODE.conf >> /home/pi/bridge/logs/ogn_rf.log"


sleep 4

echo "🪂 Avvio OGN decode..."
#start_background "$OGN_DECODE $OGN_CONF" "ogn_decode"
screen -dmS ogn-decode bash -c "cd /home/pi/ogn/rtlsdr-ogn && ./ogn-decode DSCNODE.conf >> /home/pi/bridge/logs/ogn_decode.log"



# Avvia bridge CORE
echo "🔁 Avvio bridge CORE..."
#nohup python3 "$BRIDGECORE_SCRIPT" > /home/pi/bridge_rest.log 2>&1 &
nohup python3 "$BRIDGECORE_SCRIPT" > /home/pi/bridge/logs/bridge.log 2>&1 &

# Avvia bridge WEB 
echo "🔁 Avvio bridge WEB..."
nohup python3 "$BRIDGEWEB_SCRIPT" > /dev/null 2>&1 &


echo "✅ Tutti i servizi sono stati avviati."
echo " "
echo " "
echo "Controlliamo la presenza dei servizi"

screen -ls
echo " "
ps -ef|grep aprs_local
echo " "
ps -ef|grep main.py
echo " "
echo " "
echo "Controlliamo se vi sono problemi sulla USB"
vcgencmd get_throttled
