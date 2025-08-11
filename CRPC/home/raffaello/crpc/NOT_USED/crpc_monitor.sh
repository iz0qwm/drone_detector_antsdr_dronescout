#!/usr/bin/env bash
# CRPC live monitor – compatto e leggibile
# dipendenze utili: jq, tput (già in ncurses-base)

# --- CONFIG ---
REFRESH=2                      # secondi tra aggiornamenti
USER=raffaello
CRPC_DIR="/home/$USER/crpc"
LOG_DIR="/tmp/crpc_logs"
TILES="/tmp/tiles"
TILES_DONE="/tmp/tiles_done"
SWEEP24="/tmp/hackrf_sweeps_text/sweep_2400_2500.txt"
SWEEP58="/tmp/hackrf_sweeps_text/sweep_5725_5875.txt"
DET="$LOG_DIR/detections.jsonl"
TRACKS_CURR="$LOG_DIR/tracks_current.json"
ASSOC="$LOG_DIR/associations.jsonl"
ASSOC_LOG="$LOG_DIR/assoc.log"    # nuovo: log servizio associatore

has_cmd(){ command -v "$1" >/dev/null 2>&1; }
jcat(){  # pretty JSON if jq exists
  if has_cmd jq; then jq -r '.' 2>/dev/null || cat
  else cat
  fi
}
hr(){ printf '%*s\n' "$(tput cols 2>/dev/null || echo 80)" '' | tr ' ' -; }

while true; do
  clear
  echo "🚦 CRPC Monitor — $(date '+%F %T')"
  hr

  # SERVICES
  echo "🧩 Services:"
  for s in crpc-prepare crpc-sweep crpc-tiles crpc-yolo crpc-tracker crpc-rfscan; do
    st=$(systemctl is-active "$s" 2>/dev/null || echo " ")
    printf "  - %-13s : %s\n" "$s" "$st"
  done

  # DISK /tmp
  echo
  echo "💾 /tmp usage:"
  df -h /tmp | awk 'NR==1 || NR==2 {print "  "$0}'

  # SWEEP
  echo
  echo "📡 Sweep last lines (24):"
  if [[ -s "$SWEEP24" ]]; then tail -n 3 "$SWEEP24"; else echo "  (no data)"; fi
  echo "📡 Sweep last lines (58):"
  if [[ -s "$SWEEP58" ]]; then tail -n 3 "$SWEEP58"; else echo "  (no data)"; fi

  # TILES
  echo
  echo "🖼️  Tiles:"
  c_new=$(ls -1 "$TILES"/*.png 2>/dev/null | wc -l)
  c_done=$(ls -1 "$TILES_DONE"/*.png 2>/dev/null | wc -l)
  echo "  in-queue: $c_new   done: $c_done"
  echo "  latest  :"
  ls -lt "$TILES"/*.png 2>/dev/null | head -n 3 | awk '{print "   "$0}'
  ls -lt "$TILES_DONE"/*.png 2>/dev/null | head -n 3 | awk '{print "   "$0}'

  # DETECTIONS
  echo
  echo "🎯 Detections (tail 5):"
  if [[ -s "$DET" ]]; then
    tail -n 5 "$DET" | jcat | sed 's/^/  /'
  else
    echo "  (empty)"
  fi

  # TRACKS
  echo
  echo "🧭 Tracks current:"
  if [[ -s "$TRACKS_CURR" ]]; then
    if has_cmd jq; then
      jq -c '.[] | {id:.track_id, band, fMHz:.center_freq_mhz, bw:.bandwidth_mhz, hop:.hop_rate_mhz_s, len, last:.last_seen}' "$TRACKS_CURR" 2>/dev/null | sed 's/^/  /'
    else
      cat "$TRACKS_CURR" | sed 's/^/  /'
    fi
  else
    echo "  (no tracks)"
  fi

  # ASSOCIATIONS
  echo
  echo "🔗 Associations (tail 5):"
  if [[ -s "$ASSOC" ]]; then
    tail -n 5 "$ASSOC" | jcat | sed 's/^/  /'
  else
    echo "  (empty)"
  fi

  # 📜 ASSOC LOG con controllo dimensione
  echo
  echo "📜 Assoc log (tail 10):"
  if [[ -s "$ASSOC_LOG" ]]; then
    # Se log > 5 MB, tronca alle ultime 500 righe
    if [[ $(stat -c%s "$ASSOC_LOG") -gt $((5*1024*1024)) ]]; then
      tail -n 500 "$ASSOC_LOG" > "${ASSOC_LOG}.tmp" && mv "${ASSOC_LOG}.tmp" "$ASSOC_LOG"
      echo "  (log troncato a 500 righe perché superava 5MB)"
    fi
    tail -n 10 "$ASSOC_LOG" | sed 's/^/  /'
  else
    echo "  (no log)"
  fi

  # LAST ERRORS (from services)
  echo
  echo "⚠️  Recent warnings (services, last 50 lines):"
  journalctl -u crpc-sweep -u crpc-tiles -u crpc-yolo -u crpc-tracker -u crpc-assoc --no-pager -n 50 \
    | egrep -i "error|fail|no space|oom|timeout|cannot|traceback" --color=never || echo "  (none)"

  hr
  echo "Tip: Ctrl+C per uscire • refresh=${REFRESH}s"
  sleep "$REFRESH"
done

