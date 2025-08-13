#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
# FIFO (una per banda)
FIFO_24="/tmp/hackrf_24.iq"
FIFO_58="/tmp/hackrf_58.iq"

# HackRF sample rate (Hz). 8e6–12e6 di solito sono stabili.
SR=10000000

# Overlap tra centri: STEP in MHz (<= SR/1e6). Esempio: SR=10Msps → STEP=8 → 20% overlap
STEP_MHZ=8

# Bande da coprire (Hz)
B24_LO=2400000000
B24_HI=2500000000

B58_LO=5725000000
B58_HI=5875000000

# Guadagni / alimentazione antenna
ANT_PWR=1       # 1 se ti serve bias-T, altrimenti 0
LNA_GAIN=16
VGA_GAIN=32

# Durata di campionamento per ogni center (secondi) e pausa
CHUNK_SEC=2
SLEEP_BETWEEN=0.2

# Safety
MIN_FREE_MB=200
# ============================================

get_free_mb() { df --output=avail -m /tmp | tail -1; }

mkfifo_safe () {
  local p="$1"
  if [[ -p "$p" ]]; then return; fi
  rm -f "$p" || true
  mkfifo "$p"
}

# Crea lista di center frequencies per coprire [LO, HI) con passo STEP_MHZ e un minimo di overlap
make_centers () {
  local LO=$1 HI=$2 STEP=$3
  awk -v lo="$LO" -v hi="$HI" -v stepmhz="$STEP" '
    BEGIN {
      step = stepmhz*1e6;
      # Primo centro: lo + BW/2 (BW ~= SR)
      bw = '"$SR"';
      c = lo + bw/2;
      # Se il primo centro sfora verso hi, riportalo
      if (c - bw/2 < lo) c = lo + bw/2;
      while (c + bw/2 <= hi + 1) {
        printf "%.0f\n", c;
        c += step;
      }
    }'
}

band_loop () {
  local BAND_LO="$1" BAND_HI="$2" FIFO="$3" NAME="$4"
  local centers=()
  while IFS= read -r cf; do centers+=("$cf"); done < <(make_centers "$BAND_LO" "$BAND_HI" "$STEP_MHZ")
  if [ "${#centers[@]}" -eq 0 ]; then
    echo "⚠️ Nessun center calcolato per banda $NAME"; sleep 1; return
  fi
  for CF in "${centers[@]}"; do
    local free; free="$(get_free_mb)"
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
      echo "⛔ /tmp libero ${free}MB (<${MIN_FREE_MB}). Attendo…"; sleep 2; continue
    fi
    echo "📡 $NAME  CF=$CF  SR=$SR  (${CHUNK_SEC}s)"
    # salva centro banda corrente
    if [[ "$NAME" == "2.4GHz" ]]; then
      echo "$CF" > /tmp/center_24.txt
    else
      echo "$CF" > /tmp/center_58.txt
    fi

    # Nota: -r scrive IQ int8 interleaved (I,Q) sulla FIFO
    timeout "${CHUNK_SEC}s" \
      hackrf_transfer -f "$CF" -s "$SR" -a "$ANT_PWR" -l "$LNA_GAIN" -g "$VGA_GAIN" -r "$FIFO" || true
    sleep "$SLEEP_BETWEEN"
  done
}

cleanup () { echo "🧹 stop"; exit 0; }
trap cleanup INT TERM

echo "📦 preparo FIFO…"
mkfifo_safe "$FIFO_24"
mkfifo_safe "$FIFO_58"

echo "✅ IQ sweep alternato — SR=$SR, STEP=${STEP_MHZ}MHz, chunk=${CHUNK_SEC}s"
while true; do
  band_loop "$B24_LO" "$B24_HI" "$FIFO_24" "2.4GHz"
  band_loop "$B58_LO" "$B58_HI" "$FIFO_58" "5.8GHz"
done
