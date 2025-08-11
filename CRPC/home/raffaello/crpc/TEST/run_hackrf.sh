#!/usr/bin/env bash
set -euo pipefail

##### CONFIG #####
OUT_BASE="/tmp/hackrf_sweeps"     # cartella per i .bin se non usi le FIFO
BAND_24="2400:2500"               # 2.4 GHz band
BAND_58="5725:5875"               # 5.8 GHz band
SWEEP_WIDTH=1000000               # Hz per step (1 MHz)
LNA_GAIN=16                       # preamp gain (dB)
VGA_GAIN=32                       # IF gain (dB)
SECONDS_PER_BAND=8                # durata per banda prima di switchare
SLEEP_BETWEEN=1                   # pausa tra gli switch (s)
##################

mkdir -p "$OUT_BASE"

# piccola utility per creare nome file pulito dalla banda (es. 2400_2500)
band_tag () { echo "$1" | tr ':' '_' ; }

# Se esistono FIFO con questi nomi, lo script scriverà lì,
# altrimenti scrive su file .bin timestampati dentro OUT_BASE.
FIFO_24="/tmp/hackrf_$(band_tag "$BAND_24").fifo"
FIFO_58="/tmp/hackrf_$(band_tag "$BAND_58").fifo"

# Controllo dispositivo
if ! hackrf_info >/dev/null 2>&1 ; then
  echo "❌ HackRF non rilevato. Collega il dispositivo o verifica i permessi (udev)."
  exit 1
fi

echo "✅ HackRF OK. Avvio sweep alternato su $BAND_24 e $BAND_58"
echo "   LNA_GAIN=$LNA_GAIN  VGA_GAIN=$VGA_GAIN  WIDTH=$SWEEP_WIDTH Hz  ${SECONDS_PER_BAND}s per banda"
echo "   FIFO 2.4: $FIFO_24  |  FIFO 5.8: $FIFO_58 (se presenti)"
echo "   Output files (fallback): $OUT_BASE"

run_band () {
  local BAND="$1"
  local TAG
  TAG="$(band_tag "$BAND")"

  # Scegli destinazione: FIFO se esiste, altrimenti file
  if [[ -p "/tmp/hackrf_${TAG}.fifo" ]]; then
    echo "▶️  [$BAND] streaming verso FIFO /tmp/hackrf_${TAG}.fifo"
    timeout "${SECONDS_PER_BAND}s" \
      hackrf_sweep -f "$BAND" -w "$SWEEP_WIDTH" -l "$LNA_GAIN" -g "$VGA_GAIN" -r - \
      > "/tmp/hackrf_${TAG}.fifo"
  else
    local TS OUT
    TS="$(date +%Y%m%d_%H%M%S)"
    OUT="${OUT_BASE}/sweep_${TAG}_${TS}.bin"
    echo "💾 [$BAND] scrivo su file: $OUT"
    timeout "${SECONDS_PER_BAND}s" \
      hackrf_sweep -f "$BAND" -w "$SWEEP_WIDTH" -l "$LNA_GAIN" -g "$VGA_GAIN" -r - \
      > "$OUT"
  fi
}

# Loop infinito alternando le due bande
while true; do
  run_band "$BAND_24"
  sleep "$SLEEP_BETWEEN"
  run_band "$BAND_58"
  sleep "$SLEEP_BETWEEN"
done

