#!/usr/bin/env bash
set -euo pipefail

# --- Output directory (testo) ---
OUT_DIR="/tmp/hackrf_sweeps_text"
mkdir -p "$OUT_DIR"

# --- Bande da ciclare (50 MHz ciascuna) ---
BAND_24_A="2400:2450"
BAND_24_B="2450:2500"
BAND_58_A="5725:5780"
BAND_58_B="5780:5830" 

# --- Parametri hackrf_sweep (testo, NO -r) ---
SWEEP_WIDTH=500000    # 500 kHz
LNA_GAIN=16
VGA_GAIN=32
SECONDS_PER_BAND=6
SLEEP_BETWEEN=1

# --- Fail-safe: minimo spazio libero su /tmp (MB) ---
MIN_FREE_MB=200

get_free_mb() {
  df --output=avail -m /tmp | tail -1
}

emergency_trim() {
  # cancella i file più vecchi per liberare spazio
  local DIR="$OUT_DIR"
  echo "⚠️  Spazio basso, pulizia in $DIR..."
  find "$DIR" -type f -name "sweep_*.txt" -printf '%T@ %p\n' \
    | sort -n \
    | head -n 50 \
    | awk '{print $2}' \
    | xargs -r rm -f
}

# Tag esattamente come richiesto da sweep_to_tiles.py
# Cambia band_tag per mappare i 4 range nei 2 nomi "storici"
band_tag() {
  case "$1" in
    2400:2450|2450:2500) echo "2400_2500" ;;
    5725:5780|5780:5830) echo "5725_5875" ;;  # nome "storico" per compatibilità
    *)                    echo "${1//:/_}" ;;
  esac
}

# (facoltativo) sistema il messaggio di avvio che ora cita variabili inesistenti
echo "✅ Avvio sweep alternato su $BAND_24_A, $BAND_24_B, $BAND_58_A, $BAND_58_B (testo, nomi compatibili)"


run_band () {
  local BAND="$1"
  local TAG OUT
  TAG="$(band_tag "$BAND")"
  OUT="${OUT_DIR}/sweep_${TAG}.txt"

  # 🔒 fail-safe spazio /tmp
  local free
  free="$(get_free_mb)"
  if [ "$free" -lt "$MIN_FREE_MB" ]; then
    echo "⛔ Spazio libero su /tmp: ${free}MB (<${MIN_FREE_MB}MB)."
    emergency_trim
    echo "⏭️  Salto sweep banda $BAND per evitare overflow."
    return
  fi

  echo "💾 [$BAND] append su: $OUT"
  # Produciamo TESTO (niente -r). Append per mantenere file “vivo”.
  timeout "${SECONDS_PER_BAND}s" \
    hackrf_sweep -f "$BAND" -w "$SWEEP_WIDTH" -l "$LNA_GAIN" -g "$VGA_GAIN" \
    >> "$OUT" || true

  # Troncatura soft se il file cresce troppo (~50MB)
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt $((50*1024*1024)) ]; then
    tail -n 20000 "$OUT" > "${OUT}.tmp" && mv "${OUT}.tmp" "$OUT"
  fi
}

#echo "✅ Avvio sweep alternato su $BAND_24 e $BAND_58 (testo, nomi fissi)"
while true; do
  run_band "$BAND_24_A"; sleep "$SLEEP_BETWEEN"
  run_band "$BAND_24_B"; sleep "$SLEEP_BETWEEN"
  run_band "$BAND_58_A"; sleep "$SLEEP_BETWEEN"
  run_band "$BAND_58_B"; sleep "$SLEEP_BETWEEN"
done

