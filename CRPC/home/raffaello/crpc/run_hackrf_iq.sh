#!/usr/bin/env bash
set -euo pipefail

# ================== DEFAULTS ==================
BAND=""                 # 24 | 58 (obbligatorio)
F0_MHZ=""               # centro in MHz (obbligatorio)
BW_HZ=""                # opzionale (informativo per la pipeline)
SPS="10000000"          # sample rate (Hz)
SECONDS="10"            # durata cattura (s)

# Guadagni / bias-T (regola a piacere)
ANT_PWR=1               # 1 = bias-T ON, 0 = OFF
LNA_GAIN=16
VGA_GAIN=32

# Safety
MIN_FREE_MB=512         # non scrivere se /tmp < 512MB
# ==============================================

usage() {
  echo "Uso: $0 --band {24|58} --f0 <MHz> [--bw <Hz>] [--sps <Hz>] [--seconds <s>] [--out <path>]" >&2
  exit 2
}

# Parse argomenti
OUT_PATH=""  # opzionale; di default /tmp/hackrf_${BAND}.iq (FIFO o file)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --band)    BAND="$2"; shift 2;;
    --f0)      F0_MHZ="$2"; shift 2;;
    --bw)      BW_HZ="$2"; shift 2;;
    --sps)     SPS="$2"; shift 2;;
    --seconds) SECONDS="$2"; shift 2;;
    --out)     OUT_PATH="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Argomento sconosciuto: $1" >&2; usage;;
  esac
done

# Validazioni minime
[[ -z "$BAND"   ]] && { echo "Manca --band"; usage; }
[[ -z "$F0_MHZ" ]] && { echo "Manca --f0 (MHz)"; usage; }
if [[ "$BAND" != "24" && "$BAND" != "58" ]]; then
  echo "Valore --band non valido: $BAND (usa 24 o 58)"; exit 2
fi

# Output path di default
if [[ -z "$OUT_PATH" ]]; then
  OUT_PATH="/tmp/hackrf_${BAND}.iq"
fi

# Free space guard
FREE_MB=$(df --output=avail -m /tmp | tail -1 | tr -d ' ')
if [[ "${FREE_MB:-0}" -lt "$MIN_FREE_MB" ]]; then
  echo "⚠️  /tmp libero ${FREE_MB}MB (<${MIN_FREE_MB}). Salto cattura."
  exit 0
fi

# Converti MHz → Hz senza bc (usa awk)
F0_HZ=$(awk -v m="$F0_MHZ" 'BEGIN{printf "%.0f", m*1000000}')

# Log informativo
echo "▶ HackRF capture — band=$BAND  f0=${F0_MHZ}MHz (${F0_HZ} Hz)  sps=${SPS}  dur=${SECONDS}s  out=${OUT_PATH}"
[[ -n "${BW_HZ}" ]] && echo "   (bw richiesta ≈ ${BW_HZ} Hz)"

# Se OUT_PATH è una FIFO, scriveremo lì; altrimenti creeremo un file IQ.
IS_FIFO=0
if [[ -p "$OUT_PATH" ]]; then
  IS_FIFO=1
fi

# Per retrocompatibilità: aggiorna i “center files” usati dalla dashboard (se li hai)
if [[ "$BAND" == "24" ]]; then
  echo "$F0_HZ" > /tmp/center_24.txt || true
else
  echo "$F0_HZ" > /tmp/center_58.txt || true
fi

# Avvio cattura
# Nota: hackrf_transfer ritorna non-zero se interrotto da timeout, quindi lo wrappiamo
set +e
timeout "${SECONDS}s" \
  hackrf_transfer \
    -f "$F0_HZ" \
    -s "$SPS" \
    -a "$ANT_PWR" \
    -l "$LNA_GAIN" \
    -g "$VGA_GAIN" \
    -r "$OUT_PATH"
RC=$?
set -e

# Un RC=124 è il normale esito del timeout → ok
if [[ "$RC" -ne 0 && "$RC" -ne 124 ]]; then
  echo "❌ hackrf_transfer exit code $RC"
  exit "$RC"
fi

echo "⏹️  Cattura terminata."

# Hook post-processing (opzionale): se vuoi generare tiles/waterfall al volo
# Esempi:
# if [[ "$IS_FIFO" -eq 0 ]]; then
#   python3 /home/raffaello/crpc/iq_to_tiles.py \
#     --in "$OUT_PATH" --band "$BAND" --center "$F0_MHZ" --bw "${BW_HZ:-0}" || true
# fi

