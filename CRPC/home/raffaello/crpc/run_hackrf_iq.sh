#!/usr/bin/env bash
set -euo pipefail

# ================== DEFAULTS ==================
BAND=""                 # 24 | 58 (obbligatorio)
F0_MHZ=""               # centro in MHz (obbligatorio)
BW_HZ=""                # opzionale (informativo per la pipeline)
SPS="10000000"          # sample rate (Hz)
SECONDS="10"            # durata cattura (s)

# Guadagni / bias-T (regola a piacere)
ANT_PWR=0               # 1 = bias-T ON, 0 = OFF
LNA_GAIN=16
VGA_GAIN=32

# Safety
MIN_FREE_MB=512         # non scrivere se /tmp < 512MB
# ==============================================

usage() {
  echo "Uso: $0 --band {24|52|58} --f0 <MHz> [--bw <Hz>] [--sps <Hz>] [--seconds <s>] [--out <path>] [--lna <value gain>] [--vga <value gain>] [--ant <0/1 bias-T]" >&2
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
    --lna)     LNA_GAIN="$2"; shift 2;;
    --vga)     VGA_GAIN="$2"; shift 2;;
    --ant)     ANT_PWR="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Argomento sconosciuto: $1" >&2; usage;;
  esac
done

# Validazioni minime
[[ -z "$BAND"   ]] && { echo "Manca --band"; usage; }
[[ -z "$F0_MHZ" ]] && { echo "Manca --f0 (MHz)"; usage; }
if [[ "$BAND" != "24" && "$BAND" != "58" && "$BAND" != "52" ]]; then
  echo "Valore --band non valido: $BAND (usa 24, 52 o 58)"; exit 2
fi

# Output path di default
if [[ -z "$OUT_PATH" ]]; then
  OUT_PATH="/tmp/hackrf_${BAND}.iq"
fi

# Preset per banda (se non sono stati forzati via CLI o ENV)
: "${FORCED_LNA:=${LNA_GAIN}}"
: "${FORCED_VGA:=${VGA_GAIN}}"
: "${FORCED_ANT:=${ANT_PWR}}"

if [[ -z "${OVERRIDE_DONE:-}" ]]; then
  case "$BAND" in
    "24")
      # ambiente affollato → preset più conservativi
      [[ "${FORCED_LNA}" == "${LNA_GAIN}" ]] && LNA_GAIN=20
      [[ "${FORCED_VGA}" == "${VGA_GAIN}" ]] && VGA_GAIN=34
      [[ "${FORCED_ANT}" == "${ANT_PWR}"   ]] && ANT_PWR="${ANT_PWR:-0}"
      ;;
    "52"|"58")
      # più “pulito” → spingiamo di più
      [[ "${FORCED_LNA}" == "${LNA_GAIN}" ]] && LNA_GAIN=28
      [[ "${FORCED_VGA}" == "${VGA_GAIN}" ]] && VGA_GAIN=40
      [[ "${FORCED_ANT}" == "${ANT_PWR}"   ]] && ANT_PWR="${ANT_PWR:-0}"
      ;;
  esac
fi

# Permetti override via variabili d’ambiente (se settate)
# es: EXT_LNA=1 per alimentare un LNA esterno
if [[ "${EXT_LNA:-0}" == "1" ]]; then
  ANT_PWR=1
  # con LNA esterno, tieni LNA interno un filo più basso
  if [[ "$BAND" == "52" || "$BAND" == "58" ]]; then LNA_GAIN=${LNA_GAIN:-24}; fi
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

# Aggiorna i “center files” usati dalla dashboard/tiles
case "$BAND" in
  "24") echo "$F0_HZ" > /tmp/center_24.txt || true ;;
  "58") echo "$F0_HZ" > /tmp/center_58.txt || true ;;
  "52") echo "$F0_HZ" > /tmp/center_52.txt || true ;;
esac

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

