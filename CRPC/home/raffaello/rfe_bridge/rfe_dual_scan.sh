#!/usr/bin/env bash
set -euo pipefail

PORT="/dev/ttyUSB0"
OUTDIR="/tmp/rfe/scan"
RING=10                 # quanti CSV tenere per banda (0..9)
SLEEP_BETWEEN=1         # secondi tra gli sweep
LOGTAG="[RFE-TRIPLE]"

# Bande
B24_START=2400
B24_END=2485
B52_START=5170          
B52_END=5250
B58_START=5725
B58_END=5875

# rfe_scan.py accanto a questo script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYSCAN="${SCRIPT_DIR}/rfe_scan.py"


FOCUS_JSON="/tmp/rfe/focus.json"
FAST_SWEEPS=3           # quanti sweep consecutivi quando c'è focus
FAST_SLEEP=0.25         # sleep più corto durante il focus

#
# SEZIONE ROTAZIONE ANTENNE
#

# --- GPIO helpers (SPDT & SP4T) ---
# SPDT1 (GPIO23): 1→RFE, 0→HackRF
spdt1_to_rfe()    { sudo pinctrl set 23 dh; }
spdt1_to_hackrf() { sudo pinctrl set 23 dl; }

# SPDT2 (GPIO24): 1→5.x GHz, 0→2.4 GHz
spdt2_5g() { sudo pinctrl set 24 dh; }
spdt2_24() { sudo pinctrl set 24 dl; }

# SP4T (GPIO17, GPIO27): N,E,S,O (vogliamo ruotare N→E→S→O)
sp4t_N() { sudo pinctrl set 17 dh; sudo pinctrl set 27 dl; } # Porta 2
sp4t_E() { sudo pinctrl set 17 dl; sudo pinctrl set 27 dl; } # Porta 1
sp4t_S() { sudo pinctrl set 17 dh; sudo pinctrl set 27 dh; } # Porta 4
sp4t_O() { sudo pinctrl set 17 dl; sudo pinctrl set 27 dh; } # Porta 3

# --- GPIO init (imposta tutti i pin in OUTPUT, livello basso di default) ---
gpio_init() {
  echo "[GPIO] Inizializzo pin..."
  for p in 17 23 24 27; do
    sudo pinctrl set $p op dl
  done
  sudo pinctrl set 23 dh  # SPDT1 → RFE
  sudo pinctrl set 24 dl  # SPDT2 → 2.4 GHz
  sudo pinctrl set 17 dh; sudo pinctrl set 27 dl  # SP4T → Nord
}

gpio_init


# Stato settore persistito (per continuare la rotazione al reboot)
SECTOR_STATE="/tmp/rfe/sector_state.json"
sector_next() {
  local order=(N E S O)
  local cur="N"
  if [[ -f "$SECTOR_STATE" ]]; then
    cur="$(sed -n 's/.*"sector"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$SECTOR_STATE" | head -1)"
    [[ -z "$cur" ]] && cur="N"
  fi
  local idx=0
  for i in "${!order[@]}"; do [[ "${order[$i]}" == "$cur" ]] && idx=$i && break; done
  local nxt="${order[$(((idx+1)%4))]}"
  echo "{\"sector\":\"$nxt\"}" > "$SECTOR_STATE"
  echo "$nxt"
}
apply_sector() {
  case "$1" in
    N) sp4t_N ;;
    E) sp4t_E ;;
    S) sp4t_S ;;
    O) sp4t_O ;;
  esac
  echo "$LOGTAG SP4T → $1"
}
#

get_focus_band() {
  if [[ -f "$FOCUS_JSON" ]]; then
    # estrai band e until_ts (senza dipendenze jq)
    local js; js="$(cat "$FOCUS_JSON" 2>/dev/null || true)"
    local band until
    band="$(echo "$js"  | sed -n 's/.*"band"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    until="$(echo "$js" | sed -n 's/.*"until_ts"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)"
    local now; now="$(date +%s)"
    # valida scadenza
    if [[ -n "$band" && -n "$until" ]]; then
      awk -v u="$until" -v n="$now" 'BEGIN{exit !(u>n)}' >/dev/null 2>&1 && echo "$band" && return 0
    fi
  fi
  return 1
}


mkdir -p "$OUTDIR"

echo "$LOGTAG Start dual sweep (Python) su $PORT → $OUTDIR (ring=$RING)"
i=0
while true; do

  # Durante lo scanning il front-end deve essere RFE
  spdt1_to_rfe

  # Ruota settore (N→E→S→O) ad ogni sweep completo triplo
  cur_sector="$(sector_next)"
  apply_sector "$cur_sector"


  focus_band="$(get_focus_band || true)"
  if [[ -n "$focus_band" ]]; then
    # scegli range/output in base alla banda focussata
    idx=$(( i % RING ))
    case "$focus_band" in
      24)
        for k in $(seq 1 $FAST_SWEEPS); do
          OUT24="$OUTDIR/24_${idx}.csv"
          echo "$LOGTAG [24★] focus sweep ${B24_START}-${B24_END} → $OUT24"
          # 2.4 GHz
          spdt2_24
          # scansione
          python3 "$PYSCAN" --port "$PORT" --band 24 -s $B24_START -e $B24_END --out "$OUT24" -v \
            || echo "$LOGTAG WARN: sweep 24 fallito"
          ln -sfn "$OUT24" "$OUTDIR/latest_24.csv"
          sleep "$FAST_SLEEP"
          idx=$(( (idx+1) % RING ))
        done
        i=$((i + 1))
        continue
      ;;
      52)
        for k in $(seq 1 $FAST_SWEEPS); do
          OUT52="$OUTDIR/52_${idx}.csv"
          echo "$LOGTAG [52★] focus sweep ${B52_START}-${B52_END} → $OUT52"
          # 5.2 GHz
          spdt2_5g
          # scansione
          python3 "$PYSCAN" --port "$PORT" --band 52 -s $B52_START -e $B52_END --out "$OUT52" -v \
            || echo "$LOGTAG WARN: sweep 52 fallito"
          ln -sfn "$OUT52" "$OUTDIR/latest_52.csv"
          sleep "$FAST_SLEEP"
          idx=$(( (idx+1) % RING ))
        done
        i=$((i + 1))
        continue
      ;;
      58)
        for k in $(seq 1 $FAST_SWEEPS); do
          OUT58="$OUTDIR/58_${idx}.csv"
          echo "$LOGTAG [58★] focus sweep ${B58_START}-${B58_END} → $OUT58"
          # 5.8 GHz
          spdt2_5g
          # scansione
          python3 "$PYSCAN" --port "$PORT" --band 58 -s $B58_START -e $B58_END --out "$OUT58" -v \
            || echo "$LOGTAG WARN: sweep 58 fallito"
          ln -sfn "$OUT58" "$OUTDIR/latest_58.csv"
          sleep "$FAST_SLEEP"
          idx=$(( (idx+1) % RING ))
        done
        i=$((i + 1))
        continue
      ;;
    esac
  fi


  idx=$(( i % RING ))
  # 2.4 GHz
  OUT24="$OUTDIR/24_${idx}.csv"
  echo "$LOGTAG [24] sweep ${B24_START}-${B24_END} → $OUT24"
  python3 "$PYSCAN" --port "$PORT" --band 24 -s $B24_START -e $B24_END --out "$OUT24" -v \
    || echo "$LOGTAG WARN: sweep 24 fallito"
  ln -sfn "$OUT24" "$OUTDIR/latest_24.csv"

  sleep "$SLEEP_BETWEEN"

  # 5.2 GHz  ⬅️ NUOVO BLOCCO
  OUT52="$OUTDIR/52_${idx}.csv"
  echo "$LOGTAG [52] sweep ${B52_START}-${B52_END} → $OUT52"
  python3 "$PYSCAN" --port "$PORT" --band 52 -s $B52_START -e $B52_END --out "$OUT52" -v \
    || echo "$LOGTAG WARN: sweep 52 fallito"
  ln -sfn "$OUT52" "$OUTDIR/latest_52.csv"

  sleep "$SLEEP_BETWEEN"
  
  # 5.8 GHz
  OUT58="$OUTDIR/58_${idx}.csv"
  echo "$LOGTAG [58] sweep ${B58_START}-${B58_END} → $OUT58"
  python3 "$PYSCAN" --port "$PORT" --band 58 -s $B58_START -e $B58_END --out "$OUT58" -v \
    || echo "$LOGTAG WARN: sweep 58 fallito"
  ln -sfn "$OUT58" "$OUTDIR/latest_58.csv"

  sleep "$SLEEP_BETWEEN"
  i=$((i + 1))

  
done

