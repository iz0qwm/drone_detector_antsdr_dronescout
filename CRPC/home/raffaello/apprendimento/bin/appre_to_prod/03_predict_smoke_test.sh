#!/usr/bin/env bash
set -euo pipefail

# --- Config di default (overridable via env o argv) ---
YOLO_BIN="${YOLO_BIN:-/home/raffaello/yolo-venv/bin/yolo}"
VENVPY="${VENVPY:-/home/raffaello/yolo-venv/bin/python}"
DATA_YAML="${2:-/home/raffaello/dataset/yolo_custom/data.yaml}"

if [ -n "${1:-}" ]; then
  WEIGHTS="$1"
else
  # prova a leggere il run più recente (senza dipendere dal symlink)
  RUNS="/home/raffaello/yolo_runs"
  latest=$(ls -1dt "$RUNS"/yolo_custom* 2>/dev/null | head -n1 || true)
  if [ -n "$latest" ] && [ -f "$latest/weights/best.pt" ]; then
    WEIGHTS="$latest/weights/best.pt"
  else
    # fallback al symlink storico
    WEIGHTS="/home/raffaello/yolo_runs/yolo_custom/weights/best.pt"
  fi
fi

# Cum tiles dalle recordings (default)
SRC_GLOB="${3:-/home/raffaello/apprendimento/data/recordings/*/*/tiles/*_cum_*.png}"

IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.10}"            # basso per smoke; il tracker filtrerà a 0.30
DEVICE="${YOLO_DEVICE:-}"        # "" | "cpu" | "0"
LIMIT="${LIMIT:-64}"             # numero max immagini per test
PROJECT="${PROJECT:-/home/raffaello/yolo_runs}"
RUN_NAME="${RUN_NAME:-smoke_test}"

echo "== YOLO predict smoke test =="
echo "weights: $WEIGHTS"
echo "data:    $DATA_YAML"
echo "source:  $SRC_GLOB"
echo "imgsz:   $IMGSZ   conf: $CONF   device: ${DEVICE:-auto}   limit: $LIMIT"

# --- Espandi il glob e campiona fino a LIMIT ---
shopt -s nullglob
mapfile -t FILES < <(ls -1 ${SRC_GLOB} 2>/dev/null | head -n "$LIMIT")
shopt -u nullglob

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "!! Nessun file trovato per il glob: $SRC_GLOB"
  exit 2
fi

# --- Se c'è il binario 'yolo', usalo; altrimenti fallback Python con riepilogo ---
if [ -x "$YOLO_BIN" ]; then
  tmpdir=$(mktemp -d /tmp/yolo_smoke_XXXX)
  # usa symlink per non copiare dati
  for f in "${FILES[@]}"; do ln -s "$f" "$tmpdir/"; done
  "$YOLO_BIN" detect predict \
    model="$WEIGHTS" source="$tmpdir" \
    imgsz="$IMGSZ" conf="$CONF" ${DEVICE:+device=$DEVICE} \
    project="$PROJECT" name="$RUN_NAME" \
    save_txt=True save_conf=True
  rm -rf "$tmpdir"
else
  "$VENVPY" - "$WEIGHTS" "$IMGSZ" "$CONF" "$DEVICE" "$PROJECT" "$RUN_NAME" "${FILES[@]}" <<'PY'
import sys, statistics
from ultralytics import YOLO
weights, imgsz, conf, device, project, name, *files = sys.argv[1:]
imgsz = int(imgsz); conf = float(conf)
kwargs = dict(imgsz=imgsz, conf=conf, save_txt=True, save_conf=True, project=project, name=name)
if device: kwargs["device"] = device
model = YOLO(weights)

# stream=True per iterare i risultati
results = list(model.predict(files, stream=True, **kwargs))

# Riepilogo: conteggio e conf media per classe
from collections import defaultdict
counts = defaultdict(int)
confs = defaultdict(list)
names = model.names

for r in results:
    for b in r.boxes:
        cls = int(b.cls.item())
        c = float(b.conf.item())
        counts[cls] += 1
        confs[cls].append(c)

print("\n=== SUMMARY ===")
total = 0
for cls, n in sorted(counts.items()):
    total += n
    avg = statistics.mean(confs[cls]) if confs[cls] else 0.0
    print(f"{cls:>2} {names.get(cls, str(cls)):<15}  n={n:>4}  avg_conf={avg:0.3f}")
print(f"TOTAL detections: {total}")
PY
fi
