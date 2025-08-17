#!/usr/bin/env bash
set -euo pipefail
VENVPY="/home/raffaello/yolo-venv/bin/python"
YOLO_BIN="${YOLO_BIN:-/home/raffaello/yolo-venv/bin/yolo}"

WEIGHTS="${1:-/home/raffaello/yolo_runs/yolo_custom/weights/best.pt}"
DATA_YAML="${2:-/home/raffaello/dataset/yolo_custom/data.yaml}"
SRC_GLOB="${3:-/tmp/tiles/*_cum_*.png}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"

echo "== YOLO predict smoke test =="
echo "weights: $WEIGHTS"
echo "data:    $DATA_YAML"
echo "source:  $SRC_GLOB"

if [ -x "$YOLO_BIN" ]; then
  "$YOLO_BIN" detect predict \
    model="$WEIGHTS" data="$DATA_YAML" source="$SRC_GLOB" \
    imgsz="$IMGSZ" conf="$CONF"
else
  "$VENVPY" - "$WEIGHTS" "$DATA_YAML" "$SRC_GLOB" "$IMGSZ" "$CONF" <<'PY'
import sys
from ultralytics import YOLO
weights, data_yaml, src_glob, imgsz, conf = sys.argv[1:6]
imgsz = int(imgsz); conf = float(conf)
m = YOLO(weights)
m.predict(data=data_yaml, source=src_glob, imgsz=imgsz, conf=conf)
PY
fi

