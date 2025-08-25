#!/usr/bin/env bash
set -euo pipefail
# Config
VENVPY="/home/raffaello/yolo-venv/bin/python"
YOLO_BIN="${YOLO_BIN:-/home/raffaello/yolo-venv/bin/yolo}"

DATA_YAML="${1:-/home/raffaello/dataset/yolo_custom/data.yaml}"
RUNS_DIR="${2:-/home/raffaello/yolo_runs}"
RUN_NAME="${3:-yolo_custom}"
ARCH="${4:-yolov8n.pt}"   # yolov8n.pt | yolov8s.pt | ...

EPOCHS="${EPOCHS:-60}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1337}"

echo "== YOLO train =="
echo "venv:   $VENVPY"
echo "data:   $DATA_YAML"
echo "runs:   $RUNS_DIR name=$RUN_NAME"
echo "arch:   $ARCH"
echo "epochs: $EPOCHS imgsz=$IMGSZ batch=$BATCH workers=$WORKERS seed=$SEED"

if [ -x "$YOLO_BIN" ]; then
  # Preferisci la CLI se presente
  "$YOLO_BIN" detect train \
    data="$DATA_YAML" model="$ARCH" \
    epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" \
    project="$RUNS_DIR" name="$RUN_NAME" \
    cache=True workers="$WORKERS" seed="$SEED" \
    degrees=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 \
    hsv_h=0.0 hsv_s=0.0 hsv_v=0.0
else
  # Fallback: API Python (funziona anche senza CLI)
  "$VENVPY" - "$DATA_YAML" "$RUNS_DIR" "$RUN_NAME" "$ARCH" "$EPOCHS" "$IMGSZ" "$BATCH" "$WORKERS" "$SEED" <<'PY'
import sys
from ultralytics import YOLO
data_yaml, runs_dir, run_name, arch, epochs, imgsz, batch, workers, seed = sys.argv[1:10]
epochs, imgsz, batch, workers, seed = map(int, (epochs, imgsz, batch, workers, seed))
m = YOLO(arch)
m.train(data=data_yaml, project=runs_dir, name=run_name,
        epochs=epochs, imgsz=imgsz, batch=batch, cache=True,
        workers=workers, seed=seed)
print(f"== Done. Weights at: {runs_dir}/{run_name}/weights/best.pt")
PY
fi

# --- AUTOPROMOTE: punta yolo_custom -> ultimo run che inizia per yolo_custom
# Nota: Ultralytics può rinominare in yolo_custom2,3,4... se la cartella esiste già
latest=$(ls -1dt "$RUNS_DIR"/yolo_custom* | head -n1)
ln -sfn "$latest" "$RUNS_DIR/yolo_custom"
echo "→ Promoted symlink: $RUNS_DIR/yolo_custom -> $latest"
