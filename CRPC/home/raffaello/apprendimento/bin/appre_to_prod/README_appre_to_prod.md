# apprendimento → produzione (CRPC)

Percorso: `/home/crpc/appre_to_prod/`

## Ordine operativo

1) Genera dataset YOLO dalle recordings
```bash
python3 01_make_yolo_from_recordings.py \
  --base /home/raffaello/apprendimento/data/recordings \
  --out  /home/raffaello/dataset/yolo_custom \
  --split 0.85 --prefer cum,best,live \
  --class-map /home/raffaello/apprendimento/class_map.json \
  --fs-view-mhz 1.2

2) Allena YOLO col tuo venv

./02_train_yolo.sh \
  /home/raffaello/dataset/yolo_custom/data.yaml \
  /home/raffaello/yolo_runs \
  yolo_custom \
  yolov8n.pt

3) Classmap per watcher (opzionale ma consigliato)

/home/raffaello/yolo-venv/bin/python 02b_make_classmap.py \
  /home/raffaello/dataset/yolo_custom/data.yaml \
  /home/raffaello/dataset/yolo_custom/classmap.json

4) Smoke test detection su tiles

./03_predict_smoke_test.sh \
  /home/raffaello/yolo_runs/yolo_custom/weights/best.pt \
  /home/raffaello/dataset/yolo_custom/data.yaml \
  "/tmp/tiles/*_cum_*.png"

5) Patch automatico yolo_watcher con nuovi percorsi

python3 03_update_watcher.py \
  /home/raffaello/rfe_bridge/yolo_watcher.py \
  /home/raffaello/yolo_runs/yolo_custom/weights/best.pt \
  /home/raffaello/dataset/yolo_custom/data.yaml

6) Promuovi modello RF (.pkl) a runtime

./04_promote_rf_model.sh \
  /home/raffaello/apprendimento/models/rfscan_staging.pkl \
  /home/raffaello/apprendimento/models/served

export RF_MODEL=/home/raffaello/apprendimento/models/served/rfscan.pkl
python3 /home/raffaello/rf_scan_classifier.py --w-model 0.5 --w-csv 0.3 --w-img 0.2

7) (Opz) Genera tiles “dataset-like”

./90_iq_tiles_profile_dataset.sh

