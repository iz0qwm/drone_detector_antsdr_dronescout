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
