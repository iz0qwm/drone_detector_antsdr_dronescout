yolo detect train \
  data=/home/raffaello/dataset/yolo_vision/data.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=50 \
  batch=8 \
  project=/home/raffaello/logs \
  name=yolo_train \
  device=0

