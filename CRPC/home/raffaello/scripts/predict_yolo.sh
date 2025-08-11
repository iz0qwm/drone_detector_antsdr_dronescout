# Predict YOLO
#
# Solo detection sicure
# predict_yolo.py --conf 0.5 --name rf_yolo3_conf50
#
# Senza immagini, solo .txt
#predict_yolo.py --no-save-img --name rf_yolo3_txt_only
#
source ~/yolo-venv/bin/activate
export PYTHONNOUSERSITE=1
/home/raffaello/yolo-venv/bin/python /home/raffaello/scripts/predict_yolo.py \
  --model /home/raffaello/yolo_runs/rf_yolo3/weights/best.pt \
  --source /home/raffaello/dataset/yolo_vision/test/images \
  --data /home/raffaello/dataset/yolo_vision/data.yaml \
  --project /home/raffaello/yolo_runs \
  --name rf_yolo3_test \
  --conf 0.25 \
  --imgsz 640 \
  --save-img

