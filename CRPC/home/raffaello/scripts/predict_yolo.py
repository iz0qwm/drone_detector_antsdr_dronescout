#!/usr/bin/env python3
import argparse, os, yaml
from collections import Counter
import pandas as pd
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser(description="YOLO predict + summary")
    ap.add_argument("--model", default="/home/raffaello/yolo_runs/rf_yolo3/weights/best.pt")
    ap.add_argument("--source", default="/home/raffaello/dataset/yolo_vision/test/images")
    ap.add_argument("--data", default="/home/raffaello/dataset/yolo_vision/data.yaml")
    ap.add_argument("--project", default="/home/raffaello/yolo_runs")
    ap.add_argument("--name", default="rf_yolo_infer")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-img", action="store_true", help="Salva immagini annotate")
    ap.add_argument("--no-save-img", dest="save_img", action="store_false")
    ap.set_defaults(save_img=True)
    args = ap.parse_args()

    # Carica class names da data.yaml
    with open(args.data, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    id2name = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}

    # Predizione
    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save_img,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    # Summary dalle predictions in memoria
    counts, conf_sums = Counter(), Counter()
    for r in results:
        if r.boxes is None:
            continue
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs   = r.boxes.conf.cpu().numpy()
        for c, cf in zip(cls_ids, confs):
            counts[c] += 1
            conf_sums[c] += float(cf)

    rows = []
    for cid, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        cname = id2name.get(cid, f"class_{cid}")
        avgc = conf_sums[cid] / n if n else 0.0
        rows.append({"class_id": cid, "class_name": cname, "pred_count": n, "avg_conf": round(avgc, 4)})

    outdir = os.path.join(args.project, args.name)
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "prediction_summary.csv")
    pd.DataFrame(rows, columns=["class_id","class_name","pred_count","avg_conf"]).to_csv(out_csv, index=False)

    print(f"✅ Output: {outdir}")
    print(f"   - labels: {os.path.join(outdir, 'labels')} (se create)")
    print(f"   - summary: {out_csv}")
    if rows:
        for r in rows:
            print(f"   • {r['class_name']}: {r['pred_count']} (avg_conf={r['avg_conf']})")
    else:
        print("⚠️ Nessuna detection (prova con --conf 0.10)")

if __name__ == "__main__":
    main()

