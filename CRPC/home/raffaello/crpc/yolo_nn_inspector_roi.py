#!/usr/bin/env python3
# yolo_nn_inspector_roi.py  — ROI-aware nearest-neighbor inspector (with --save-roi)
#
# What’s new:
#   --save-roi <DIR>  → saves, for each tile, a PNG montage containing:
#     [col 0] tile ROI  + (optional YOLO pred label/conf)
#     [col 1..K] top-K nearest dataset ROIs with class, split, cosine similarity
#   Additionally saves individual crops in <DIR>/<tile_stem>/ as separate PNGs.
#
# Example:
#   python3 yolo_nn_inspector_roi.py \
#     --dataset /home/raffaello/dataset/yolo_custom \
#     --tiles_glob "/tmp/tiles_done/*.png" \
#     --yolo_weights /home/raffaello/yolo_runs/yolo_custom/weights/best.pt \
#     --yolo_conf 0.01 --yolo_iou 0.2 \
#     --roi label --embed_size 128 --topk 5 \
#     --save-roi /home/raffaello/roi_debug \
#     --csv /home/raffaello/nn_inspector_roi.csv
#
import argparse, sys, glob, csv, os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np

from PIL import Image, ImageDraw, ImageFont

def load_yaml(p: Path):
    import yaml
    with open(p, 'r') as f:
        return yaml.safe_load(f)

def l2_normalize(x: np.ndarray, eps: float=1e-8) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n

def embed_vec(arr: np.ndarray, size: int=128) -> np.ndarray:
    im = Image.fromarray((arr*255).astype(np.uint8), mode='L').resize((size, size), Image.BICUBIC)
    v = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    return l2_normalize(v)

def open_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert('L')
    return np.asarray(im, dtype=np.float32) / 255.0  # [0..1]

def parse_label_for_image(img_path: Path, dataset_root: Path):
    # Return (class_id, split, bbox_norm) where bbox_norm is (cx, cy, w, h) in normalized [0..1]
    parts = img_path.parts
    split = None
    if "images" in parts:
        i = parts.index("images")
        if i+1 < len(parts):
            split = parts[i+1]
    if split is None:
        split = "train" if "train" in parts else ("val" if "val" in parts else "?")
    stem = img_path.stem
    label_path = dataset_root / "labels" / split / f"{stem}.txt"
    if not label_path.exists():
        return (-1, split, None)
    try:
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return (-1, split, None)
        first = lines[0].split()
        cls_id = int(float(first[0]))
        bbox_norm = None
        if len(first) >= 5:
            cx, cy, w, h = map(float, first[1:5])
            bbox_norm = (cx, cy, w, h)
        return (cls_id, split, bbox_norm)
    except Exception:
        return (-1, split, None)

def bbox_from_norm(wh, bbox_norm, margin: float=0.06):
    W, H = wh
    cx, cy, w, h = bbox_norm
    x = (cx - w/2.0) * W
    y = (cy - h/2.0) * H
    bw = w * W
    bh = h * H
    x -= bw*margin; y -= bh*margin
    bw *= (1+2*margin); bh *= (1+2*margin)
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(W, int(np.ceil(x + bw)))
    y1 = min(H, int(np.ceil(y + bh)))
    if x1 <= x0 or y1 <= y0:
        return (0,0,W,H)
    return (x0,y0,x1,y1)

def crop_roi_auto(arr: np.ndarray, pctl: float=97.0, min_frac: float=0.002):
    thr = np.percentile(arr, pctl)
    mask = arr >= thr
    if mask.sum() < arr.size * min_frac:
        thr = np.percentile(arr, 90.0)
        mask = arr >= thr
    ys, xs = np.where(mask)
    H, W = arr.shape
    if len(xs)==0 or len(ys)==0:
        return arr, (0,0,W,H)
    x0, x1 = int(xs.min()), int(xs.max())+1
    y0, y1 = int(ys.min()), int(ys.max())+1
    mx = int(0.04*W); my = int(0.04*H)
    x0 = max(0, x0-mx); x1 = min(W, x1+mx)
    y0 = max(0, y0-my); y1 = min(H, y1+my)
    return arr[y0:y1, x0:x1], (x0,y0,x1,y1)

def scan_dataset_images(dataset_root: Path):
    imgs = []
    for split in ("train", "val"):
        d = dataset_root / "images" / split
        if d.exists():
            imgs.extend(sorted(d.glob("**/*.*")))
    imgs = [p for p in imgs if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".webp")]
    return imgs

def build_class_map(data_yaml: Path):
    data = load_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}

def maybe_load_yolo(weights, data_yaml: Path, conf: float, iou: float):
    if not weights:
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights))
        m.overrides = m.overrides or {}
        m.overrides["data"] = str(data_yaml)
        m.overrides["conf"] = conf
        m.overrides["iou"] = iou
        return m
    except Exception as e:
        print(f"[WARN] YOLO load failed: {e}")
        return None

def yolo_predict_bbox(model, path: Path):
    if model is None:
        return None
    try:
        res = model.predict(source=str(path), conf=model.overrides.get("conf",0.01),
                            iou=model.overrides.get("iou",0.2), agnostic_nms=True, verbose=False)[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
            i = int(np.argmax(areas))
            x0,y0,x1,y1 = xyxy[i].astype(int).tolist()
            return (max(0,x0),max(0,y0),max(0,x1),max(0,y1))
    except Exception:
        pass
    return None

def yolo_predict_label(model, path: Path):
    if model is None:
        return (None, None)
    try:
        res = model.predict(source=str(path), conf=model.overrides.get("conf",0.01),
                            iou=model.overrides.get("iou",0.2), agnostic_nms=True, verbose=False)[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            i = int(np.argmax(confs))
            conf = float(confs[i])
            cid = int(clss[i])
            name = res.names.get(cid, str(cid)) if isinstance(res.names, dict) else str(cid)
            return (name, conf)
    except Exception:
        pass
    return (None, None)

def to_image(arr: np.ndarray, w: int, h: int):
    im = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode='L')
    return im.resize((w,h), Image.BICUBIC)

def caption(img: Image.Image, text: str, pad: int=4):
    # Add a black strip at bottom with white text
    draw = ImageDraw.Draw(img)
    W,H = img.size
    strip_h = 18 + pad*2
    base = Image.new('L', (W, H+strip_h), 0)
    base.paste(img, (0,0))
    # black strip already 0
    # choose font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw = ImageDraw.Draw(base)
    txt = text[:120]
    draw.text((6, H+pad), txt, fill=255, font=font)
    return base

def make_montage(images: List[Image.Image], cols: int=6, pad: int=6, bg: int=0):
    if not images:
        return None
    W = max(im.size[0] for im in images)
    H = max(im.size[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new('L', (cols*(W+pad)+pad, rows*(H+pad)+pad), bg)
    x=y=pad
    c=0
    for im in images:
        out.paste(im, (x,y))
        c+=1
        x += W+pad
        if c % cols == 0:
            x = pad; y += H+pad
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tiles_glob", default="/tmp/tiles_done/*.png")
    ap.add_argument("--data_yaml", default=None)
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--yolo_conf", type=float, default=0.01)
    ap.add_argument("--yolo_iou", type=float, default=0.2)
    ap.add_argument("--roi", choices=["full","label","auto"], default="label",
                    help="ROI mode for dataset images: 'label' uses YOLO labels; tiles use YOLO pred or auto. 'full' disables ROI.")
    ap.add_argument("--embed_size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--save-roi", default=None, help="If set, saves ROI crops and a montage per tile into this directory.")
    ap.add_argument("--thumb", type=int, default=256, help="Width of each ROI thumbnail in montage (height auto-scaled).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    data_yaml = Path(args.data_yaml).expanduser().resolve() if args.data_yaml else (dataset_root / "data.yaml")
    if not dataset_root.exists():
        print(f"ERROR: dataset not found {dataset_root}"); sys.exit(2)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found {data_yaml}"); sys.exit(2)

    class_map = build_class_map(data_yaml)
    ds_imgs = scan_dataset_images(dataset_root)
    if not ds_imgs:
        print("ERROR: no dataset images found under images/train|val"); sys.exit(2)
    print(f"[INFO] Dataset images: {len(ds_imgs)}")

    yolo_model = maybe_load_yolo(Path(args.yolo_weights) if args.yolo_weights else None, data_yaml, args.yolo_conf, args.yolo_iou)
    if yolo_model is not None:
        print(f"[INFO] YOLO loaded: {args.yolo_weights}  (conf={args.yolo_conf}, iou={args.yolo_iou})")

    # Build dataset embeddings with ROI
    ds_vecs = []
    ds_meta = []
    for p in ds_imgs:
        try:
            arr = open_gray(p)
            H, W = arr.shape
            cls_id, split, bbox_norm = parse_label_for_image(p, dataset_root)
            roi = arr
            if args.roi != "full" and bbox_norm:
                x0,y0,x1,y1 = bbox_from_norm((W,H), bbox_norm, margin=0.06)
                roi = arr[y0:y1, x0:x1]
            ds_vecs.append(embed_vec(roi, size=args.embed_size))
            ds_meta.append({"path": str(p),
                            "cls_id": cls_id,
                            "cls_name": class_map.get(cls_id, str(cls_id) if cls_id>=0 else "?"),
                            "split": split,
                            "bbox_norm": bbox_norm})
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    ds_vecs = np.stack(ds_vecs, axis=0)  # [N,D]

    tiles = sorted(glob.glob(args.tiles_glob))
    if not tiles:
        print(f"ERROR: no tiles matched {args.tiles_glob}"); sys.exit(2)
    print(f"[INFO] Tiles to inspect: {len(tiles)}")

    writer = None; fcsv = None
    if args.csv:
        fcsv = open(args.csv, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["tile","pred_label","pred_conf","rank","nn_path","nn_split","nn_cls","cosine_sim"])

    save_root = Path(args.save_roi).expanduser().resolve() if args.save_roi else None
    if save_root:
        ensure_dir(save_root)

    def crop_dataset_roi_img(meta_entry) -> Image.Image:
        p = Path(meta_entry["path"])
        arr = open_gray(p)
        H, W = arr.shape
        im = to_image(arr, args.thumb, int(args.thumb*H/W))
        if args.roi == "full":
            return im
        bbox_norm = meta_entry.get("bbox_norm")
        if bbox_norm:
            x0,y0,x1,y1 = bbox_from_norm((W,H), bbox_norm, margin=0.06)
            roi = arr[y0:y1, x0:x1]
            H2,W2 = roi.shape
            return to_image(roi, args.thumb, int(args.thumb*H2/W2))
        return im

    def short(p: str, n=40):
        p = str(p)
        if len(p) <= n: return p
        return "…" + p[-(n-1):]

    for t in tiles:
        tp = Path(t)
        arr = open_gray(tp)
        Ht, Wt = arr.shape
        roi_arr = arr
        bbox = yolo_predict_bbox(yolo_model, tp) if yolo_model is not None else None
        if bbox:
            x0,y0,x1,y1 = bbox
            roi_arr = arr[y0:y1, x0:x1]
        elif args.roi == "auto":
            roi_arr, _ = crop_roi_auto(arr, pctl=97.0)

        tv = embed_vec(roi_arr, size=args.embed_size)
        sims = ds_vecs @ tv
        idx = np.argsort(-sims)[:args.topk]

        pred_label, pred_conf = yolo_predict_label(yolo_model, tp)

        print("\n" + "="*88)
        print(f"TILE: {tp}")
        if pred_label is not None:
            try:
                print(f"YOLO PRED: {pred_label}  conf={pred_conf:.3f}")
            except Exception:
                print(f"YOLO PRED: {pred_label}  conf={pred_conf}")
        print(f"TOP-{args.topk} NEAREST (ROI embedding, cosine):")

        # Prepare montage images
        montage_imgs: List[Image.Image] = []
        # Column 0: tile ROI with caption
        tile_thumb = to_image(roi_arr, args.thumb, int(args.thumb*roi_arr.shape[0]/roi_arr.shape[1]))
        cap = f"[TILE] {tp.name}"
        if pred_label is not None:
            ctxt = f"{pred_label} {pred_conf:.2f}" if isinstance(pred_conf, float) else f"{pred_label}"
            cap += f" | YOLO: {ctxt}"
        montage_imgs.append(caption(tile_thumb, cap))

        # Also save individual crops if requested
        tile_dir = None
        if save_root:
            tile_dir = save_root / tp.stem
            ensure_dir(tile_dir)
            tile_thumb.convert('L').save(tile_dir / f"tile_roi.png")

        for r, i in enumerate(idx, start=1):
            m = ds_meta[int(i)]
            sim = float(sims[int(i)])
            print(f"  #{r:>2}  sim={sim:6.3f}  [{m['split']}] {m['cls_name']:>18}  -> {m['path']}")
            if writer:
                writer.writerow([str(tp), pred_label, f"{pred_conf:.4f}" if isinstance(pred_conf, float) else (pred_conf or ""),
                                 r, m["path"], m["split"], m["cls_name"], f"{sim:.6f}"])
            # Build ROI thumb
            roi_im = crop_dataset_roi_img(m)
            label = f"[#{r}] {m['cls_name']} • {m['split']} • cos={sim:.3f}"
            label2 = short(m['path'], 60)
            montage_imgs.append(caption(roi_im, label + "  " + label2))
            if tile_dir:
                roi_im.convert('L').save(tile_dir / f"nn_{r:02d}_{m['cls_name']}.png")

        if save_root:
            # Columns = topk + 1 (tile)
            cols = min(args.topk + 1, 6)
            out = make_montage(montage_imgs, cols=cols, pad=6, bg=0)
            out_name = save_root / f"{tp.stem}_montage.png"
            out.convert('L').save(out_name)
            print(f"[SAVE] Montage: {out_name}")
            if tile_dir:
                print(f"[SAVE] Crops dir: {tile_dir}")

    if fcsv:
        fcsv.close()
        print(f"\n[OK] CSV written: {args.csv}")

if __name__ == "__main__":
    main()
