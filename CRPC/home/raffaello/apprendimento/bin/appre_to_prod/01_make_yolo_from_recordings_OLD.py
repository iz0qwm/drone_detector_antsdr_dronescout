
import os, sys, json, argparse, shutil, random
from pathlib import Path
import numpy as np
from PIL import Image

def read_meta(session_dir: Path):
    mp = session_dir / "meta.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}

def find_session_image(tiles_dir: Path, prefer_order):
    for name in prefer_order:
        p = tiles_dir / f"{name}.png"
        if p.exists():
            return p
    return None

def detect_vertical_band_center(img: Image.Image):
    """Return the most prominent vertical band center column (float in [0,W)) and the smoothed profile."""
    g = img.convert("L")
    A = np.asarray(g, dtype=np.float32) / 255.0  # H x W
    H, W = A.shape
    col_sum = A.sum(axis=0)  # W
    # Smooth (moving average window=7)
    pad = 3
    kernel = np.ones(2*pad+1, dtype=np.float32)/(2*pad+1)
    smooth = np.convolve(col_sum, kernel, mode="same")
    j = int(np.argmax(smooth))
    return float(j), smooth

def detect_vertical_band_bbox(img: Image.Image, kstd: float, min_width_frac: float, meta_bw_mhz: float, fs_view_mhz: float):
    """
    Combine image evidence (center) + meta bw to generate YOLO (xc,yc,w,h).
    Width is derived from bw_mhz / fs_view_mhz; center from the strongest column.
    """
    H, W = img.size[1], img.size[0]  # PIL: (W,H)
    j_center, smooth = detect_vertical_band_center(img)

    # Width from metadata:
    if fs_view_mhz is None or fs_view_mhz <= 0:
        fs_view_mhz = meta_bw_mhz if meta_bw_mhz > 0 else 1.2  # fallback
    w_frac = (meta_bw_mhz / fs_view_mhz) if (meta_bw_mhz and fs_view_mhz) else min_width_frac
    # clamp to reasonable range
    w_frac = float(max(min_width_frac, min(0.95, w_frac)))

    # Convert to pixel extent
    w_px = max(1, int(round(w_frac * W)))
    cx = int(round(j_center))
    x0 = int(max(0, cx - w_px//2))
    x1 = int(min(W-1, x0 + w_px - 1))
    # normalize
    xc = ((x0 + x1) / 2.0) / W
    yc = 0.5
    w  = (x1 - x0 + 1) / W
    h  = 1.0
    return float(xc), float(yc), float(w), float(h)

def sanitize_class(name: str) -> str:
    return name.strip().replace(" ", "_")

def collect_sessions(base: Path, prefer_order, class_map):
    """
    Expect: base/<band>/<drone_id>/session_.../tiles/(cum|best|live).png
    Returns list: (image_path, class_name, session_dir)
    """
    items = []
    for band_dir in sorted(base.iterdir()):
        if not band_dir.is_dir():
            continue
        for drone_dir in sorted(band_dir.iterdir()):
            if not drone_dir.is_dir():
                continue
            drone_name = drone_dir.name
            cls = sanitize_class(class_map.get(drone_name, drone_name))
            for sess in sorted(drone_dir.glob("session_*")):
                tiles = sess / "tiles"
                if not tiles.is_dir():
                    continue
                img_path = find_session_image(tiles, prefer_order)
                if img_path:
                    items.append((img_path, cls, sess))
    return items

def train_val_split(items, split=0.85, seed=1337):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_tr = int(round(n * split))
    return items[:n_tr], items[n_tr:]

def save_yolo_example(dst_img, dst_lbl, img_src, bbox, cls_idx):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, dst_img)
    xc, yc, w, h = bbox
    with open(dst_lbl, "w", encoding="utf-8") as f:
        f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Root recordings dir")
    ap.add_argument("--out",  required=True, help="Output YOLO dataset root")
    ap.add_argument("--split", type=float, default=0.85, help="Train split fraction [0..1]")
    ap.add_argument("--prefer", type=str, default="cum,best,live", help="Image preference order (comma sep)")
    ap.add_argument("--class-map", type=str, default="", help="Optional JSON map { 'Air3S':'DJI AIR3S', ... }")
    ap.add_argument("--kstd", type=float, default=2.0, help="(kept for compatibility)")
    ap.add_argument("--min-width-frac", type=float, default=0.02, help="Min band width fraction if meta missing")
    ap.add_argument("--fs-view-mhz", type=float, default=-1.0, help="Override fs_view (MHz). If <=0, use min(meta.bw_mhz,1.2).")
    args = ap.parse_args()

    base = Path(args.base).expanduser()
    out  = Path(args.out).expanduser()
    prefer_order = [s.strip() for s in args.prefer.split(",") if s.strip()]
    cls_map = {}
    if args.class_map:
        with open(args.class_map, "r", encoding="utf-8") as f:
            cls_map = json.load(f)

    items = collect_sessions(base, prefer_order, cls_map)
    if not items:
        print("No sessions found. Check --base and folder structure.", file=sys.stderr)
        sys.exit(2)

    classes = sorted({c for _, c, _ in items})
    class_to_idx = {c:i for i,c in enumerate(classes)}

    train_items, val_items = train_val_split(items, split=args.split)

    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def process(split_items, split_name):
        ok, fail = 0, 0
        for img_path, cls, sess_dir in split_items:
            try:
                im = Image.open(img_path)
                meta = read_meta(sess_dir)
                meta_bw = float(meta.get("bw_mhz", 0.0) or 0.0)
                # Estimate fs_view (what the tile shows horizontally)
                if args.fs_view_mhz and args.fs_view_mhz > 0:
                    fs_view_mhz = float(args.fs_view_mhz)
                else:
                    fs_view_mhz = float(min(meta_bw if meta_bw > 0 else 1.2, 1.2))
                bbox = detect_vertical_band_bbox(
                    im,
                    kstd=args.kstd,
                    min_width_frac=args.min_width_frac,
                    meta_bw_mhz=meta_bw,
                    fs_view_mhz=fs_view_mhz
                )
                stem = f"{sess_dir.parent.name}_{sess_dir.name}"  # e.g., Air3S_session_YYYY...
                dst_img = out / "images" / split_name / f"{stem}.png"
                dst_lbl = out / "labels" / split_name / f"{stem}.txt"
                save_yolo_example(dst_img, dst_lbl, img_path, bbox, class_to_idx[cls])
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[WARN] Failed {img_path}: {e}", file=sys.stderr)
        return ok, fail

    tr_ok, tr_fail = process(train_items, "train")
    va_ok, va_fail = process(val_items, "val")

    data_yaml = out / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names: {classes}\n")

    print("=== SUMMARY ===")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {tr_ok} ok, {tr_fail} failed")
    print(f"Val:   {va_ok} ok, {va_fail} failed")
    print(f"Dataset root: {out}")
    print(f"YAML: {data_yaml}")

if __name__ == "__main__":
    main()
