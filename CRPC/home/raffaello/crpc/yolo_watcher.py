#!/usr/bin/env python3
from pathlib import Path
import os, time, subprocess, json, shlex, sys

TILES_DIR = Path("/tmp/tiles")
PROC_DIR  = Path("/tmp/tiles_proc")
DONE_DIR  = Path("/tmp/tiles_done")
LOG_DIR   = Path("/tmp/crpc_logs")
DET_OUT   = LOG_DIR / "detections.jsonl"

# Fail-safe su /tmp
MIN_FREE_MB = 200  # se scende sotto, facciamo pulizia

def get_free_mb(path="/tmp") -> int:
    """Ritorna spazio libero (MB) sul filesystem che contiene 'path'."""
    st = os.statvfs(path)
    return int((st.f_bavail * st.f_frsize) / (1024 * 1024))

def emergency_trim(folder: Path, keep_last: int = 500):
    """
    Mantiene solo gli ultimi 'keep_last' file più recenti in 'folder',
    cancella i più vecchi. Ignora errori.
    """
    if not folder.exists():
        return
    files = sorted((p for p in folder.glob("*") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep_last:]:
        try:
            p.unlink()
        except OSError:
            pass

# Modello / dataset / esecuzione YOLO (usiamo l'interprete CURRENT del venv)
YOLO_CMD = (
  f"{shlex.quote(sys.executable)} /home/raffaello/scripts/predict_yolo.py "
  "--model /home/raffaello/yolo_runs/rf_yolo3/weights/best.pt "
  "--data /home/raffaello/dataset/yolo_vision/data.yaml "
  "--source {img} "
  "--conf 0.05 --imgsz 640 "
  "--project /tmp/yolo_runs --name crpc_watch --no-save-img"
)

def run_predict(img: Path):
    """Esegue YOLO su 'img' e ritorna lista di detection lette dai .txt."""
    out_dir = Path("/tmp/yolo_runs/crpc_watch")
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    cmd = YOLO_CMD.format(img=str(img))
    try:
        subprocess.check_call(shlex.split(cmd),
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=60)
    except subprocess.CalledProcessError:
        print(f"❌ YOLO err: {img.name}")
        return []
    except subprocess.TimeoutExpired:
        print(f"⏱️ YOLO timeout su {img.name}")
        return []

    lab = labels_dir / f"{img.stem}.txt"
    if not lab.exists():
        return []

    dets = []
    with open(lab, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:  # class xc yc w h conf
                continue
            dets.append({
                "cls": int(p[0]),
                "xc": float(p[1]), "yc": float(p[2]),
                "w": float(p[3]),  "h": float(p[4]),
                "conf": float(p[5]),
            })
    return dets

def append_jsonl(path: Path, items):
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    for d in (TILES_DIR, PROC_DIR, DONE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"👀  Watching {TILES_DIR} → {DET_OUT}")
    try:
        while True:
            # 🔒 fail‑safe: spazio su /tmp
            if get_free_mb("/tmp") < MIN_FREE_MB:
                print("⛔ Spazio basso su /tmp: pulizia di emergenza in tiles_done…")
                emergency_trim(DONE_DIR, keep_last=500)
                time.sleep(2.0)
                continue

            for img in sorted(TILES_DIR.glob("*.png")):
                proc_img = PROC_DIR / img.name
                try:
                    img.rename(proc_img)  # atomic move
                except OSError:
                    # file probabilmente ancora in scrittura → riprova
                    continue

                dets = run_predict(proc_img)
                ts = time.time()
                enriched = []
                for d in dets:
                    d = dict(d)
                    d.setdefault("source", "yolo")
                    d.setdefault("image", proc_img.name)
                    d.setdefault("image_path", str(proc_img))
                    d.setdefault("ts", ts)
                    enriched.append(d)

                try:
                    append_jsonl(DET_OUT, enriched)
                except Exception as e:
                    print(f"⚠️ append_jsonl error: {e}")

                # finito: sposta in DONE (con retry + trim se needed)
                try:
                    proc_img.rename(DONE_DIR / proc_img.name)
                except OSError:
                    if get_free_mb("/tmp") < MIN_FREE_MB:
                        emergency_trim(DONE_DIR, keep_last=400)
                    try:
                        proc_img.rename(DONE_DIR / proc_img.name)
                    except OSError:
                        proc_img.unlink(missing_ok=True)

            time.sleep(0.15)
    except KeyboardInterrupt:
        print("👋  stop")

if __name__ == "__main__":
    main()

