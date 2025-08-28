#!/usr/bin/env python3
from pathlib import Path
import os, time, subprocess, json, shlex, sys, re
from PIL import Image, ImageOps
import builtins, pwd, grp, threading
import sys

TILES_DIR = Path("/tmp/tiles")
PROC_DIR  = Path("/tmp/tiles_proc")
DONE_DIR  = Path("/tmp/tiles_done")
LOG_DIR   = Path("/tmp/crpc_logs")
DET_OUT   = LOG_DIR / "detections.jsonl"
CLASSMAP_PATH = Path("/home/raffaello/apprendimento/class_map.json")

PRE_DIR = Path("/tmp/tiles_pre")
PRE_DIR.mkdir(exist_ok=True, parents=True)

MIN_FREE_MB = 200  # fail‑safe

LOG_PATH = Path(os.getenv("YOLOW_LOG", "/tmp/crpc_logs/yolo_watcher.log"))
LOG_MAX_KB = float(os.getenv("YOLOW_LOG_MAXKB", "1024"))   # ruota a ~1MB (default)
LOG_OWNER  = os.getenv("YOLOW_LOG_OWNER", "raffaello:raffaello")  # opzionale "user:group"

# --- HINT output (soglia più bassa per stimatori) ---
HINT_OUT    = LOG_DIR / "detections_hint.jsonl"
HINT_CONF   = float(os.getenv("YOLOW_HINT_CONF", "0.05"))
# Nomi classe separati da virgola (dopo norm_name)
HINT_CLASSES = os.getenv(
    "YOLOW_HINT_CLASSES",
    "FPV_analogico,FPV_analogicos,VTS_digital"
)



# ---- util ----
# === Logging (tee su file) ===
_log_lock = threading.Lock()
_log_fh = None

def _ensure_log_file():
    global _log_fh
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # apri in append, line-buffering
        _log_fh = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
        # prova a settare owner se richiesto
        if LOG_OWNER and ":" in LOG_OWNER:
            u, g = LOG_OWNER.split(":", 1)
            try:
                uid = pwd.getpwnam(u).pw_uid
                gid = grp.getgrnam(g).gr_gid
                os.chown(LOG_PATH, uid, gid)
            except Exception:
                pass
        # permessi rilassati (lettura a tutti)
        try:
            os.chmod(LOG_PATH, 0o664)
        except Exception:
            pass
    except Exception as e:
        # se non riusciamo ad aprire il log, proseguiamo solo a stdout
        _log_fh = None

def _maybe_rotate():
    try:
        st = LOG_PATH.stat()
        if st.st_size > LOG_MAX_KB * 1024:
            # ruota in-place: .1 sovrascritta
            try:
                os.replace(LOG_PATH, LOG_PATH.with_suffix(LOG_PATH.suffix + ".1"))
            except FileNotFoundError:
                pass
            # riapri file nuovo
            _ensure_log_file()
    except FileNotFoundError:
        _ensure_log_file()
    except Exception:
        pass

def _tee_write(line: str):
    # line già senza newline? aggiungilo
    if not line.endswith("\n"): line += "\n"
    # stdout “vero”
    try:
        sys.__stdout__.write(line)
        sys.__stdout__.flush()
    except Exception:
        pass
    # file
    with _log_lock:
        if _log_fh is None:
            _ensure_log_file()
        if _log_fh:
            try:
                _log_fh.write(line)
                _log_fh.flush()
                _maybe_rotate()
            except Exception:
                pass

# sostituisci print globale con una versione che “teia”

def _print_tee(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    msg = sep.join(str(a) for a in args) + ("" if end == "" else end)
    _tee_write(msg)

builtins.print = _print_tee
_ensure_log_file()

# --------------------------------------

def preproc_autocontrast(in_path: Path) -> Path:
    try:
        im = Image.open(in_path).convert("RGB")
        # taglio 2% code istogramma e autocontrast
        im2 = ImageOps.autocontrast(im, cutoff=2)
        # leggera equalizzazione su luminanza
        im3 = ImageOps.equalize(im2)
        out = PRE_DIR / in_path.name
        im3.save(out)
        return out
    except Exception as e:
        print(f"⚠️ preproc fail {in_path.name}: {e}")
        return in_path

def get_free_mb(path="/tmp") -> int:
    st = os.statvfs(path)
    return int((st.f_bavail * st.f_frsize) / (1024 * 1024))

def emergency_trim(folder: Path, keep_last: int = 500):
    if not folder.exists(): return
    files = sorted((p for p in folder.glob("*") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep_last:]:
        try: p.unlink()
        except OSError: pass

def load_classmap():
    try:
        return json.loads(CLASSMAP_PATH.read_text())
    except Exception:
        return {"by_id": {}, "by_name": {}}

CLASSMAP = load_classmap()
BY_ID = {str(k): v for k, v in CLASSMAP.get("by_id", {}).items()}

# ---- YOLO runner ----
#YOLO_CMD = (
#  f"{shlex.quote(sys.executable)} /home/raffaello/scripts/predict_yolo.py "
#  "--model /home/raffaello/yolo_runs/rf_yolo3/weights/best.pt "
#  "--data /home/raffaello/dataset/yolo_custom/data.yaml "
#  "--source {img} "
#  "--conf 0.05 --imgsz 640 "
#  "--project /tmp/yolo_runs --name crpc_watch --no-save-img"
#)

YOLO_CMD = (
  f"{shlex.quote(sys.executable)} /home/raffaello/scripts/predict_yolo.py "
  "--model /home/raffaello/yolo_runs/yolo_custom/weights/best.pt "
  "--data /home/raffaello/dataset/yolo_custom/data.yaml "
  "--source {img} "
  "--conf 0.05 --imgsz 640 "
  "--project /tmp/yolo_runs --name crpc_watch --no-save-img"
)

def norm_name(s: str) -> str:
    # normalizzazione leggera del nome classe (facoltativa)
    s = str(s or "").strip()
    s = s.replace("_", " ")
    return s

def run_predict(img: Path):
    out_dir = Path("/tmp/yolo_runs/crpc_watch")
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    def _call(conf, imgsz):
        cmd = (
          f"{shlex.quote(sys.executable)} /home/raffaello/scripts/predict_yolo.py "
          "--model /home/raffaello/yolo_runs/yolo_custom/weights/best.pt "
          "--data /home/raffaello/dataset/yolo_custom/data.yaml "
          f"--source {img} "
          f"--conf {conf} --imgsz {imgsz} "
          "--project /tmp/yolo_runs --name crpc_watch --no-save-img"
        )
        #cmd = (
        #  f"{shlex.quote(sys.executable)} /home/raffaello/scripts/predict_yolo.py "
        #  "--model /home/raffaello/yolo_runs/rf_yolo3/weights/best.pt "
        #  "--data /home/raffaello/dataset/yolo_vision/data.yaml "
        #  f"--source {img} "
        #  f"--conf {conf} --imgsz {imgsz} "
        #  "--project /tmp/yolo_runs --name crpc_watch --no-save-img"
        #)
        try:
            subprocess.check_call(shlex.split(cmd),
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  timeout=60)
            lab = labels_dir / f"{img.stem}.txt"
            return lab if lab.exists() else None
        except subprocess.TimeoutExpired:
            print(f"⏱️ YOLO timeout su {img.name}")
            return None
        except subprocess.CalledProcessError:
            print(f"❌ YOLO err: {img.name}")
            return None

    # try 1: set di base
    lab = _call(conf=0.05, imgsz=640)
    # try 2: più permissivo + più grande
    if not lab:
        print("🔁 Retry YOLO: conf=0.02, imgsz=896 (+preproc)")
        pre_img = preproc_autocontrast(img)
        lab = _call(conf=0.02, imgsz=896 if pre_img else 640)

    # try 3: ancora più permissivo
    if not lab:
        print("🔁 Retry YOLO: conf=0.01, imgsz=1024")
        lab = _call(conf=0.01, imgsz=1024)

    if not lab:
        return []

    dets = []
    with open(lab, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            cls_id = str(int(p[0]))
            name = BY_ID.get(cls_id)
            if name: name = norm_name(name) 
            dets.append({
                "cls": int(p[0]),
                "cls_name": BY_ID.get(cls_id),
                "xc": float(p[1]), "yc": float(p[2]),
                "w": float(p[3]),  "h": float(p[4]),
                "conf": float(p[5]),
            })
    return dets


def append_jsonl(path: Path, items):
    if not items: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ---- parsing nome file → banda/tile ----
# Esempi validi: 58_cum_2025-08-14_23-35-36.png, 24_live_2025-08-15_10-12-00.png
TS_RE = re.compile(r"_(20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
def infer_meta_from_name(name: str):
    base = os.path.basename(name)
    parts = base.split("_")
    band = parts[0] if parts and parts[0] in ("24","52","58") else None
    tile_kind = "cum" if "_cum_" in base else ("live" if "_live_" in base else None)
    has_ts = bool(TS_RE.search(base))
    return band, tile_kind, has_ts

# Evita rielaborazioni
SEEN = set()

def main():
    for d in (TILES_DIR, PROC_DIR, DONE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"👀  Watching {TILES_DIR} → {DET_OUT}")
    try:
        while True:
            if get_free_mb("/tmp") < MIN_FREE_MB:
                print("⛔ Spazio basso su /tmp: pulizia di emergenza in tiles_done…")
                emergency_trim(DONE_DIR, keep_last=500)
                time.sleep(2.0)
                continue

            for img in sorted(TILES_DIR.glob("*.png")):
                # 1) Skippa file “non definitivi” (senza timestamp) e file cumulativi generici
                band, tile_kind, has_ts = infer_meta_from_name(img.name)
                # Accetta anche i tile senza timestamp (legacy producer).
                # Skippa solo se non riusciamo a capire la banda dal prefisso.
                if band is None:
                    continue

                # ⬅ QUI ci metti il check “file stabile”
                try:
                    if time.time() - img.stat().st_mtime < 0.2:
                        continue  # aspetta che sia “stabile” da ≥200 ms
                except Exception:
                    pass

                print(f"📥 Trovato tile: {img.name}  band={band} kind={tile_kind}")    
                # 2) dedup per sicurezza
                #if img.name in SEEN:
                #    continue

                proc_img = PROC_DIR / img.name
                try:
                    img.rename(proc_img)  # atomic move
                except OSError:
                    print(f"⚠️ Non riesco a spostare {img.name}, forse ancora in scrittura")
                    continue  # probabilmente ancora in scrittura

                print(f"🔍 YOLO su {proc_img.name}...")
                dets = run_predict(proc_img)
                print(f"   → {len(dets)} detection trovate")

                ts = time.time()

                # 3) arricchisci con metadati + w_mhz/freq_mhz se possibile
                enriched = []
                # set di classi HINT (normalizzate come in norm_name)
                _hint_classes = set([norm_name(s) for s in HINT_CLASSES.split(",") if s.strip()])
                _hint_rows = []
                # determina span (MHz) prioritizzando lo span reale scritto dal renderer
                span_mhz = None
                try:
                    if band:
                        span_path = Path(f"/tmp/span_{band}.txt")
                        if span_path.exists():
                            try:
                                span_hz = float(span_path.read_text().strip())
                                span_mhz = span_hz / 1e6
                            except Exception:
                                span_mhz = None
                    if span_mhz is None:
                        # fallback solo per tiles a banda piena (RFE/mosaico)
                        span_mhz = {"24": 2500-2400, "58": 5875-5725, "52": 5250-5170}.get(band or "", None)
                except Exception:
                    span_mhz = None

                # center (MHz) per convertire xc → freq_mhz
                center_mhz = None
                try:
                    if band:
                        cpath = Path(f"/tmp/center_{band}.txt")
                        if cpath.exists():
                            center_mhz = float(cpath.read_text().strip()) / 1e6
                except Exception:
                    center_mhz = None

                for d in dets:
                    d = dict(d)
                    d.setdefault("source", "yolo")
                    d.setdefault("image", proc_img.name)
                    d.setdefault("image_path", str(proc_img))
                    d.setdefault("ts", ts)
                    d["band"] = band
                    d["tile_kind"] = tile_kind
                    d["span_mhz"] = span_mhz
                    d["is_zoom"]  = (span_mhz is not None and span_mhz <= 5.0)
                    d["center_mhz"] = center_mhz  # se disponibile
                    
                    # 🔤 nome classe normalizzato (comodo per i filtri HINT)
                    try:
                        d["name"] = norm_name(d.get("cls_name") or "")
                    except Exception:
                        d["name"] = ""

                    # ⛳️ Plausibilità per banda: “FPV analogico” non esiste su 2.4/5.2
                    name_l = d["name"].lower()
                    if band in ("24","52") and ("fpv analogico" in name_l or "FPV_analogico" in name_l or "analog" == name_l):
                        d["name"] = "VTS_digital"  # etichetta neutra per gli hint

                    # calcolo w_mhz e freq_mhz se abbiamo abbastanza info
                    try:
                        if span_mhz is not None:
                            w = float(d.get("w", 0.0))
                            d["w_mhz"] = w * float(span_mhz)
                        if (center_mhz is not None) and (span_mhz is not None):
                            xc = float(d.get("xc", 0.5))
                            d["freq_mhz"] = center_mhz + (xc - 0.5) * float(span_mhz)
                    except Exception:
                        pass

                    enriched.append(d)

                    # 📌 Flusso HINT: classi target + soglia più bassa
                    try:
                        if d["name"] in _hint_classes and float(d.get("conf", 0.0)) >= HINT_CONF:
                            _hint_rows.append(d)
                    except Exception:
                        pass

                # scrivi su JSONL
                try:
                    append_jsonl(DET_OUT, enriched)
                    if enriched:
                        print(f"💾 Salvate {len(enriched)} detection in {DET_OUT}")
                    # ✍️ salva anche il flusso HINT (se presente)
                    if _hint_rows:
                        append_jsonl(HINT_OUT, _hint_rows)
                        print(f"💾 (HINT) Salvate {len(_hint_rows)} detection in {HINT_OUT}")
    
                except Exception as e:
                    print(f"⚠️ append_jsonl error: {e}")


                # 4) completa: sposta in DONE
                try:
                    proc_img.rename(DONE_DIR / proc_img.name)
                except OSError:
                    if get_free_mb("/tmp") < MIN_FREE_MB:
                        emergency_trim(DONE_DIR, keep_last=400)
                    try:
                        proc_img.rename(DONE_DIR / proc_img.name)
                    except OSError:
                        proc_img.unlink(missing_ok=True)

                SEEN.add(img.name)

            time.sleep(0.15)
    except KeyboardInterrupt:
        print("👋  stop")

if __name__ == "__main__":
    main()
