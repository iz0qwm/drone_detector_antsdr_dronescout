#!/usr/bin/env python3
import os, time, json
from pathlib import Path
from collections import deque

# === INPUT/OUTPUT ===
DET_PATH   = Path("/tmp/crpc_logs/detections.jsonl")     # input dal watcher YOLO
OUT_CURR   = Path("/tmp/crpc_logs/tracks_current.json")  # snapshot corrente
OUT_JSONL  = Path("/tmp/crpc_logs/tracks.jsonl")         # storico append

# === PATH TILE (per risalire al file fisico)
TILES_DIR       = Path("/tmp/tiles")
TILES_PROC_DIR  = Path("/tmp/tiles_proc")
TILES_DONE_DIR  = Path("/tmp/tiles_done")

# === BANDE (MHz) ricavate dal prefisso del filename: "24_" o "58_" ===
BANDS = {
    "24": (2400.0, 2500.0),
    "58": (5725.0, 5875.0),
}

# === PARAMETRI TRACKER ===
READ_FROM_START   = True   # True: rigioca anche lo storico già presente
FREQ_GATE_MHZ     = 12.0   # massima distanza (MHz) per assegnare a un track
TIME_GATE_S       = 3.0    # massimo delta t per compatibilità aggiornamento
TRACK_TIMEOUT_S   = 5.0    # se un track non si aggiorna per Xs, lo chiudiamo
HOP_WINDOW        = 5      # punti recenti usati per stime (center/bw/hop)
MIN_CONF          = 0.06   # ignora detection troppo deboli (era 0.10)
HISTORY_MAX       = 20     # quanti campioni tenere per le feature RF

class Track:
    _next_id = 1

    def __init__(self, band_key, f_center_mhz, bw_mhz, ts, cls=None,
                 yolo_conf=None, tile_path=None, img_name=None):
        self.id = Track._next_id; Track._next_id += 1
        self.band_key = band_key
        self.points = deque(maxlen=256)  # (ts, f_center_mhz, bw_mhz)
        self.points.append((ts, f_center_mhz, bw_mhz))
        self.cls = cls
        self.last_ts = ts

        # --- metadati utili per rf_scan_classifier ---
        self.history = deque(maxlen=HISTORY_MAX)  # [{t, f}]
        self.history.append({"t": float(ts), "f": float(f_center_mhz)})
        self.yolo_conf = float(yolo_conf) if yolo_conf is not None else None
        self.tile_path = tile_path
        self.img_name  = img_name

    def update(self, f_center_mhz, bw_mhz, ts, yolo_conf=None, tile_path=None, img_name=None):
        self.points.append((ts, f_center_mhz, bw_mhz))
        self.last_ts = ts
        # aggiorna metadati
        self.history.append({"t": float(ts), "f": float(f_center_mhz)})
        if yolo_conf is not None:
            self.yolo_conf = float(yolo_conf)
        if tile_path is not None:
            self.tile_path = tile_path
        if img_name is not None:
            self.img_name = img_name

    def _last_pts(self):
        return list(self.points)[-HOP_WINDOW:]  # deque -> list per slicing

    def center_freq(self):
        pts = self._last_pts()
        if not pts: return None
        return sum(p[1] for p in pts) / len(pts)

    def bandwidth(self):
        pts = self._last_pts()
        if not pts: return None
        return sum(p[2] for p in pts) / len(pts)

    def hop_rate(self):
        """MHz/s sugli ultimi HOP_WINDOW punti (regressione lineare)."""
        pts = self._last_pts()
        if len(pts) < 2: return 0.0
        t0 = pts[0][0]
        xs = [p[0] - t0 for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs)
        sxy = sum(x*y for x, y in zip(xs, ys))
        den = n*sxx - sx*sx
        if abs(den) < 1e-6: return 0.0
        return (n*sxy - sx*sy) / den

    def to_dict(self):
        # history serializzabile
        hist_list = list(self.history)
        return {
            "track_id": self.id,
            "band": self.band_key,
            "center_freq_mhz": round(self.center_freq() or 0.0, 3),
            "bandwidth_mhz":   round(self.bandwidth() or 0.0,   3),
            "hop_rate_mhz_s":  round(self.hop_rate(),            3),
            "last_seen": self.last_ts,
            "len": len(self.points),
            "cls": self.cls,
            # --- nuovi campi per rf_scan_classifier ---
            "history": hist_list,                # [{t, f}, ...]
            "yolo_conf": self.yolo_conf,         # 0..1 (se disponibile)
            "tile_path": self.tile_path,         # path assoluto se trovato
            "img": self.img_name,                # nome immagine originale
        }

def map_to_freq(band_key, xc_norm, w_norm):
    """Da bbox normalizzata → (center_MHz, bw_MHz) nella banda."""
    f_lo, f_hi = BANDS[band_key]
    span = f_hi - f_lo
    f_center = f_lo + float(xc_norm) * span
    bw = max(0.0, float(w_norm) * span)
    return f_center, bw

def resolve_tile_path(name, image_path=None):
    """
    Prova a risalire al percorso della tile:
    1) /tmp/tiles_done/<name>
    2) image_path (se esiste e punta alla png corrente)
    3) /tmp/tiles_proc/<name>
    4) /tmp/tiles/<name>
    """
    if name:
        p_done = TILES_DONE_DIR / name
        if p_done.exists():
            return str(p_done)
    if image_path and os.path.exists(image_path):
        return image_path
    if name:
        p_proc = TILES_PROC_DIR / name
        if p_proc.exists():
            return str(p_proc)
        p_live = TILES_DIR / name
        if p_live.exists():
            return str(p_live)
    return None

def parse_det(line):
    """Parsa una riga JSON prodotta dal watcher YOLO."""
    try:
        d = json.loads(line)
        name = d.get("image","")
        if name.startswith("24_"):
            band_key = "24"
        elif name.startswith("58_"):
            band_key = "58"
        else:
            return None
        if band_key not in BANDS:
            return None

        conf = float(d.get("conf", 0.0))
        if conf < MIN_CONF:
            return None

        xc = float(d["xc"]); w = float(d["w"])
        ts = float(d.get("ts", time.time()))
        cls = d.get("cls")
        img_path_reported = d.get("image_path")  # es. /tmp/tiles_proc/24_000140.png

        f_center, bw = map_to_freq(band_key, xc, w)
        tile_path = resolve_tile_path(os.path.basename(name), img_path_reported)

        return {
            "band": band_key,
            "f": f_center,
            "bw": bw,
            "ts": ts,
            "cls": cls,
            "img": os.path.basename(name),
            "tile_path": tile_path,
            "conf": conf
        }
    except Exception:
        return None

def tail_jsonl(path: Path, from_start=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("r", buffering=1, errors="ignore")
    if not from_start:
        f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.05)
            continue
        yield line

def safe_write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        tmp.replace(path)
    except Exception:
        pass  # non bloccare il loop in caso di I/O error

def main():
    active_tracks = []  # lista Track
    last_flush = 0.0

    print(f"🔎 Tracking da {DET_PATH}")
    for line in tail_jsonl(DET_PATH, READ_FROM_START):
        det = parse_det(line)
        if not det:
            continue

        # 1) associa a un track esistente (stessa banda + coerenza freq/tempo)
        candidates = []
        for tr in active_tracks:
            if tr.band_key != det["band"]:
                continue
            dt = det["ts"] - tr.last_ts
            if dt < 0: dt = 0
            if dt > TIME_GATE_S:
                continue
            df = abs((tr.center_freq() or det["f"]) - det["f"])
            if df <= FREQ_GATE_MHZ:
                candidates.append((df + 0.01*dt, tr))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best = candidates[0]
            best.update(det["f"], det["bw"], det["ts"],
                        yolo_conf=det.get("conf"),
                        tile_path=det.get("tile_path"),
                        img_name=det.get("img"))
            print(f"↪️  Update track {best.id} [{best.band_key}] f={det['f']:.1f} MHz bw={det['bw']:.1f} @Δt={det['ts']-best.last_ts:.2f}s")
        else:
            tr = Track(det["band"], det["f"], det["bw"], det["ts"],
                       cls=det.get("cls"),
                       yolo_conf=det.get("conf"),
                       tile_path=det.get("tile_path"),
                       img_name=det.get("img"))
            active_tracks.append(tr)
            print(f"➕ New track {tr.id} [{tr.band_key}] f={det['f']:.1f} MHz bw={det['bw']:.1f}")

        # 2) timeout/cleanup
        now = det["ts"]
        before = len(active_tracks)
        active_tracks = [t for t in active_tracks if (now - t.last_ts) <= TRACK_TIMEOUT_S]
        closed = before - len(active_tracks)
        if closed > 0:
            print(f"🧹 Closed {closed} track(s) for timeout")

        # 3) flush periodico
        if now - last_flush >= 0.5:
            snapshot = [t.to_dict() for t in active_tracks]
            safe_write_json(OUT_CURR, snapshot)
            with OUT_JSONL.open("a") as g:
                for t in snapshot:
                    g.write(json.dumps(t, ensure_ascii=False) + "\n")
            last_flush = now

if __name__ == "__main__":
    try:
        OUT_CURR.parent.mkdir(parents=True, exist_ok=True)
        main()
    except KeyboardInterrupt:
        print("👋 stop")

