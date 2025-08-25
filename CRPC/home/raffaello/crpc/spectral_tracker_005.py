#!/usr/bin/env python3
import os, time, json
from pathlib import Path
from collections import deque
import builtins, pwd, grp, threading
import sys

# === INPUT/OUTPUT ===
DET_PATH   = Path("/tmp/crpc_logs/detections.jsonl")     # input dal watcher YOLO
OUT_CURR   = Path("/tmp/crpc_logs/tracks_current.json")  # snapshot corrente
OUT_JSONL  = Path("/tmp/crpc_logs/tracks.jsonl")         # storico append
DET_HINT_PATH = Path("/tmp/crpc_logs/detections_hint.jsonl")  # opzionale

# === PATH TILE (per risalire al file fisico)
TILES_DIR       = Path("/tmp/tiles")
TILES_PROC_DIR  = Path("/tmp/tiles_proc")
TILES_DONE_DIR  = Path("/tmp/tiles_done")

LOG_PATH = Path(os.getenv("TRACKER_LOG", "/tmp/crpc_logs/spectral_tracker.log"))
LOG_MAX_KB = float(os.getenv("TRACKER_LOG_MAXKB", "1024"))   # ruota a ~1MB (default)
LOG_OWNER  = os.getenv("TRACKER_LOG_OWNER", "raffaello:raffaello")  # opzionale "user:group"

# === BANDE (MHz) ricavate dal prefisso del filename: "24_" o "58_" ===
BANDS = {
    "24": (2400.0, 2500.0),
    "58": (5725.0, 5875.0),
    "52": (5170.0, 5250.0),
}

# === PARAMETRI TRACKER ===
READ_FROM_START   = True   # True: rigioca anche lo storico già presente
FREQ_GATE_MHZ     = 20.0   # massima distanza (MHz) per assegnare a un track
TIME_GATE_S       = 8.0    # massimo delta t per compatibilità aggiornamento
TRACK_TIMEOUT_S   = 12.0    # se un track non si aggiorna per Xs, lo chiudiamo
HOP_WINDOW        = 5      # punti recenti usati per stime (center/bw/hop)
MIN_CONF = float(os.getenv("TRACKER_MIN_CONF", "0.05"))   # ignora detection troppo deboli (era 0.10)
HISTORY_MAX       = 20     # quanti campioni tenere per le feature RF

# === Debounce/Throttle per gli update del tracker ===
DEBOUNCE_DT_S   = 0.20     # 200 ms
DEBOUNCE_DF_MHz = 0.05
DEBOUNCE_DBW_MHz= 0.10


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

        # NEW: memorizza anche ultimo f/bw per il debounce
        self.last_f_mhz = float(f_center_mhz)
        self.last_bw_mhz = float(bw_mhz)

        # --- metadati utili per rf_scan_classifier ---
        self.history = deque(maxlen=HISTORY_MAX)  # [{t, f}]
        self.history.append({"t": float(ts), "f": float(f_center_mhz)})
        self.yolo_conf = float(yolo_conf) if yolo_conf is not None else None
        self.tile_path = tile_path
        self.img_name  = img_name

    def update(self, f_center_mhz, bw_mhz, ts, yolo_conf=None, tile_path=None, img_name=None):
        self.points.append((ts, f_center_mhz, bw_mhz))
        self.last_ts = ts
        # NEW: aggiorna ultimo f/bw per il debounce
        self.last_f_mhz = float(f_center_mhz)
        self.last_bw_mhz = float(bw_mhz)
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

def map_to_freq(band_key, xc_norm, w_norm, span_mhz=None):
    """Da bbox normalizzata → (center_MHz, bw_MHz).
       Se span_mhz è fornito (tile zoom), usalo; altrimenti usa lo span dell’intera banda."""
    f_lo, f_hi = BANDS[band_key]
    span = float(span_mhz) if (span_mhz and span_mhz > 0) else (f_hi - f_lo)
    f_center = f_lo + float(xc_norm) * (f_hi - f_lo) if not span_mhz else (float(f_lo) + float(xc_norm) * span)  # compatibilità
    # Nota: se passiamo span_mhz, xc_norm va riferito allo zoom. Se non hai xc relativo allo zoom, usa direttamente center_mhz/freq_mhz dal JSONL.
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
        elif name.startswith("52_"):
            band_key = "52"
        else:
            return None
        if band_key not in BANDS:
            return None

        conf = float(d.get("conf", 0.0))
        if conf < MIN_CONF:
            print(f"🟡 YOLO skip (conf={conf:.3f} < {MIN_CONF:.3f}) on {d.get('image')}")
            return None

        # Campi grezzi
        xc = float(d.get("xc", 0.0))
        w  = float(d.get("w", 0.0))
        ts = float(d.get("ts", time.time()))
        cls = d.get("cls")
        img_path_reported = d.get("image_path")
        img_name = os.path.basename(name)

        # Campi in MHz (preferiti)
        center_mhz = d.get("center_mhz")
        freq_mhz   = d.get("freq_mhz")   # spesso è già il centro
        w_mhz      = d.get("w_mhz")
        span_mhz   = d.get("span_mhz")   # span del TILE, es. 1.2 MHz su zoom
        is_zoom    = d.get("is_zoom", False)

        # 1) Se abbiamo già center/w in MHz, usiamoli
        if (w_mhz is not None) and (freq_mhz is not None):
            f_center = float(freq_mhz)
            bw = max(0.0, float(w_mhz))
        elif (w_mhz is not None) and (center_mhz is not None):
            f_center = float(center_mhz)
            bw = max(0.0, float(w_mhz))
        # 2) Altrimenti, se abbiamo span_mhz del tile (zoom), mappa con quello
        elif span_mhz:
            f_center, bw = map_to_freq(band_key, xc, w, span_mhz=float(span_mhz))
            # Se nel JSON c'è il centro in MHz (freq_mhz/center_mhz), preferiscilo per il centro
            if center_mhz is not None:
                f_center = float(center_mhz)
            elif freq_mhz is not None:
                f_center = float(freq_mhz)
        # 3) Fallback “banda intera” (sconsigliato ma necessario come ultima spiaggia)
        else:
            f_center, bw = map_to_freq(band_key, xc, w)  # usa (f_hi - f_lo)

        tile_path = resolve_tile_path(img_name, img_path_reported)

        return {
            "band": band_key,
            "f": f_center,
            "bw": bw,
            "ts": ts,
            "cls": cls,
            "img": img_name,
            "tile_path": tile_path,
            "conf": conf,
            "is_zoom": bool(is_zoom),
        }
    except Exception:
        return None

def should_skip_update(track, f_mhz, bw_mhz, now_ts):
    """
    Ritorna True se l'update è praticamente identico all'ultimo
    ed è arrivato troppo in fretta (debounce).
    """
    last_ts  = getattr(track, "last_ts", None)
    last_f   = getattr(track, "last_f_mhz", None)
    last_bw  = getattr(track, "last_bw_mhz", None)

    if last_ts is None or last_f is None or last_bw is None:
        return False

    if (now_ts - last_ts) < DEBOUNCE_DT_S:
        if abs(f_mhz - last_f) < DEBOUNCE_DF_MHz and abs(bw_mhz - last_bw) < DEBOUNCE_DBW_MHz:
            return True
    return False

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

    # Scegli la sorgente: preferisci HINT se esiste
    src_path = DET_HINT_PATH if DET_HINT_PATH.exists() else DET_PATH
    using_hint = (src_path == DET_HINT_PATH)
    print(f"🔎 Tracking da {src_path} ({'HINT' if using_hint else 'FULL'})")

    for line in tail_jsonl(src_path, READ_FROM_START):
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

            # calcola Δt PRIMA dell'update
            dt_before = det["ts"] - best.last_ts
            if dt_before < 0:
                dt_before = 0.0

            # DEBOUNCE: se aggiornamento è praticamente identico e ravvicinato, salta
            if should_skip_update(best, det["f"], det["bw"], det["ts"]):
                # opzionale: log leggero per debug (non spammare)
                # print(f"↪️  Skip dup T{best.id} [{best.band_key}] f={det['f']:.1f} MHz bw={det['bw']:.1f} @Δt={dt_before:.2f}s")
                continue

            # esegui l'update (passiamo anche la conf YOLO)
            best.update(det["f"], det["bw"], det["ts"],
                        yolo_conf=det.get("conf"),
                        tile_path=det.get("tile_path"),
                        img_name=det.get("img"))

            # log con confidenza
            conf = det.get("conf")
            conf_str = f" conf={float(conf):.3f}" if conf is not None else ""
            print(f"↪️  Update track {best.id} [{best.band_key}] f={det['f']:.1f} MHz bw={det['bw']:.1f}{conf_str} @Δt={dt_before:.2f}s")
        else:
            # ⬇️ CREA UN NUOVO TRACK QUANDO NESSUN CANDIDATO MATCHA
            tr = Track(det["band"], det["f"], det["bw"], det["ts"],
                       cls=det.get("cls"),
                       yolo_conf=det.get("conf"),
                       tile_path=det.get("tile_path"),
                       img_name=det.get("img"))
            active_tracks.append(tr)
            conf = det.get("conf")
            conf_str = f" conf={float(conf):.3f}" if conf is not None else ""
            print(f"➕ New track {tr.id} [{tr.band_key}] f={det['f']:.1f} MHz bw={det['bw']:.1f}{conf_str}")

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

