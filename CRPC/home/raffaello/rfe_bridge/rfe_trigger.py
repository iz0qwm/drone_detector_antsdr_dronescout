#!/usr/bin/env python3
import time, os, csv, json, math, statistics as stats
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
import builtins, pwd, grp, threading
import sys

# === Config ===
RFE_DIR = Path("/tmp/rfe/scan")
CRCP_LOG_DIR = Path("/tmp/crpc_logs")
TRIG_JSON = CRCP_LOG_DIR / "last_trigger.json"
TRIGGER_FIFO = Path("/tmp/hackrf_trigger.fifo")
STATE_FILE = Path("/tmp/rfe_trigger_state.json")
LOG_PATH = Path(os.getenv("RFE_LOG", "/tmp/crpc_logs/rfe_trigger.log"))
LOG_MAX_KB = float(os.getenv("RFE_LOG_MAXKB", "1024"))   # ruota a ~1MB (default)
LOG_OWNER  = os.getenv("RFE_LOG_OWNER", "raffaello:raffaello")  # opzionale "user:group"

BANDS = {
    "24": {"path": "/tmp/rfe/scan/latest_24.csv", "min_f": 2400.0, "max_f": 2485.0},
    "58": {"path": "/tmp/rfe/scan/latest_58.csv", "min_f": 5725.0, "max_f": 5875.0},
    "52": {"path": "/tmp/rfe/scan/latest_52.csv", "min_f": 5170.0, "max_f": 5250.0},
}

# Env override (float/int)
def envf(k, d): 
    try: return float(os.getenv(k, d))
    except: return d
def envi(k, d): 
    try: return int(os.getenv(k, d))
    except: return d
def envb(k, d=False):
    v = (os.getenv(k) or "").lower()
    return d if v=="" else v in ("1","true","yes","on")

# Soglie SNR per banda (dB sopra floor)
PEAK_DB_ABOVE_FLOOR = envf("RFE_PEAK_DB", 4.0)     # dB sopra rumore 
PEAK_DB_ABOVE_FLOOR_24 = envf("RFE_PEAK_DB_24", 5)   # più severo a 2.4 (meno Wi-Fi)
PEAK_DB_ABOVE_FLOOR_58 = envf("RFE_PEAK_DB_58", 3.5)   # un filo più permissivo a 5.8
MIN_CONSEC_SWEEPS   = envi("RFE_MIN_SWEEPS", 2)    # persistenza

# Consenti di fondere picchi separati ~3 MHz (tipico O3 10 MHz con 3-4 spike)
#MERGE_BIN_HZ   = envf("RFE_MERGE_HZ", 2.0e6)   # > 0.766 MHz per fondere i bin adiacenti = 2MHz
MERGE_BIN_HZ   = envf("RFE_MERGE_HZ", 3.0e6)


# Considera valido un blob “video” già da 6 MHz (un 10 MHz “imperfetto” lo supera comunque)
#MIN_BLOB_BW_HZ = envf("RFE_MIN_BW_HZ", 8.0e6)  # solo segnali larghi (video) 8MHz
MIN_BLOB_BW_HZ = envf("RFE_MIN_BW_HZ", 6.0e6)

# Limita le larghezze candidate (anti-falsi "Wi-Fi aggregate")
MAX_BLOB_BW_HZ = 32.0e6     # hard cap globale
# Cap larghezza per banda (anti Wi-Fi aggregate)
MAX_BLOB_BW_24_HZ = envf("RFE_MAX_BW_24_HZ", 22.0e6)  # ↓ da 22 MHz
#MAX_BLOB_BW_24_HZ = envf("RFE_MAX_BW_24_HZ", 2.0e7)  # ↓ da 20 MHz
MAX_BLOB_BW_58_HZ = envf("RFE_MAX_BW_58_HZ", 30.0e6)  # invariato

MAX_TRIGGERS_PER_MIN= envi("RFE_MAX_TPM", 4)
COOLDOWN_S          = envf("RFE_COOLDOWN_S", 15)
HIST_SWEEPS         = envi("RFE_HIST_SWEEPS", 20)
DEBUG               = envb("RFE_DEBUG", True)

SAME_BLOB_DEBOUNCE_S = float(os.getenv("RFE_SAME_BLOB_DEBOUNCE_S", 12))
last_blob = {b: None for b in BANDS}
# Persistenza "fuzzy"
#OVERLAP_FRAC    = envf("RFE_OVERLAP_FRAC", 0.35)  # soglia di sovrapposizione (IoU) 0..1
#CENTER_TOL_MHZ  = envf("RFE_CENTER_TOL_MHZ", 10.0)  # tolleranza alternativa sul centro (MHz)
#PERSIST_WINDOW = envi("RFE_PERSIST_WINDOW", 5)  # nuove: ultime N sweep

# Persistenza fuzzy un filo più permissiva
OVERLAP_FRAC    = envf("RFE_OVERLAP_FRAC", 0.30)
CENTER_TOL_MHZ  = envf("RFE_CENTER_TOL_MHZ", 10.0)
PERSIST_WINDOW  = envi("RFE_PERSIST_WINDOW", 6)

# Tolleranze di centro/IoU per banda (fallback di persistenza)
CENTER_TOL_24 = envf("RFE_CENTER_TOL_24", 8.0)
CENTER_TOL_58 = envf("RFE_CENTER_TOL_58", 12.0)
OVERLAP_FRAC_24 = envf("RFE_IOU_24", 0.40)            # ↑ da 0.30 → 0.35
OVERLAP_FRAC_58 = envf("RFE_IOU_58", 0.30)

# Ponte tra isole di picchi (single-link):
BRIDGE_MAX_HZ_DEFAULT   = 5.5e6   # default per 2.4
BRIDGE_MAX_HZ_24        = 5.5e6   # O3 a 10/20 MHz spesso ha spike 3–5 MHz
BRIDGE_MAX_HZ_58        = 4.0e6   # a 5.8 facciamo più severo (meno falsi)

# Picchi minimi nel blob:
MIN_PEAKS_PER_BLOB_DEF  = 3
# Picchi minimi per blob (per banda)
MIN_PEAKS_PER_BLOB_24 = envi("RFE_MIN_PEAKS_24", 5)   # ↑ da 3 → 4 a 2.4
MIN_PEAKS_PER_BLOB_58 = envi("RFE_MIN_PEAKS_58", 3)   # ↓ da 4 → 3 a 5.8

# Finestra "golden" 5.8: forziamo bootstrap
GOLDEN_58_LO = envf("RFE_58_WIN_LO", 5771.0)
GOLDEN_58_HI = envf("RFE_58_WIN_HI", 5791.0)

# Stato in RAM
history = {b: deque(maxlen=HIST_SWEEPS) for b in BANDS}
last_trigger_ts = {b: 0 for b in BANDS}
last_seen_file  = {b: None for b in BANDS}  # (inode, size, mtime)
trigger_count = deque(maxlen=120)  # timestamps

# --- Helper ---

CSV_FREQ_KEYS = ("freq_mhz","frequency","freq","mhz","freqmhz","center_mhz","center")
CSV_DBM_KEYS  = ("power_dbm","dbm","amp_dbm","amp","amplitude","power","db")

# --- AGGIUNGI IN CIMA vicino alle altre costanti ---
FOCUS_FILE = Path("/tmp/rfe/focus.json")
FOCUS_HOLD_S = float(os.getenv("RFE_FOCUS_HOLD_S", "8"))
FOCUS_HOLD_GOLDEN_S = float(os.getenv("RFE_FOCUS_HOLD_GOLDEN_S", "12"))

SUPPRESS_FILE = Path("/tmp/rfe/suppress_24.json")
SUPPRESS_HOLD_S = 30.0
SUPPRESS_CF_TOL = 12.0  # ±MHz attorno al centro “spento”
# --- Wi-Fi 2.4 helpers + suppress window ---
WIFI2G_CH_CENTERS = [2412 + 5*i for i in range(0, 13)]  # CH1..13

def is_wifi20_24(center_mhz: float, bw_mhz: float, tol_bw=3.0, tol_cf=2.0) -> bool:
    if bw_mhz < 16.0 or bw_mhz > 24.0:
        return False
    near_ch = any(abs(center_mhz - ch) <= tol_cf for ch in WIFI2G_CH_CENTERS)
    return near_ch

def add_suppress_24(center_mhz: float, now=None):
    now = now or time.time()
    data = {"until_ts": now + SUPPRESS_HOLD_S, "center": center_mhz}
    SUPPRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUPPRESS_FILE.write_text(json.dumps(data))

def is_suppressed_24(center_mhz: float, now=None) -> bool:
    now = now or time.time()
    if not SUPPRESS_FILE.exists(): return False
    try:
        d = json.loads(SUPPRESS_FILE.read_text())
        if now > float(d.get("until_ts", 0)): return False
        return abs(center_mhz - float(d.get("center", 0.0))) <= SUPPRESS_CF_TOL
    except Exception:
        return False



def _write_focus(band, hold_s):
    try:
        FOCUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"band": str(band), "until_ts": time.time() + float(hold_s)}
        FOCUS_FILE.write_text(json.dumps(payload))
        try: os.chmod(FOCUS_FILE, 0o666)
        except Exception: pass
        if DEBUG: print(f"🎯 focus set → {payload}")
    except Exception as e:
        if DEBUG: print(f"focus write error: {e}")

def blob_interval_mhz(blob):
    # usa i capi del blob (più robusto del centro stimato)
    fmin = min(f for f, _ in blob)
    fmax = max(f for f, _ in blob)
    return (fmin, fmax)

def interval_overlap(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    return max(0.0, hi - lo)

def interval_iou(a, b):
    inter = interval_overlap(a, b)
    if inter <= 0: return 0.0
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter / union if union > 0 else 0.0

def blob_center_width_mhz(blob):
    fmin, fmax = blob_interval_mhz(blob)
    return (0.5*(fmin+fmax), (fmax - fmin))

def bin_spacing_hz(bins):
    if len(bins) < 2: return 0
    diffs = [(bins[i+1][0] - bins[i][0]) for i in range(len(bins)-1)]
    diffs = [d for d in diffs if d > 0]
    diffs.sort()
    med = diffs[len(diffs)//2] if diffs else 0
    return med * 1e6


def _first_present(d, keys):
    for k in keys:
        if k in d: 
            return k
    return None

def read_latest_csv(band):
    path = BANDS[band]["path"]
    if not os.path.exists(path):
        if DEBUG: print(f"[{band}] file assente: {path}")
        return None
    try:
        st = os.stat(path)  # segue il symlink
        sig = (st.st_ino, st.st_size, int(st.st_mtime))
    except Exception as e:
        if DEBUG: print(f"[{band}] stat errore: {e}")
        return None

    if sig == last_seen_file[band]:
        # nessun cambiamento rilevante
        return None

    rows = []
    try:
        with open(path, newline="") as f:
            # Sniff header robusto
            reader = csv.DictReader(f)
            header = [h.strip().lower() for h in (reader.fieldnames or [])]
            # normalizza riga per riga
            for r in reader:
                rl = { (k or "").strip().lower(): (v or "").strip() for k,v in r.items() }
                fk = _first_present(rl, CSV_FREQ_KEYS)
                ak = _first_present(rl, CSV_DBM_KEYS)
                if not fk or not ak:
                    continue
                try:
                    fmhz = float(rl[fk])
                    dbm  = float(rl[ak])
                except:
                    continue
                rows.append((fmhz, dbm))
    except Exception as e:
        if DEBUG: print(f"[{band}] read errore: {e}")
        return None

    if rows:
        rows.sort(key=lambda x: x[0])
        last_seen_file[band] = sig
        if DEBUG:
            print(f"[{band}] sweep nuovo: bins={len(rows)} file={path} sig={sig}")
        return rows

    if DEBUG: print(f"[{band}] nessuna riga valida nel CSV.")
    last_seen_file[band] = sig  # evita di ri-leggere a vuoto
    return None


def estimate_floor_dB(bins, band=None):
    """
    Floor per-bin. Se la storia è sufficiente, usa mediana+0.5*MAD sui bin storici
    **della stessa banda**. Altrimenti usa baseline locale sul singolo sweep.
    """
    n = len(bins)
    # 1) Se abbiamo abbastanza storia per **questa banda**, usa lo schema storico
    have_hist = band in history and len(history[band]) >= 5
    if have_hist:
        per_bin = defaultdict(list)
        for sweep in history[band]:
            if len(sweep) != n:
                continue  # salta sweep non allineati
            for i, (_, db) in enumerate(sweep):
                per_bin[i].append(db)
        floor = []
        for i, (_, cur_db) in enumerate(bins):
            v = per_bin.get(i)
            if not v:
                floor.append(cur_db)  # fallback locale
            else:
                m = stats.median(v)
                mad = stats.median([abs(x - m) for x in v]) if len(v) > 3 else 0.0
                floor.append(m + 0.5*mad)
        return floor

    # 2) Nessuna/povera storia: baseline locale sul singolo sweep
    W = 9
    half = W // 2
    series = [db for _, db in bins]
    mm = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        mm.append(stats.median(series[a:b]))
    qs = sorted(series)
    q25 = qs[max(0, int(0.25 * (n - 1)))] if n else -100.0
    floor = [min(m, q25 + 0.5) for m in mm]
    return floor



def blobs_from_peaks(bins, floor, band=None):
    # individua bin sopra soglia
    over = []
    # soglia per banda (5.x più severa)
    if band == "24":
        thr = max(PEAK_DB_ABOVE_FLOOR, PEAK_DB_ABOVE_FLOOR_24)
    elif band == "58":
        thr = max(PEAK_DB_ABOVE_FLOOR, PEAK_DB_ABOVE_FLOOR_58)
    elif band == "52":
        thr = max(PEAK_DB_ABOVE_FLOOR, 4.0)
    else:
        thr = PEAK_DB_ABOVE_FLOOR
    for (f, db), fl in zip(bins, floor):
        if db >= fl + thr:
            over.append((f, db))
    if not over:
        return []

    spacing = bin_spacing_hz(bins)
    merge_hz_eff = max(MERGE_BIN_HZ, 1.6 * spacing)

    # Parametri per banda
    bridge_max_hz = BRIDGE_MAX_HZ_DEFAULT
    min_peaks_req = MIN_PEAKS_PER_BLOB_DEF
    if band == "24":
        bridge_max_hz = BRIDGE_MAX_HZ_24
        min_peaks_req = MIN_PEAKS_PER_BLOB_24
    elif band in ("52","58"):
        bridge_max_hz = BRIDGE_MAX_HZ_58
        min_peaks_req = MIN_PEAKS_PER_BLOB_58


    blobs, cur = [], [over[0]]
    last_f = over[0][0]
    for f, db in over[1:]:
        gap_hz = (f - last_f) * 1e6
        if gap_hz <= merge_hz_eff or gap_hz <= bridge_max_hz:
            cur.append((f, db))
        else:
            blobs.append(cur); cur = [(f, db)]
        last_f = f
    blobs.append(cur)



    filtered = []
    for b in blobs:
        fspan_hz = (b[-1][0] - b[0][0]) * 1e6
        min_bw = MIN_BLOB_BW_HZ
        if band in ("52","58"):
            min_bw = max(min_bw, 8.0e6)
        if fspan_hz >= min_bw and len(b) >= min_peaks_req:
            filtered.append(b)
    return filtered




def blob_to_params(blob):
    freqs = [f for f, _ in blob]
    pows  = [10**(db/10.0) for _, db in blob]
    wsum  = sum(pows)
    f0    = sum(f*p for f, p in zip(freqs, pows)) / (wsum if wsum else len(freqs))
    bw_hz = (max(freqs) - min(freqs)) * 1e6
    bw_hz = max(bw_hz * 1.2, 2.0e6)
    return f0, bw_hz

def persist_state():
    try:
        STATE_FILE.write_text(json.dumps({
            "last_trigger_ts": last_trigger_ts,
            "trigger_count": list(trigger_count),
            "params": {
                "PEAK_DB_ABOVE_FLOOR": PEAK_DB_ABOVE_FLOOR,
                "MIN_CONSEC_SWEEPS": MIN_CONSEC_SWEEPS,
                "MERGE_BIN_HZ": MERGE_BIN_HZ,
                "MIN_BLOB_BW_HZ": MIN_BLOB_BW_HZ,
                "COOLDOWN_S": COOLDOWN_S
            }
        }))
    except Exception:
        pass

def rate_limit_ok():
    now = time.time()
    while trigger_count and now - trigger_count[0] > 60:
        trigger_count.popleft()
    return (len(trigger_count) < MAX_TRIGGERS_PER_MIN)


def write_last_trigger_json(band, f0_mhz, bw_hz):
    try:
        payload = {
            "band": str(band),
            "f0_mhz": round(float(f0_mhz), 3),
            "bw_hz": int(bw_hz),
            "ts_iso": datetime.utcnow().isoformat()+"Z",
            "ts_epoch": time.time()
        }
        CRCP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRIG_JSON, "w") as jf:
            json.dump(payload, jf)
        try: os.chmod(TRIG_JSON, 0o664)
        except Exception: pass
        print(f"📝 last_trigger.json scritto: {payload}")
    except Exception as e:
        print(f"⚠️  write_last_trigger_json errore: {e}")


def send_trigger(band, f0_mhz, bw_hz, hold_s=10):
    if not TRIGGER_FIFO.exists():
        try:
            os.mkfifo(TRIGGER_FIFO)
        except FileExistsError:
            pass
    payload = {
        "band": band,
        "f0_mhz": round(f0_mhz, 3),
        "bw_hz": int(bw_hz),
        "hold_s": int(hold_s),
        "ts": datetime.utcnow().isoformat()+"Z"
    }
    try:
        fd = os.open(TRIGGER_FIFO, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if DEBUG: print(f"[{band}] fifo non pronta: {e}")
        return False
    try:
        os.write(fd, (json.dumps(payload)+"\n").encode("utf-8"))
        return True
    finally:
        os.close(fd)


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

def main():
    # Avvio
    print("▶ RFExplorer trigger daemon avviato.")
    print(f"⚙️  Parametri: TH={PEAK_DB_ABOVE_FLOOR}dB  MIN_SWEEPS={MIN_CONSEC_SWEEPS}  COOLDOWN={COOLDOWN_S}s")
    while True:
        for band in ("58", "52", "24"):  # priorità 5.8 > 5.2 > 2.4
            rows = read_latest_csv(band)
            if rows:
                print(f"📈 RFE[{band}] nuovo sweep {time.strftime('%H:%M:%S')}: bins={len(rows)} file={BANDS[band]['path']}")
            if not rows:
                continue

            # aggiorna storia
            history[band].append(rows)

            # floor + blob sullo sweep corrente
            floor = estimate_floor_dB(rows, band)
            blobs = blobs_from_peaks(rows, floor)

            now = time.time()
            if now - last_trigger_ts[band] < COOLDOWN_S:
                continue
            if not rate_limit_ok():
                continue
            if not blobs:
                print(f"[{band}] nessun blob > {PEAK_DB_ABOVE_FLOOR:.1f}dB sopra floor.")
                continue

            # === Persistenza fuzzy: confronta per sovrapposizione tra sweep ===
            # usa SOLO gli sweep PRECEDENTI (niente auto-voto del corrente)
            hist_list = list(history[band])[:-1]
            sweeps_intervals = []
            for sweep in hist_list:
                fl_sweep = estimate_floor_dB(sweep, band)
                bl_sweep = blobs_from_peaks(sweep, fl_sweep, band=band)
                sweeps_intervals.append([blob_interval_mhz(b) for b in bl_sweep])


            # preferisci i video larghi
            blobs.sort(key=lambda b: (b[-1][0] - b[0][0]), reverse=True)

            candidate = None
            for b in blobs:
                cand_int = blob_interval_mhz(b)
                cand_c, cand_w = blob_center_width_mhz(b)

                # Reject per larghezza eccessiva (tipico Wi-Fi aggregato)
                max_bw = MAX_BLOB_BW_HZ
                if band == "24":
                    max_bw = min(max_bw, MAX_BLOB_BW_24_HZ)
                elif band in ("52","58"):
                    max_bw = min(max_bw, MAX_BLOB_BW_58_HZ)

                if cand_w * 1e6 > max_bw:
                    if DEBUG:
                        print(f"[{band}] skip cand: bw troppo ampia {cand_w:.2f} MHz > {max_bw/1e6:.1f} MHz")
                    continue

                if band == "24" and is_suppressed_24(cand_c):
                    if DEBUG: print(f"🔇 [24] suppress window attiva su ~{cand_c:.2f} MHz → skip")
                    continue

                if band == "24" and is_wifi20_24(cand_c, cand_w):
                    if DEBUG:
                        print(f"🛑 [24] skip cand Wi-Fi20-like: f0={cand_c:.3f} bw≈{cand_w:.2f} (vicino canale Wi-Fi)")
                    continue

                # ✅ ACCETTAZIONE IMMEDIATA IN GOLDEN WINDOW 5.8
                if band == "58" and (GOLDEN_58_LO <= cand_c <= GOLDEN_58_HI):
                    # opzionale: sanity su larghezza
                    if 8.0 <= cand_w <= 30.0:
                        f0, bw = blob_to_params(b)
                        candidate = (f0, bw)
                        print(f"🎯 [58] GOLDEN window hit: f0≈{cand_c:.3f}MHz bw≈{cand_w:.2f}MHz → trigger immediato")
                        break

                votes = 0
                recent = sweeps_intervals[-PERSIST_WINDOW:] if PERSIST_WINDOW > 0 else sweeps_intervals

                # conta quanti sweep recenti hanno almeno un blob
                nonempty = sum(1 for ints in recent if ints)

                # per-band IoU/center tol
                base_center_tol = CENTER_TOL_58 if band in ("58","52") else CENTER_TOL_24
                iou_needed = OVERLAP_FRAC_58 if band in ("58","52") else OVERLAP_FRAC_24

                for ints in recent:
                    hit = False
                    # 1) voto per IoU
                    for itv in ints:
                        if interval_iou(cand_int, itv) >= iou_needed:
                            hit = True
                            break
                    # 2) fallback: tolleranza sul centro (dipende da banda e ampiezza)
                    if (not hit) and base_center_tol > 0:
                        for itv in ints:
                            ic = 0.5*(itv[0] + itv[1]); iw = (itv[1] - itv[0])
                            dyn_tol = max(base_center_tol, 0.5*max(cand_w, iw))
                            if abs(cand_c - ic) <= dyn_tol and min(cand_w, iw)/max(cand_w, iw) >= 0.3:
                                hit = True
                                break
                    votes += 1 if hit else 0

                    print(
                        f"🧪 [{band}] cand f0={cand_c:.3f}MHz bw≈{cand_w:.2f}MHz  "
                        f"🗳️ {votes}/{MIN_CONSEC_SWEEPS}  (IoU≥{iou_needed:.2f} | ±{base_center_tol:.1f}MHz)"
                    )

                    # --- nuova logica "required" ---
                    required = MIN_CONSEC_SWEEPS

                    # bootstrap: se non abbiamo storico utile, 1 voto basta
                    if nonempty == 0:
                        required = 1

                    # finestra "golden" a 5.8 (5771–5791): 1 voto basta
                    if band == "58" and (GOLDEN_58_LO <= cand_c <= GOLDEN_58_HI):
                        required = 1

                    # a 2.4 riduciamo i falsi richiedendo più persistenza
                    if band == "24":
                        required = max(required, 3)

                    # Fallback: se eravamo in modalità "required==1" (bootstrap o golden)
                    # ma non abbiamo raccolto voti (history vuota/non matching), accetta lo stesso.
                    if required <= 1 and votes == 0:
                        f0, bw = blob_to_params(b)
                        candidate = (f0, bw)
                        print(f"🆗 [{band}] bootstrap/golden senza storico: promozione single-sweep (f0≈{cand_c:.3f}MHz bw≈{cand_w:.2f}MHz)")
                        break

                    if votes >= required:
                        f0, bw = blob_to_params(b)
                        candidate = (f0, bw)
                        break


            if candidate:
                f0, bw = candidate
                # 🔧 calcolo intervallo e centro del candidato per il debounce
                cand_int = blob_interval_mhz(b)
                cand_c, cand_w = blob_center_width_mhz(b)

                # 🔧 Debounce: se è lo stesso blob di poco fa, non ritriggherare
                prev = last_blob.get(band)
                if prev:
                    prev_int = prev["interval"]
                    prev_c   = 0.5 * (prev_int[0] + prev_int[1])
                    prev_w   = (prev_int[1] - prev_int[0])
                    same_center = abs(cand_c - prev_c) <= max(5.0, 0.25 * max(cand_w, prev_w))
                    same_iou    = interval_iou(cand_int, prev_int) >= 0.50
                    if (time.time() - prev["ts"] < SAME_BLOB_DEBOUNCE_S) and (same_center or same_iou):
                        # Debounce, stesso blob di poco fa
                        print(f"🔕 [{band}] trigger soppresso (debounce {SAME_BLOB_DEBOUNCE_S:.0f}s): stesso blob")

                        continue

                ok = send_trigger(band, f0, bw, hold_s=12 if band in ("58","52") else 10)
                if ok:
                    write_last_trigger_json(band, f0, bw)
                    # ... dopo write_last_trigger_json(...) ...
                    hold = FOCUS_HOLD_GOLDEN_S if band == "58" else FOCUS_HOLD_S
                    _write_focus(band, hold)
                    # Ora il trigger
                    last_trigger_ts[band] = now
                    trigger_count.append(now)
                    # 🔧 memorizza il blob per futuri debounce
                    last_blob[band] = {"interval": cand_int, "ts": now}
                    print("════════════════════════════════════════")
                    print(
                        f"🚀🧨 ✅ TRIGGER [{band}] → f0={f0:.3f} MHz  bw≈{bw/1e6:.2f} MHz  "
                        f"→ handoff a HackRF / Classifier"
                    )
                    print("════════════════════════════════════════")
                    persist_state()

            else:
                # Blob visti ma niente trigger: chiedi focus breve
                _write_focus(band, FOCUS_HOLD_S)
                tops = [blob_center_width_mhz(b) for b in blobs]
                msg = ", ".join([f"{c:.3f}MHz/{w:.2f}MHz" for c, w in tops[:3]])
                print(f"🤔 [{band}] blobs visti ma NO trigger (persistence): {msg}")


        time.sleep(0.4)



if __name__ == "__main__":
    main()
