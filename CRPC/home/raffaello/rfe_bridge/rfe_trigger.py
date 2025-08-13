#!/usr/bin/env python3
import time, os, csv, json, math, statistics as stats
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime

# === Config ===
RFE_DIR = Path("/tmp/rfe/scan")
TRIGGER_FIFO = Path("/tmp/hackrf_trigger.fifo")
STATE_FILE = Path("/tmp/rfe_trigger_state.json")

BANDS = {
    "24": {"path": "/tmp/rfe/scan/latest_24.csv", "min_f": 2400.0, "max_f": 2485.0},
    "58": {"path": "/tmp/rfe/scan/latest_58.csv", "min_f": 5725.0, "max_f": 5875.0},
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

PEAK_DB_ABOVE_FLOOR = envf("RFE_PEAK_DB", 4.0)     # dB sopra rumore
MIN_CONSEC_SWEEPS   = envi("RFE_MIN_SWEEPS", 2)    # persistenza
MERGE_BIN_HZ   = envf("RFE_MERGE_HZ", 2.0e6)   # > 0.766 MHz per fondere i bin adiacenti = 2MHz
MIN_BLOB_BW_HZ = envf("RFE_MIN_BW_HZ", 8.0e6)  # solo segnali larghi (video) 8MHz
MAX_TRIGGERS_PER_MIN= envi("RFE_MAX_TPM", 4)
COOLDOWN_S          = envf("RFE_COOLDOWN_S", 15)
HIST_SWEEPS         = envi("RFE_HIST_SWEEPS", 20)
DEBUG               = envb("RFE_DEBUG", True)

SAME_BLOB_DEBOUNCE_S = float(os.getenv("RFE_SAME_BLOB_DEBOUNCE_S", 12))
last_blob = {"24": None, "58": None}
# Persistenza "fuzzy"
OVERLAP_FRAC    = envf("RFE_OVERLAP_FRAC", 0.5)  # soglia di sovrapposizione (IoU) 0..1
CENTER_TOL_MHZ  = envf("RFE_CENTER_TOL_MHZ", 2.0)  # tolleranza alternativa sul centro (MHz)

# Stato in RAM
history = {b: deque(maxlen=HIST_SWEEPS) for b in BANDS}
last_trigger_ts = {b: 0 for b in BANDS}
last_seen_file  = {b: None for b in BANDS}  # (inode, size, mtime)
trigger_count = deque(maxlen=120)  # timestamps

# --- Helper ---

CSV_FREQ_KEYS = ("freq_mhz","frequency","freq","mhz","freqmhz","center_mhz","center")
CSV_DBM_KEYS  = ("power_dbm","dbm","amp_dbm","amp","amplitude","power","db")

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


def estimate_floor_dB(bins):
    """
    Floor per-bin. Se la storia è sufficiente, usa mediana+0.5*MAD sui bin storici.
    Altrimenti usa una baseline locale sullo sweep corrente (mediana mobile + quantile).
    """
    n = len(bins)
    # 1) Se abbiamo abbastanza storia e i bin sono stabili, usa lo schema storico
    have_hist = sum(len(sw) for sw in history.values()) >= 5
    if have_hist:
        per_bin = defaultdict(list)
        for sweeps in history.values():
            for sweep in sweeps:
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
    #    a) mediana mobile (finestra dispari W) per abbassare i picchi
    W = 9  # ampiezza finestra (≈ smoothing leggero)
    half = W // 2
    series = [db for _, db in bins]
    mm = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        mm.append(stats.median(series[a:b]))

    #    b) quantile globale “basso”: se lo sweep è piatto, usarlo come tetto del floor
    qs = sorted(series)
    q25 = qs[max(0, int(0.25 * (n - 1)))] if n else -100.0

    #    c) floor finale: prendi il min tra mediana mobile e q25+1dB (così i picchi restano sopra)
    floor = [min(m, q25 + 1.0) for m in mm]
    return floor


def blobs_from_peaks(bins, floor, band=None):
    # individua bin sopra soglia
    over = []
    for (f, db), fl in zip(bins, floor):
        if db >= fl + PEAK_DB_ABOVE_FLOOR:
            over.append((f, db))
    if not over:
        return []

    # Merge: usa il max tra MERGE_BIN_HZ impostato e 1.3×spacing misurato
    spacing = bin_spacing_hz(bins)
    merge_hz_eff = max(MERGE_BIN_HZ, 1.3 * spacing)

    blobs, cur = [], [over[0]]
    for f, db in over[1:]:
        if (f - cur[-1][0]) * 1e6 <= merge_hz_eff:
            cur.append((f, db))
        else:
            blobs.append(cur); cur = [(f, db)]
    blobs.append(cur)

    # larghezza minima
    filtered = []
    for b in blobs:
        fspan_hz = (b[-1][0] - b[0][0]) * 1e6
        if fspan_hz >= MIN_BLOB_BW_HZ:
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

def main():
    print("▶ RFExplorer trigger daemon avviato.")
    print(f"Parametri: TH={PEAK_DB_ABOVE_FLOOR}dB  MIN_SWEEPS={MIN_CONSEC_SWEEPS}  COOLDOWN={COOLDOWN_S}s")
    while True:
        for band in ("58", "24"):  # priorità 5.8 > 2.4
            rows = read_latest_csv(band)
            if rows:
                print(f"RFE[{band}] nuovo sweep: {time.strftime('%H:%M:%S')} bins={len(rows)} file={BANDS[band]['path']}")
            if not rows:
                continue

            # aggiorna storia
            history[band].append(rows)

            # floor + blob sullo sweep corrente
            floor = estimate_floor_dB(rows)
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
            # Per ogni sweep in memoria, calcola floor “vero” di quello sweep e i suoi blob
            sweeps_intervals = []
            for sweep in history[band]:
                fl_sweep = estimate_floor_dB(sweep)                # <<< FIX QUI
                bl_sweep = blobs_from_peaks(sweep, fl_sweep)       # <<< FIX QUI
                sweeps_intervals.append([blob_interval_mhz(b) for b in bl_sweep])

            # preferisci i video larghi
            blobs.sort(key=lambda b: (b[-1][0] - b[0][0]), reverse=True)

            candidate = None
            for b in blobs:
                cand_int = blob_interval_mhz(b)
                cand_c, cand_w = blob_center_width_mhz(b)

                votes = 0
                recent = sweeps_intervals[-MIN_CONSEC_SWEEPS:] if MIN_CONSEC_SWEEPS > 0 else sweeps_intervals
                for ints in recent:
                    hit = False
                    # 1) voto per IoU (sovrapposizione relativa)
                    for itv in ints:
                        if interval_iou(cand_int, itv) >= OVERLAP_FRAC:
                            hit = True
                            break
                    # 2) fallback: tolleranza sul centro (ampiezza non troppo diversa)
                    if (not hit) and CENTER_TOL_MHZ > 0:
                        for itv in ints:
                            ic = 0.5*(itv[0] + itv[1]); iw = (itv[1] - itv[0])
                            if abs(cand_c - ic) <= CENTER_TOL_MHZ and min(cand_w, iw)/max(cand_w, iw) >= 0.3:
                                hit = True
                                break
                    votes += 1 if hit else 0

                print(f"[{band}] cand f0={cand_c:.3f}MHz bw≈{cand_w:.2f}MHz votes={votes}/{MIN_CONSEC_SWEEPS} "
                      f"(IoU≥{OVERLAP_FRAC:.2f} | ±{CENTER_TOL_MHZ:.1f}MHz)")

                if votes >= MIN_CONSEC_SWEEPS:
                    f0, bw = blob_to_params(b)  # stima pesata per il trigger
                    candidate = (f0, bw)
                    break

            if candidate:
                f0, bw = candidate
                ok = send_trigger(band, f0, bw, hold_s=12 if band == "58" else 10)
                if ok:
                    last_trigger_ts[band] = now
                    trigger_count.append(now)
                    print(f"✅ Trigger {band}: f0={f0:.3f} MHz bw≈{bw/1e6:.2f} MHz")
                    persist_state()
            else:
                tops = [blob_center_width_mhz(b) for b in blobs]
                msg = ", ".join([f"{c:.3f}MHz/{w:.2f}MHz" for c, w in tops[:3]])
                print(f"[{band}] blobs visti ma NO trigger (persistence): {msg}")

        time.sleep(0.4)



if __name__ == "__main__":
    main()
