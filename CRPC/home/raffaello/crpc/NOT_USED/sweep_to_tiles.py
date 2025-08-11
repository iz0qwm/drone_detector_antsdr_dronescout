#!/usr/bin/env python3
import os, time, threading, numpy as np, glob
import matplotlib
matplotlib.use('Agg')  # backend headless per lavorare nei thread
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- INPUT/OUTPUT ----
IN_24 = "/tmp/hackrf_sweeps_text/sweep_2400_2500.txt"   # 2.4 GHz (può contenere sotto-bande)
IN_58 = "/tmp/hackrf_sweeps_text/sweep_5725_5875.txt"   # 5.x GHz (può contenere sotto-bande)
OUT_DIR = "/tmp/tiles"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- GEOMETRIA “DATASET-LIKE” ----
FIG_W, FIG_H = 1800, 1200
SPEC_W, SPEC_H = 1395, 920
DPI = 100

# ---- FINITURA / LOOK ----
DURATION_S = 0.10                 # asse X “stilistico”
FMAX_LABEL = 5e7                  # asse Y “stilistico” (ticks 0..5 con 1e7)
CMAP = "turbo"
CLIP_LO, CLIP_HI = 5, 99.5        # percentili robusti
SMOOTHING = True                  # smoothing temporale leggero

# ---- FINESTRA TEMPORALE & PRODUZIONE TILES ----
TARGET_W = 480         # pixels in X (tempo) del pannello
WINDOW_LINES = 6       # quante righe “tempo” accumulare per tile
STRIDE_LINES = 6       # niente overlap → meno tile
READ_FROM_START = True

# ---- BACK-PRESSURE / RATE LIMIT / RING BUFFER ----
MIN_FREE_MB = 200                  # spazio minimo /tmp
#MAX_PENDING_PER_TAG = 200          # max PNG per tag (24_*.png / 58_*.png)
MAX_PENDING_PER_TAG = 50          # max PNG per tag (24_*.png / 58_*.png)
DELETE_OLDEST_N = 30               # quando superi il limite, rimuovi i più vecchi
SAVE_EVERY_N_TILES = 1             # salva ogni N (1=ogni tile)
#MIN_SECONDS_BETWEEN_SAVES = 0.6    # rate limit per thread
MIN_SECONDS_BETWEEN_SAVES = 5.6    # rate limit per thread

# ----------------- UTILS -----------------
def get_free_mb(path="/tmp"):
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) // (1024 * 1024)

def count_pending(prefix):
    try:
        return len(glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.png")))
    except Exception:
        return 0

def delete_oldest(prefix, n=DELETE_OLDEST_N):
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.png")),
                   key=os.path.getmtime)
    for f in files[:n]:
        try: os.remove(f)
        except: pass

def emergency_trim(dir_path: str, pattern="*.png", keep_last=500):
    files = sorted(glob.glob(os.path.join(dir_path, pattern)),
                   key=os.path.getmtime)
    if len(files) > keep_last:
        for f in files[:len(files)-keep_last]:
            try: os.remove(f)
            except: pass

def follow(path):
    while not os.path.exists(path):
        time.sleep(0.1)
    with open(path, "r", buffering=1, errors="ignore") as f:
        if not READ_FROM_START:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            yield line

def parse_line(line):
    """
    date, time, hz_low, hz_high, bin_hz, samples_per_bin, dBm...
    Esempio:
    2025-08-08, 22:39:55.885395, 2400000000, 2405000000, 500000.00, 20, -55.65, ...
    """
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 7:
        return None
    try:
        hz_low  = int(parts[2])
        hz_high = int(parts[3])
        bin_hz  = float(parts[4])
        # spb  = int(parts[5])  # non usato ora
        expected = max(int(round((hz_high - hz_low) / bin_hz)), 1)
        vals = parts[6:]
        if len(vals) < expected:
            return None
        powers = np.array([float(x) for x in vals[:expected]], dtype=np.float32)
        return hz_low, hz_high, bin_hz, powers
    except Exception:
        return None

def to_uint8(matrix, lo=CLIP_LO, hi=CLIP_HI):
    a = np.percentile(matrix, lo)
    b = np.percentile(matrix, hi)
    if b - a < 1e-6:
        b = a + 1e-6
    m = np.clip((matrix - a) / (b - a), 0, 1)
    return (m * 255.0).astype(np.uint8)

def render_dataset_like(img01, out_path, title="Sweep"):
    """
    img01: array [T, F] normalizzato 0..1 (già dB+whitening+smooth)
    """
    fig = plt.figure(figsize=(FIG_W/DPI, FIG_H/DPI), dpi=DPI, facecolor="white")
    ax_w = SPEC_W / FIG_W
    ax_h = SPEC_H / FIG_H
    left = (1 - ax_w) / 2
    bottom = (1 - ax_h) / 2
    ax = fig.add_axes([left, bottom, ax_w, ax_h], facecolor="white")

    extent = [0.00, DURATION_S, 0.0, FMAX_LABEL]
    ax.imshow(img01, cmap=CMAP, origin="lower", aspect="auto", extent=extent)

    # Assi X (0.00..0.10)
    xt = np.linspace(0.00, DURATION_S, 6)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:.2f}" for t in xt])

    # Assi Y (0..5 con 1e7)
    yt = np.linspace(0, FMAX_LABEL, 6)
    ax.set_yticks(yt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(round(y/1e7))}"))
    ax.text(0.0, FMAX_LABEL*1.015, "1e7", ha="left", va="bottom", fontsize=14)

    ax.set_title(title, fontsize=16, pad=8)
    fig.savefig(out_path, facecolor="white", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)

# ----------------- WORKER (UNO PER FILE) -----------------
def writer_thread(in_path, tag_title):
    """
    tag_title: "24" oppure "58" → usato sia come prefisso file che come titolo
    """
    buffer_rows = []
    tile_idx = 0
    F_MIN = None; F_MAX = None; F_AXIS = None
    last_save_ts = 0.0

    print(f"👀 Watching {in_path} (tag {tag_title})")
    for line in follow(in_path):

        # Fail-safe spazio
        if get_free_mb("/tmp") < MIN_FREE_MB:
            print(f"⛔ /tmp low space: trimming…")
            emergency_trim(OUT_DIR, f"{tag_title}_*.png", keep_last=400)
            time.sleep(0.5)
            continue

        parsed = parse_line(line)
        if not parsed:
            continue
        hz_low, hz_high, bin_hz, powers = parsed

        # Aggiorna F_AXIS dinamico per unire sotto-bande
        if F_MIN is None:
            F_MIN, F_MAX = hz_low, hz_high
        else:
            if hz_low < F_MIN:  F_MIN = hz_low
            if hz_high > F_MAX: F_MAX = hz_high

        step = max(int(bin_hz), 1)
        NEW_AXIS = np.arange(F_MIN, F_MAX, step, dtype=np.float32)

        if (F_AXIS is None) or (NEW_AXIS.size != (F_AXIS.size if F_AXIS is not None else -1)) or \
           (F_AXIS is not None and (NEW_AXIS[0] != F_AXIS[0] or NEW_AXIS[-1] != F_AXIS[-1])):
            # ricampiona le righe già in buffer sulla nuova griglia
            if F_AXIS is not None and buffer_rows:
                buf_resampled = []
                old_native = np.linspace(F_AXIS[0], F_AXIS[-1], num=F_AXIS.size, endpoint=True, dtype=np.float32)
                for old in buffer_rows:
                    buf_resampled.append(np.interp(NEW_AXIS, old_native, old).astype(np.float32))
                buffer_rows = buf_resampled
            F_AXIS = NEW_AXIS

        # Interpola la riga corrente sulla griglia unificata
        f_native = np.linspace(hz_low, hz_high, num=len(powers), endpoint=False, dtype=np.float32)
        row_interp = np.interp(F_AXIS, f_native, powers,
                               left=powers[0], right=powers[-1]).astype(np.float32)
        buffer_rows.append(row_interp)

        # Quando abbiamo la finestra completa → render
        if len(buffer_rows) >= WINDOW_LINES:
            mat = np.stack(buffer_rows[-WINDOW_LINES:], axis=0)   # [T, F]

            # Resize SOLO nel tempo → TARGET_W
            t = np.linspace(0, WINDOW_LINES - 1, TARGET_W, dtype=np.float32)
            mat_lin = np.vstack([
                np.interp(t, np.arange(WINDOW_LINES, dtype=np.float32), mat[:, j])
                for j in range(mat.shape[1])
            ]).T  # [TARGET_W, F_bins]

            # dB (robusto a offset)
            eps = 1e-6
            mat_db = 10.0 * np.log10(np.maximum(mat_lin - mat_lin.min() + eps, eps))

            # Whitening per frequenza (toglie colore medio colonna)
            mat_w = mat_db - np.median(mat_db, axis=0, keepdims=True)

            # Smoothing temporale leggero
            if SMOOTHING and mat_w.shape[0] >= 3:
                m = mat_w.copy()
                m[1:-1,:] = 0.2*mat_w[:-2,:] + 0.6*mat_w[1:-1,:] + 0.2*mat_w[2:,:]
                mat_w = m

            # Normalizza a 0..1 via percentili
            img01 = (to_uint8(mat_w, CLIP_LO, CLIP_HI).astype(np.float32)) / 255.0

            # Back-pressure: ring buffer se troppi PNG
            pending = count_pending(tag_title)
            if pending > MAX_PENDING_PER_TAG:
                delete_oldest(tag_title, n=DELETE_OLDEST_N)

            # Rate limit / drop per densità
            if (tile_idx % SAVE_EVERY_N_TILES) != 0:
                tile_idx += 1
                # avanza finestra senza accumulare eccesso
                if 0 < STRIDE_LINES < WINDOW_LINES:
                    buffer_rows = buffer_rows[-(WINDOW_LINES - STRIDE_LINES):]
                else:
                    buffer_rows.clear()
                continue

            now = time.time()
            if now - last_save_ts < MIN_SECONDS_BETWEEN_SAVES:
                tile_idx += 1
                if 0 < STRIDE_LINES < WINDOW_LINES:
                    buffer_rows = buffer_rows[-(WINDOW_LINES - STRIDE_LINES):]
                else:
                    buffer_rows.clear()
                continue

            # Render “dataset-like”
            out_path = os.path.join(OUT_DIR, f"{tag_title}_{tile_idx:06d}.png")
            render_dataset_like(img01, out_path, title="Sweep")
            last_save_ts = now
            print(f"🖼️ Saved {out_path}")
            tile_idx += 1

            # finestra scorrevole
            if 0 < STRIDE_LINES < WINDOW_LINES:
                buffer_rows = buffer_rows[-(WINDOW_LINES - STRIDE_LINES):]
            else:
                buffer_rows.clear()

def main():
    for p in (IN_24, IN_58):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "a").close()

    print(f"✅ Tile generator running. Output → {OUT_DIR}")
    th24 = threading.Thread(target=writer_thread, args=(IN_24, "24"), daemon=True)
    th58 = threading.Thread(target=writer_thread, args=(IN_58, "58"), daemon=True)
    th24.start(); th58.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("👋 Stop.")

if __name__ == "__main__":
    main()
