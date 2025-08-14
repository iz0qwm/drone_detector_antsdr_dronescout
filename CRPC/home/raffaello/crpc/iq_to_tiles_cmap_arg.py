#!/usr/bin/env python3
# IQ → STFT → immagini "dataset-like" con colormap selezionabile da riga di comando

import os, time, numpy as np, datetime as dt, glob, threading, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

# ---------- Argomenti da riga di comando ----------
parser = argparse.ArgumentParser(description="IQ to tiles converter")
parser.add_argument("--cmap", type=str, default="turbo",
                    help="Colormap: turbo, viridis, orange_red, ecc.")
args = parser.parse_args()

# ---------- Colormap personalizzata ----------
def build_orange_red():
    colors = [
        (1.00, 0.95, 0.75),  # quasi crema
        (1.00, 0.90, 0.60),  # arancione chiaro
        (1.00, 0.75, 0.30),  # arancione medio
        (1.00, 0.55, 0.15),  # arancio-rosso
        (0.90, 0.20, 0.05),  # rosso acceso
        (0.75, 0.00, 0.00),  # rosso intenso
    ]
    return LinearSegmentedColormap.from_list("orange_to_red", colors, N=256)

def get_cmap(name):
    if name.lower() in ("orange_red", "orange-to-red", "dataset"):
        return build_orange_red()
    try:
        return plt.get_cmap(name)
    except Exception:
        return build_orange_red()

CMAP = get_cmap(args.cmap)

# ---------- FIFO ----------
FIFO_24 = "/tmp/hackrf_24.iq"
FIFO_58 = "/tmp/hackrf_58.iq"
FIFO_52 = "/tmp/hackrf_52.iq"

# ---------- Output ----------
OUT_DIR = "/tmp/tiles"
os.makedirs(OUT_DIR, exist_ok=True)
LIVE_24 = os.path.join(OUT_DIR, "24_live.png")
LIVE_58 = os.path.join(OUT_DIR, "58_live.png")
LIVE_52 = os.path.join(OUT_DIR, "52_live.png")
CUM_24  = os.path.join(OUT_DIR, "24_cum.png")
CUM_58  = os.path.join(OUT_DIR, "58_cum.png")
CUM_52  = os.path.join(OUT_DIR, "52_cum.png")
# ---------- Center frequency file ----------
CENTER_FILE_24 = "/tmp/center_24.txt"
CENTER_FILE_58 = "/tmp/center_58.txt"
CENTER_FILE_52 = "/tmp/center_52.txt"
# ---------- Locks per salvataggio atomico ----------
RENDER_LOCKS = {"24": threading.Lock(), "58": threading.Lock(), "52": threading.Lock()}

# ---------- STFT / Layout ----------
FS   = 10_000_000
NFFT = 4096
HOP  = NFFT//4
FIG_W, FIG_H = 1800, 1200
SPEC_W, SPEC_H = 1395, 920
DPI = 100
DURATION_S = 0.10
FMAX_LABEL = 5e7
T_PIX = 480
F_PIX = 920
SMOOTH = True
P_LO, P_HI = 5, 99.5

# ---------- Cumulativa ----------
CUM_MODE = os.environ.get("CUM_MODE", "EWMA").upper()
EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.15"))
SAVE_PERIOD_S = 2
STAMPED_EVERY_S = 7.0
MAX_STAMPED_PER_BAND = 200

def ensure_fifo(path):
    if not os.path.exists(path):
        os.mkfifo(path)

def read_center(path):
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except:
        return None

def to_col01(db_vec):
    col = db_vec - np.median(db_vec)
    if SMOOTH and col.size >= 3:
        tmp = col.copy()
        tmp[1:-1] = 0.2*col[:-2] + 0.6*col[1:-1] + 0.2*col[2:]
        col = tmp
    lo, hi = np.percentile(col, [P_LO, P_HI])
    if hi - lo < 1e-6: hi = lo + 1e-6
    col01 = np.clip((col - lo) / (hi - lo), 0, 1).astype(np.float32)
    f_native = np.linspace(0, 1, col01.size, dtype=np.float32)
    f_axis   = np.linspace(0, 1, F_PIX, dtype=np.float32)
    return np.interp(f_axis, f_native, col01).astype(np.float32)

def atomic_savefig(fig, out_path, **kwargs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".png"
    tmp = f"{base}.tmp{ext}"
    fmt = ext.lstrip(".").lower()
    try:
        fig.savefig(tmp, format=fmt, **kwargs)
    except Exception as e:
        print(f"[atomic_savefig] Errore in savefig per {out_path}: {e}")
        plt.close(fig)
        return
    try:
        os.replace(tmp, out_path)
    except FileNotFoundError:
        print(f"[atomic_savefig] File temporaneo mancante: {tmp}")
    except Exception as e:
        print(f"[atomic_savefig] Errore nel rename {tmp} -> {out_path}: {e}")

def render_dataset_like(img_tf01, out_path, title="Sweep", center_freq=None, band_key=None):
    m = img_tf01.copy()
    fig = plt.figure(figsize=(FIG_W/DPI, FIG_H/DPI), dpi=DPI, facecolor="white")
    ax_w = SPEC_W / FIG_W; ax_h = SPEC_H / FIG_H
    left = (1 - ax_w)/2;   bottom = (1 - ax_h)/2
    ax = fig.add_axes([left, bottom, ax_w, ax_h], facecolor="white")
    extent = [0.00, DURATION_S, 0.0, FMAX_LABEL]
    ax.imshow(m.T, cmap=CMAP, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0)
    xt = np.linspace(0.00, DURATION_S, 6)
    ax.set_xticks(xt); ax.set_xticklabels([f"{t:.2f}" for t in xt])
    yt = np.linspace(0, FMAX_LABEL, 6)
    ax.set_yticks(yt); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(round(y/1e7))}"))
    ax.text(0.0, FMAX_LABEL*1.015, "1e7", ha="left", va="bottom", fontsize=14)
    ax.set_title(title, fontsize=16, pad=8)
    if center_freq:
        ax.text(0.98, 0.98, f"{center_freq/1e6:.1f} MHz", transform=ax.transAxes,
                ha="right", va="top", fontsize=14, color="black",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"))
    lock = RENDER_LOCKS.get(band_key or "", None)
    if lock:
        with lock:
            atomic_savefig(fig, out_path, facecolor="white", dpi=DPI, bbox_inches=None, pad_inches=0)
    else:
        atomic_savefig(fig, out_path, facecolor="white", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)

def save_stamped_copy(prefix, img_tf01, center_freq=None):
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(OUT_DIR, f"{prefix}_cum_{ts}.png")
    render_dataset_like(img_tf01, path, title="Sweep", center_freq=center_freq, band_key=prefix)
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_cum_*.png")), key=os.path.getmtime)
    if len(files) > MAX_STAMPED_PER_BAND:
        for f in files[:len(files) - MAX_STAMPED_PER_BAND]:
            try: os.remove(f)
            except: pass

def band_worker(fifo_path, live_out, cum_out, prefix, center_file):
    ensure_fifo(fifo_path)
    img_live = np.zeros((F_PIX, T_PIX), np.float32)
    img_cum  = np.zeros((F_PIX, T_PIX), np.float32)
    win = np.hanning(NFFT).astype(np.float32)
    with open(fifo_path, "rb", buffering=0) as f:
        buf = np.zeros(NFFT, np.complex64)
        filled = 0
        last_save = 0.0
        last_stamp = 0.0
        while True:
            need = (NFFT - filled) * 2
            raw = f.read(need)
            if not raw:
                time.sleep(0.001); continue
            iq = np.frombuffer(raw, dtype=np.int8)
            if iq.size < 2: continue
            if iq.size % 2 == 1: iq = iq[:-1]
            I = iq[0::2].astype(np.float32); Q = iq[1::2].astype(np.float32)
            c = (I + 1j*Q) / 128.0

            take = min(c.size, NFFT - filled)
            buf[filled:filled+take] = c[:take]
            filled += take
            c = c[take:]
            if filled < NFFT:
                continue

            x = buf * win
            spec = np.fft.fft(x, n=NFFT)
            psd  = np.abs(spec)**2
            half = psd[:NFFT//2]
            db = 10*np.log10(np.maximum(half, 1e-12))
            col01 = to_col01(db)

            img_live = np.hstack([img_live[:, 1:], col01.reshape(F_PIX, 1)])
            if CUM_MODE == "MAX":
                img_cum = np.maximum(img_cum, img_live)
            else:
                img_cum = (1.0 - EWMA_ALPHA) * img_cum + EWMA_ALPHA * img_live

            now = time.time()
            cfreq = read_center(center_file)
            if now - last_save >= SAVE_PERIOD_S:
                if int(now // SAVE_PERIOD_S) % 2 == 0:
                    render_dataset_like(img_live, live_out, title="Sweep", center_freq=cfreq, band_key=prefix)
                else:
                    render_dataset_like(img_cum, cum_out, title="Sweep", center_freq=cfreq, band_key=prefix)
                last_save = now

            if now - last_stamp >= STAMPED_EVERY_S:
                save_stamped_copy(prefix, img_cum, center_freq=cfreq)
                last_stamp = now

            if HOP < NFFT:
                buf[:-HOP] = buf[HOP:]
                filled = NFFT - HOP
            else:
                filled = 0

def main():
    t1 = threading.Thread(target=band_worker, args=(FIFO_24, LIVE_24, CUM_24, "24", CENTER_FILE_24), daemon=True)
    t2 = threading.Thread(target=band_worker, args=(FIFO_58, LIVE_58, CUM_58, "58", CENTER_FILE_58), daemon=True)
    t3 = threading.Thread(target=band_worker, args=(FIFO_52, LIVE_52, CUM_52, "52", CENTER_FILE_52), daemon=True)
    t1.start(); t2.start(); t3.start()
    print(f"✅ IQ→tiles attivo. Colormap: {args.cmap}")
    print(f"   istantanee: {os.path.basename(LIVE_24)}, {os.path.basename(LIVE_58)}, {os.path.basename(LIVE_52)}")
    print(f"   cumulative: {os.path.basename(CUM_24)},  {os.path.basename(CUM_58)},  {os.path.basename(CUM_52)}  (+ copie timestampate)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("👋 stop")

if __name__ == "__main__":
    main()
