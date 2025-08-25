#!/usr/bin/env python3
# IQ → STFT → immagini "dataset-like" stile RFUAV/Phantom (v3.3)
# - Fix: nessuna riga con "dc-notch" (solo args.dc_notch)
# - FFT shift + crop sullo span visibile (--fs-view)
# - Decodifica IQ robusta: --iq-format auto|u8|i8 (default auto; HackRF=u8 offset)
# - HOP configurabile (--hop-div) per più colonne in 0.10 s
# - Baseline bg-sub con fast-settle
# - Prefill iniziale e smoothing temporale anti-glitch

import os, time, glob, threading, argparse, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ap = argparse.ArgumentParser(description="IQ→STFT RFUAV-like (robust v3.3)")
ap.add_argument("--cmap", type=str, default="turbo")
ap.add_argument("--fs", type=float, default=10e6, help="Sample rate Hz (HackRF tipicamente 2e6..10e6)")
ap.add_argument("--fs-view", type=float, default=2e6, help="Span visibile Hz (zoom canale)")
ap.add_argument("--freq-mode", type=str, default="rel", choices=["rel","abs"], help="Asse Y: rel o abs")
ap.add_argument("--duration", type=float, default=0.10, help="Durata pannello s")
ap.add_argument("--mode", type=str, default="EWMA", choices=["LIVE","EWMA","MAX"])
ap.add_argument("--alpha", type=float, default=0.20, help="Alpha EWMA cumulativa")
ap.add_argument("--norm", type=str, default="bgsub", choices=["percentile","bgsub"])
ap.add_argument("--bg-alpha", type=float, default=0.02, help="Alpha baseline (bgsub)")
ap.add_argument("--fast-settle-cols", type=int, default=120)
ap.add_argument("--fast-settle-alpha", type=float, default=0.25)
ap.add_argument("--delta-floor", type=float, default=-10.0)
ap.add_argument("--delta-ceil",  type=float, default= 10.0)
ap.add_argument("--db-floor", type=float, default=None)
ap.add_argument("--db-ceil",  type=float, default=None)
ap.add_argument("--gamma", type=float, default=0.9)
ap.add_argument("--detrend", action="store_true")
ap.add_argument("--dc-notch", action="store_true")
ap.add_argument("--min-save-cols", type=int, default=64)
ap.add_argument("--hop-div", type=int, default=8, help="HOP = NFFT//hop-div (più alto → più colonne)")
ap.add_argument("--nfft", type=int, default=4096)
ap.add_argument("--iq-format", type=str, default="auto", choices=["auto","u8","i8"],
                help="HackRF: u8 (offset). auto prova u8 e fallback a i8 se serve.")
args = ap.parse_args()

# scrivi span file per ciascuna banda che gestisci qui
try:
    # se lo script ha già la mappa delle FIFO/bande, per ogni banda scrivi lo stesso fs-view
    for band in ("24","52","58"):
        with open(f"/tmp/span_{band}.txt","w") as f:
            f.write(str(int(args.fs_view)))
except Exception:
    pass
    
def build_orange_red():
    colors = [(1.00,0.95,0.75),(1.00,0.90,0.60),(1.00,0.75,0.30),(1.00,0.55,0.15),(0.90,0.20,0.05),(0.75,0.00,0.00)]
    return LinearSegmentedColormap.from_list("orange_to_red", colors, N=256)
def get_cmap(name):
    if name.lower() in ("orange_red","orange-to-red","dataset"): return build_orange_red()
    try: return plt.get_cmap(name)
    except: return build_orange_red()
CMAP = get_cmap(args.cmap)

FIFO = { "24":"/tmp/hackrf_24.iq", "58":"/tmp/hackrf_58.iq", "52":"/tmp/hackrf_52.iq" }
OUTD = "/tmp/tiles"; os.makedirs(OUTD, exist_ok=True)
LIVE = { b: os.path.join(OUTD, f"{b}_live.png") for b in FIFO }
CUM  = { b: os.path.join(OUTD, f"{b}_cum.png")  for b in FIFO }
CENTER = { "24":"/tmp/center_24.txt", "58":"/tmp/center_58.txt", "52":"/tmp/center_52.txt" }

FS   = float(args.fs)
NFFT = int(args.nfft)
HOP  = max(1, NFFT//max(2, args.hop_div))
time_per_col = HOP / FS
T_PIX = max(96, int(np.ceil(args.duration / max(time_per_col, 1e-9))))
F_PIX = 920

FIG_W, FIG_H = 1800, 1200
SPEC_W, SPEC_H = 1395, 920
DPI = 100
SAVE_PERIOD_S = 1.5
STAMPED_EVERY_S = 7.0
MAX_STAMPED = 200

def read_center(path):
    try:
        with open(path,"r") as f: return float(f.read().strip())
    except: return None

def interp_to_fpix(vec):
    if vec.size == F_PIX: return vec.astype(np.float32, copy=False)
    fx = np.linspace(0,1,vec.size, dtype=np.float32)
    fy = np.linspace(0,1,F_PIX,   dtype=np.float32)
    return np.interp(fy, fx, vec).astype(np.float32)

def col_percentile(db_vec):
    x = db_vec.copy()
    if args.detrend: x -= np.median(x)
    if args.db_floor is not None and args.db_ceil is not None and args.db_ceil > args.db_floor:
        lo, hi = args.db_floor, args.db_ceil
    else:
        lo, hi = np.percentile(x, [5,99.5])
        if hi - lo < 1e-6: hi = lo + 1e-6
    x = np.clip(x, lo, hi)
    col = (x - lo)/(hi - lo)
    if args.gamma != 1.0: col = np.power(col, args.gamma)
    return interp_to_fpix(col)

def col_bgsub(db_vec, baseline_db, col_idx):
    x = interp_to_fpix(db_vec)
    if args.detrend: x -= np.median(x)
    if col_idx == 0 and np.all(baseline_db == 0): baseline_db[:] = x
    ba = args.bg_alpha if col_idx >= args.fast_settle_cols else args.fast_settle_alpha
    baseline_db[:] = (1.0 - ba)*baseline_db + ba*x
    delta = x - baseline_db
    lo, hi = args.delta_floor, args.delta_ceil
    if hi - lo < 1e-6: hi = lo + 1e-6
    d = np.clip(delta, lo, hi)
    col = (d - lo)/(hi - lo)
    if args.gamma != 1.0: col = np.power(col, args.gamma)
    return col

def atomic_save(fig, out_path):
    # Crea sempre la cartella di destinazione
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(out_path)
    ext = ext or ".png"
    tmp = f"{base}.tmp{ext}"

    try:
        fig.savefig(
            tmp,
            format=ext.lstrip("."),
            facecolor="white",
            dpi=DPI,
            bbox_inches=None,
            pad_inches=0
        )
    except Exception as e:
        # niente tmp -> niente replace; chiudi e rientra
        print(f"[savefig] error for {out_path}: {e}")
        plt.close(fig)
        return

    try:
        # Se per qualsiasi motivo il tmp non c'è, evita l'eccezione
        if os.path.exists(tmp):
            os.replace(tmp, out_path)
        else:
            print(f"[atomic_save] tmp file missing: {tmp}")
    finally:
        plt.close(fig)


def render(img_tf01, out_path, center_freq=None):
    m = img_tf01.copy()
    fig = plt.figure(figsize=(FIG_W/DPI, FIG_H/DPI), dpi=DPI, facecolor="white")
    ax_w, ax_h = SPEC_W/FIG_W, SPEC_H/FIG_H
    left, bottom = (1-ax_w)/2, (1-ax_h)/2
    ax = fig.add_axes([left, bottom, ax_w, ax_h], facecolor="white")

    if args.freq_mode == "rel":
        fmin_mhz, fmax_mhz = 0.0, args.fs_view/1e6
    else:
        cf = center_freq or 0.0
        fmin_mhz, fmax_mhz = (cf-args.fs_view/2)/1e6, (cf+args.fs_view/2)/1e6

    duration_s = T_PIX * (HOP/FS)
    extent = [0.00, duration_s, fmin_mhz, fmax_mhz]
    ax.imshow(m, cmap=CMAP, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0)

    xt = np.linspace(0.00, duration_s, 6)
    # extent già è [0, duration_s, fmin_mhz, fmax_mhz] con fmin=0, fmax=fs_view/1e6
    # Trasforma i tick per mostrarli "centrati" (−BW/2 … +BW/2) pur tenendo la mappa 0..BW
    yt = np.linspace(0.0, args.fs_view/1e6, 6)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:.1f}" for y in yt])

    if args.freq_mode == "abs" and center_freq:
        ax.text(0.98, 0.98, f"CF {center_freq/1e6:.3f} MHz\nBW {args.fs_view/1e6:.2f} MHz",
                transform=ax.transAxes, ha="right", va="top", fontsize=14, color="black",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"))

    atomic_save(fig, out_path)

def stamped(prefix, img_tf01, center_freq=None):
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = os.path.join(OUTD, f"{prefix}_cum_{ts}.png")
    render(img_tf01, p, center_freq)
    files = sorted(glob.glob(os.path.join(OUTD, f"{prefix}_cum_*.png")), key=os.path.getmtime)
    if len(files) > MAX_STAMPED:
        for f in files[:len(files)-MAX_STAMPED]:
            try: os.remove(f)
            except: pass

def decode_iq(raw_bytes, state):
    # Default u8 offset (HackRF). In auto, prova u8 e se la media è strana riprova i8.
    b = np.frombuffer(raw_bytes, dtype=np.uint8)
    if args.iq_format == "i8":
        v = b.view(np.int8).astype(np.float32)
        I = v[0::2] / 128.0; Q = v[1::2] / 128.0
    else:
        f = b.astype(np.float32)
        I = (f[0::2] - 127.5) / 127.5
        Q = (f[1::2] - 127.5) / 127.5
        if args.iq_format == "auto" and state["auto_probe"] < 3:
            mI, mQ = float(np.mean(I)), float(np.mean(Q))
            if abs(mI) < 0.25 and abs(mQ) < 0.25:
                state["fmt"] = "u8"; state["auto_probe"] += 1
            else:
                v = b.view(np.int8).astype(np.float32)
                I = v[0::2] / 128.0; Q = v[1::2] / 128.0
                state["fmt"] = "i8"; state["auto_probe"] += 1
    return (I + 1j*Q).astype(np.complex64)

def band_worker(band, fifo_path, live_out, cum_out, center_file):
    if not os.path.exists(fifo_path): os.mkfifo(fifo_path)
    img_live = np.zeros((F_PIX, T_PIX), np.float32)
    img_cum  = np.zeros((F_PIX, T_PIX), np.float32)
    baseline_db = np.zeros(F_PIX, np.float32)
    win = np.hanning(NFFT).astype(np.float32)
    state = {"fmt": args.iq_format, "auto_probe": 0}

    with open(fifo_path, "rb", buffering=0) as f:
        buf = np.zeros(NFFT, np.complex64); filled = 0
        last_save = 0.0; last_stamp = 0.0; col_idx = 0

        while True:
            need = (NFFT - filled) * 2  # 2 byte per (I,Q) in 8 bit
            raw = f.read(need)
            if not raw: time.sleep(0.001); continue

            c = decode_iq(raw, state)
            take = min(c.size, NFFT - filled)
            buf[filled:filled+take] = c[:take]; filled += take
            c = c[take:]
            if filled < NFFT: continue

            x = buf * win

            spec = np.fft.fftshift(np.fft.fft(x, n=NFFT))
            psd  = np.abs(spec)**2

            mid = NFFT // 2                       # DC
            n_pos = max(16, int(round(NFFT * (args.fs_view / FS))))
            n_pos = min(n_pos, NFFT - mid)        # clamp per sicurezza

            view = psd[mid : mid + n_pos]         # SOLO 0 → +fs_view

            # notch DC sul primo bin (0 Hz), se richiesto
            if args.dc_notch and n_pos >= 3:
                view[0] = 0.5*(view[1] + view[2])


            db = 10*np.log10(np.maximum(view, 1e-12))

            # Colonna
            if args.norm == "bgsub":
                col01 = col_bgsub(db, baseline_db, col_idx)
            else:
                col01 = col_percentile(db)

            # Prefill e smoothing anti-glitch
            if col_idx == 0:
                col_img = col01.reshape(F_PIX,1)
                img_live[:] = np.repeat(col_img, T_PIX, axis=1)
                img_cum[:]  = img_live
            else:
                # in LIVE niente smoothing → niente “righe verticali”
                img_live = np.hstack([img_live[:,1:], col01.reshape(F_PIX,1)])


            # Accumulo
            if args.mode == "MAX":
                img_cum = np.maximum(img_cum, img_live)
            elif args.mode == "EWMA":
                img_cum = (1.0-args.alpha)*img_cum + args.alpha*img_live
            else:
                img_cum = img_live.copy()

            now = time.time()
            cfreq = read_center(center_file)

            # Salvataggi
            if col_idx >= args.min_save_cols and now - last_save >= SAVE_PERIOD_S:
                if int(now // SAVE_PERIOD_S) % 2 == 0:
                    render(img_live, live_out, center_freq=cfreq)
                else:
                    render(img_cum,  cum_out,  center_freq=cfreq)
                last_save = now
            if col_idx >= args.min_save_cols and now - last_stamp >= STAMPED_EVERY_S:
                stamped(band, img_cum, center_freq=cfreq); last_stamp = now

            # overlap shift
            if HOP < NFFT:
                buf[:-HOP] = buf[HOP:]; filled = NFFT - HOP
            else:
                filled = 0

            col_idx += 1

def main():
    for b in FIFO:
        threading.Thread(target=band_worker, args=(b, FIFO[b], LIVE[b], CUM[b], CENTER[b]), daemon=True).start()
    #threading.Thread(target=band_worker, args=("58", FIFO["58"], LIVE["58"], CUM["58"], CENTER["58"]), daemon=True).start()
    print(f"✅ RFUAV-like v3.3 attivo. fs={args.fs/1e6:.2f} Msps, span={args.fs_view/1e6:.2f} MHz, NFFT={NFFT}, hop_div={args.hop_div}")
    print(f"   norm={args.norm} (bg-alpha={args.bg_alpha}, fast {args.fast_settle_alpha} x {args.fast_settle_cols}), delta=[{args.delta_floor},{args.delta_ceil}] dB")
    print(f"   out: {OUTD}/*_live.png, *_cum.png")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("bye")

if __name__ == "__main__":
    main()
