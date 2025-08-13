#!/usr/bin/env python3
# Realtime waterfall web per HackRF (STFT su IQ da FIFO) - dual source + CF/BW + CPU throttling
import os, io, threading, time
import numpy as np
from flask import Flask, send_file, request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ------------ Config ------------
FIFO_24 = os.environ.get("WF_FIFO_24", "/tmp/hackrf_24.iq")
FIFO_58 = os.environ.get("WF_FIFO_58", "/tmp/hackrf_58.iq")
FIFO_DEFAULT = os.environ.get("WF_FIFO", "/tmp/hackrf_live.iq")

FS      = int(float(os.environ.get("WF_FS_HZ", "10000000")))  # Hz (solo label)
CF_24   = float(os.environ.get("WF24_CF_HZ", "2400000000"))
CF_58   = float(os.environ.get("WF58_CF_HZ", "5800000000"))
BW_DEF  = float(os.environ.get("WF_BW_HZ",  "20000000"))

# Throttling
RENDER_MS   = int(os.environ.get("WF_RENDER_MS", "800"))   # minimo intervallo tra render PNG
CLIENT_MS   = int(os.environ.get("WF_CLIENT_MS", "1200"))  # intervallo refresh <img> in pagina

# STFT / rendering
NFFT = 4096
HOP  = NFFT // 4
CMAP = "turbo"

# PNG grande (resta full-res), ma si scala in pagina
FIG_W, FIG_H   = 900, 600       # più leggera della 1800x1200
SPEC_W, SPEC_H = 698, 460
DPI = 100
DURATION_S = 5.10               # asse X "tempo" fittizio come nel tuo file
FMAX_LABEL = 5e7

T_PIX = 480
F_PIX = 920

# ------------ Stato per sorgenti ------------
class SourceState:
    def __init__(self, fifo_path: str):
        self.fifo = fifo_path
        self.img01 = np.zeros((F_PIX, T_PIX), np.float32)
        self.lock = threading.Lock()
        self.started = False
        # cache PNG
        self.png_bytes = None
        self.png_ts = 0.0
        # parametri usati nell'ultimo render (per cache coerente con CF/BW)
        self.last_cf = None
        self.last_bw = None

SOURCES = {
    "live": SourceState(FIFO_DEFAULT),  # unica sorgente consigliata
    "24":   SourceState(FIFO_24),
    "58":   SourceState(FIFO_58),
}

def _reader_loop(state: SourceState):
    while not os.path.exists(state.fifo):
        time.sleep(0.1)
    f = open(state.fifo, "rb", buffering=0)
    win = np.hanning(NFFT).astype(np.float32)
    buf = np.zeros(NFFT, np.complex64)
    filled = 0
    while True:
        need = (NFFT - filled) * 2
        raw = f.read(need)
        if not raw:
            time.sleep(0.002);  # leggero backoff per non martellare la CPU
            continue
        iq = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        if iq.size % 2: iq = iq[:-1]
        I = iq[0::2]; Q = iq[1::2]
        c = (I + 1j*Q) / 128.0
        take = min(c.size, NFFT - filled)
        buf[filled:filled+take] = c[:take]
        filled += take
        if filled < NFFT:
            continue

        x = buf * win
        spec = np.fft.fft(x, n=NFFT)
        psd = np.abs(spec)**2
        half = psd[:NFFT//2]

        eps = 1e-12
        db = 10*np.log10(np.maximum(half, eps))
        db -= np.median(db)

        f_native = np.linspace(0, 1, half.size, dtype=np.float32)
        f_axis   = np.linspace(0, 1, F_PIX,    dtype=np.float32)
        col = np.interp(f_axis, f_native, db).astype(np.float32)

        if F_PIX >= 3:
            tmp = col.copy()
            tmp[1:-1] = 0.2*col[:-2] + 0.6*col[1:-1] + 0.2*col[2:]
            col = tmp

        lo, hi = np.percentile(col, [5, 99.5])
        if hi - lo < 1e-6: hi = lo + 1e-6
        col01 = np.clip((col - lo) / (hi - lo), 0, 1)

        with state.lock:
            state.img01 = np.hstack([state.img01[:,1:], col01.reshape(F_PIX,1)])

        if HOP < NFFT:
            buf[:-HOP] = buf[HOP:]
            filled = NFFT - HOP
        else:
            filled = 0

def _ensure_reader(src_key: str):
    st = SOURCES.get(src_key) or SOURCES["x"]
    if not st.started:
        threading.Thread(target=_reader_loop, args=(st,), daemon=True).start()
        st.started = True
    return st

def _format_hz(v: float):
    try:
        v = float(v)
    except Exception:
        return "—"
    if v == 0:
        return "—"
    units = [(1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")]
    for scale, name in units:
        if abs(v) >= scale:
            return f"{v/scale:.3f} {name}"
    return f"{v:.0f} Hz"

def _render_png(img01: np.ndarray, cf_hz: float, bw_hz: float) -> bytes:
    m = img01.copy()
    bio = io.BytesIO()
    fig = plt.figure(figsize=(FIG_W/DPI, FIG_H/DPI), dpi=DPI, facecolor="white")
    ax_w = SPEC_W / FIG_W; ax_h = SPEC_H / FIG_H
    left = (1 - ax_w)/2; bottom = (1 - ax_h)/2
    ax = fig.add_axes([left, bottom, ax_w, ax_h], facecolor="white")

    extent = [0.00, DURATION_S, 0.0, FMAX_LABEL]
    ax.imshow(m.T, cmap=CMAP, origin="lower", aspect="auto", extent=extent)

    xt = np.linspace(0.00, DURATION_S, 6)
    ax.set_xticks(xt); ax.set_xticklabels([f"{t:.2f}" for t in xt])
    yt = np.linspace(0, FMAX_LABEL, 6)
    ax.set_yticks(yt); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(round(y/1e7))}"))
    ax.text(0.0, FMAX_LABEL*1.015, "1e7", ha="left", va="bottom", fontsize=12)

    title = f"CF {_format_hz(cf_hz)}  •  BW {_format_hz(bw_hz)}  •  Fs {_format_hz(FS)}"
    ax.set_title(title, fontsize=14, pad=6)

    fig.savefig(bio, format="png", facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return bio.getvalue()

# ------------ Web ------------
app = Flask(__name__)

HTML = f"""
<!doctype html><title>HackRF Waterfall</title>
<style>
  body{{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:12px}}
  .top{{display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap}}
  button{{background:#30363d;color:#eee;border:1px solid #3a3f46;border-radius:8px;padding:6px 10px;cursor:pointer}}
  input{{background:#0b0f16;color:#eee;border:1px solid #3a3f46;border-radius:6px;padding:6px 8px;width:140px}}
  label{{font-size:12px;color:#8b949e;margin-right:6px}}
  .imgwrap{{overflow:auto}}
  img#wf{{transform: scale(1); transform-origin: top left; border:2px solid #333; border-radius:8px}}
  .muted{{color:#8b949e;font-size:12px}}
</style>
<div class="top">
  <div>
    <button id="btnLive">SRC live</button>
    <button id="btn24">SRC 2.4</button>
    <button id="btn58">SRC 5.8</button>
  </div>
  <div>
    <label>CF (Hz)</label><input id="cf" type="number" step="1" placeholder="es. 2400000000">
    <label>BW (Hz)</label><input id="bw" type="number" step="1" placeholder="es. 20000000">
    <button id="apply">Applica</button>
  </div>
  <div class="muted" id="hint"></div>
</div>
<div class="imgwrap">
  <img id="wf" src="" alt="waterfall">
</div>
<script>
const CLIENT_MS = {CLIENT_MS};
function setSrc(src){{
  const u = new URL('/waterfall.png', location.origin);
  u.searchParams.set('src',src);
  const cf = document.getElementById('cf').value;
  const bw = document.getElementById('bw').value;
  if(cf) u.searchParams.set('cf', cf);
  if(bw) u.searchParams.set('bw', bw);
  document.getElementById('wf').src = u.toString();
  localStorage.setItem('wf_src', src);
  document.getElementById('hint').textContent = 'Sorgente: '+src+'  |  CF='+ (cf||'—') +'  BW='+ (bw||'—') + '  | refresh '+CLIENT_MS+' ms';
}}
function tick(){{
  const img = document.getElementById('wf');
  if(!img.src) return;
  const url = new URL(img.src);
  url.searchParams.set('_', Date.now());
  img.src = url.toString();
}}
document.getElementById('btn24').onclick = ()=> setSrc('24');
document.getElementById('btn58').onclick = ()=> setSrc('58');
document.getElementById('btnLive').onclick  = ()=> setSrc('live');
document.getElementById('apply').onclick = ()=> {{
  setSrc(localStorage.getItem('wf_src') || '24');
  localStorage.setItem('wf_cf', document.getElementById('cf').value||'');
  localStorage.setItem('wf_bw', document.getElementById('bw').value||'');
}};
(function(){{
  document.getElementById('cf').value = localStorage.getItem('wf_cf') || '';
  document.getElementById('bw').value = localStorage.getItem('wf_bw') || '';
  const src = localStorage.getItem('wf_src') || '24';
  setSrc(src);
  setInterval(tick, CLIENT_MS);
}})();
</script>
"""

@app.get("/")
def index():
    return HTML

@app.get("/waterfall.png")
def wf_png():
    src = request.args.get("src", "24")
    st = _ensure_reader(src)

    # CF/BW di default per titolo
    if src == "24":
        cf_default = CF_24
    elif src == "58":
        cf_default = CF_58
    else:
        cf_default = 0.0

    try:
        cf = float(request.args.get("cf", cf_default or 0.0))
    except Exception:
        cf = cf_default or 0.0
    try:
        bw = float(request.args.get("bw", BW_DEF or 0.0))
    except Exception:
        bw = BW_DEF or 0.0

    # Cache PNG: se ultimo render è recente e CF/BW non sono cambiati, riusa
    now = time.time()
    with st.lock:
        if (st.png_bytes is not None
            and (now - st.png_ts) * 1000.0 < RENDER_MS
            and st.last_cf == cf and st.last_bw == bw):
            data = st.png_bytes
        else:
            data = _render_png(st.img01, cf, bw)
            st.png_bytes = data
            st.png_ts = now
            st.last_cf, st.last_bw = cf, bw

    return send_file(io.BytesIO(data), mimetype="image/png")

def main():
    app.run(host="0.0.0.0", port=8081, threaded=True)

if __name__ == "__main__":
    main()
