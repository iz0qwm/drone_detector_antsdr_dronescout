#!/usr/bin/env python3
# RF scan classifier — versione capability-first (senza fingerprint DB)
#
# Parametri di esecuzione (rf_scan_classifier.py):
#
#   --min-track-len 1
#       Lunghezza minima (in numero di sweep consecutivi) di un track RF perché sia considerato.
#       Serve per scartare tracce troppo corte/instabili.
#
#   --poll-sec 0.25
#       Intervallo di polling (secondi) con cui il classifier rilegge tracks_current.json.
#       Valore basso = risposta più rapida ma più carico CPU.
#
#   --cap-margin 1.0
#       Margine di confidenza per la classificazione capability. 
#       Più alto → switch di famiglia più conservativo.
#
#   --cap-hold-s 2.0
#       Tempo minimo (in secondi) prima di confermare uno switch di famiglia 
#       nel classificatore capability. Evita rimbalzi.
#
#   --mem-ttl-s 15.0
#       TTL (secondi) della memoria bande viste in BandMemory. 
#       Influenza la stabilità della classificazione multi-banda.
#
#   --peak-prom-db 4.0
#       Prominenza minima (dB) per considerare un picco nello spettro.
#       Usato da count_multi_peaks_from_psd.
#
#   --peak-min-spacing 0.5
#       Spaziatura minima (MHz) tra picchi per non contarli come duplicati.
#       Se 0 = auto in base alla larghezza.
#
#   --gate-sec 22
#       Finestra temporale (secondi) di validità di un trigger RFE. 
#       Solo le tracce RF entro questo tempo possono essere "gated" come UAV.
#
#   --gate-center-mhz 16
#       Tolleranza massima (MHz) sulla frequenza centrale rispetto al f0 del trigger
#       per accettare il gating.
#
#   --squelch-after-trigger-s 22
#       Vedi la prima detection con tag [GATE] subito dopo il trigger.
#       Le successive non‑gated entro 22 s vengono silenziate (log “🔇 Squelch post‑trigger…”), 
#       quindi non finiscono nella dashboard/alert.
#       Su 2.4 GHz, i panettoni larghi (≥ 45 MHz) non‑gated post‑trigger vengono zittiti in modo esplicito
#
#   --gate-max-expansion 1.8
#       Se vuoi essere ancora più severo, abbassa --gate-max-expansion (es. 1.5)
#
#   (non mettere --gate-allow-wifi80 così da rifiutare i GATE “finti”)
#
# NB: Alcuni altri parametri esistono con default (es. --log-trim-mb, --dets-tail-kb),
#     ma qui sono lasciati ai valori di default.
#
# Esempio di run (come usato da te):
#   /usr/bin/python3 /home/raffaello/crpc/rf_scan_classifier.py \
#       --min-track-len 1 --poll-sec 0.25 --cap-margin 1.0 --cap-hold-s 2.0 \
#       --mem-ttl-s 15.0 --peak-prom-db 4.0 --peak-min-spacing 0.5 \
#       --gate-sec 22 --gate-center-mhz 16


import os, time, json, datetime, sys, argparse
from pathlib import Path
from collections import defaultdict, deque

from capability_classifier import (
    CapabilityClassifier, BandMemory,
    load_latest_rfe_csv_for_band, count_multi_peaks_from_psd
)

TRACKS_CURR = Path("/tmp/crpc_logs/tracks_current.json")
OUT_JSONL   = Path("/tmp/crpc_logs/rfscan.jsonl")
OUT_SNAP    = Path("/tmp/crpc_logs/rfscan_current.json")
LOG_PATH    = Path("/tmp/crpc_logs/assoc.log")

DET_PATH        = Path("/tmp/crpc_logs/detections.jsonl")
DET_HINT_PATH   = Path("/tmp/crpc_logs/detections_hint.jsonl")
CLASSMAP_PATH   = Path("/home/raffaello/dataset/yolo_custom/classmap.json")

LAST_TRIGGER_JSON = Path("/tmp/crpc_logs/last_trigger.json")
# nuovi parametri di gating
GATE_IOU_MIN = float(os.environ.get("GATE_IOU_MIN", "0.20"))   # soglia di overlap min
GATE_CENTER_MULT = float(os.environ.get("GATE_CENTER_MULT", "0.6"))  # moltiplica la tolleranza centre

# === Gating narrowband / analog ===
NARROW_P95_MHZ   = 2.0     # p95 della BW sotto cui considero "narrow"
ANALOG_BW_MHZ    = 1.5     # expected BW quando il classifier dice ANALOG video
MIN_IOU_BW_MHZ   = 1.2     # banda minima per IoU quando narrow
ADAPT_AFTER_S    = 6.0     # dopo quanto tempo dal trigger inizio ad adattare
ADAPT_TAU_S      = 10.0    # costante di tempo per lo shrink

VIDEO_IDS = derive_video_ids(CLASSMAP)
HINT_TIME_S = float(os.environ.get("HINT_TIME_S", "45")) # se usi l'HINT temporale allarga la finestra
# --- Config HINT/classi ---
# Se RF_HINT_DISABLE_CLASS_FILTER=1, non filtrare per classe nei calcoli HINT
HINT_DISABLE_CLASS_FILTER = os.environ.get("RF_HINT_DISABLE_CLASS_FILTER", "0") == "1"
# Classi/nome ammessi per HINT (oltre a VIDEO_IDS); confrontati in lower(), separati da virgola
HINT_NAME_ALLOW = os.environ.get("RF_HINT_NAME_ALLOW", "fpv analogico,fpv_analogico,analog").lower().split(",")

BANDS = {
    "24": (2400.0, 2500.0),
    "58": (5725.0, 5875.0),
    "52": (5170.0, 5250.0),
}

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def iou_1d(lo1, hi1, lo2, hi2):
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    inter = max(0.0, hi - lo)
    uni = max(0.0, (hi1 - lo1)) + max(0.0, (hi2 - lo2)) - inter
    return (inter / uni) if uni > 0 else 0.0

def log(msg):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > args.log_trim_mb * 1024 * 1024:
            try:
                tail = LOG_PATH.read_text(errors="ignore").splitlines()[-500:]
                LOG_PATH.write_text("\n".join(tail) + "\n")
            except Exception:
                pass
        with LOG_PATH.open("a") as f:
            f.write(f"[{now_str()}] {msg}\n")
    except Exception:
        pass
    #print(msg, flush=True)

# ---------- Classmap & hint video ----------
def load_classmap():
    try:
        return json.loads(CLASSMAP_PATH.read_text())
    except Exception:
        return {"by_id": {}, "by_name": {}}

CLASSMAP = load_classmap()

def derive_video_ids(classmap):
    explicit = classmap.get("video_ids")
    if isinstance(explicit, list) and all(isinstance(x, (str,int)) for x in explicit):
        return {str(x) for x in explicit}
    by_id = classmap.get("by_id", {})
    vids = set()
    for cid, name in by_id.items():
        s = str(name).lower()
        if ("image_transmission" in s) or ("video" in s) or ("vts" in s) or ("square" in s):
            vids.add(str(cid))
    return vids or {"3","12"}





def read_last_trigger(max_age_s):
    """Legge /tmp/crpc_logs/last_trigger.json se recente (in base a mtime)."""
    try:
        st = LAST_TRIGGER_JSON.stat()
        age = time.time() - st.st_mtime
        if age > max_age_s:
            return None, age
        d = json.loads(LAST_TRIGGER_JSON.read_text())
        return d, age
    except Exception:
        return None, None

def is_wifi80_band24(band, bw_mhz, hop_rate):
    try:
        b = str(band); bw = float(bw_mhz or 0.0); hop = float(hop_rate or 0.0)
    except Exception:
        return False
    # 2.4 GHz con "panettone" tra 50 e 95 MHz e hop basso -> Wi‑Fi 80 MHz
    return (b == "24") and (70.0 <= bw <= 95.0) and (abs(hop) <= 2.5)

def is_wifi_wide_cluster_24(band, bw_mhz, hop_rate):
    try:
        b = str(band); bw = float(bw_mhz or 0.0); hop = float(hop_rate or 0.0)
    except Exception:
        return False
    return (b == "24") and (45.0 <= bw < 70.0) and (abs(hop) <= 3.0)

def _select_det_source():
    # Preferisci il file HINT, se presente e non vuoto; altrimenti usa detections.jsonl
    try:
        if DET_HINT_PATH.exists() and DET_HINT_PATH.stat().st_size > 0:
            return DET_HINT_PATH
    except Exception:
        pass
    return DET_PATH

def estimate_vts_bw_from_image(track_img, band_key, tail_kb, ts_center=None, max_lines=2000):
    src = _select_det_source()
    if not track_img or not src.exists() or band_key not in BANDS:
        return None
    f_lo, f_hi = BANDS[band_key]
    band_span = f_hi - f_lo
    # raccolgo il massimo tra:
    #  - w_mhz (se presente)  → migliore
    #  - w * span_mhz         → fallback
    max_w_mhz = 0.0
    try:
        with src.open("rb") as f:
            try:
                f.seek(-tail_kb*1024, os.SEEK_END)
            except Exception:
                pass
            tail = f.read().decode(errors="ignore").splitlines()[-max_lines:]
        for line in tail:
            try:
                d = json.loads(line)

                # --- matching per immagine/banda/tempo ---
                det_img = os.path.basename(d.get("image",""))
                same_img  = (det_img == track_img)
                same_band = (str(d.get("band")) == str(band_key))

                ts = float(d.get("ts", 0.0))
                if ts_center is not None and abs(ts - float(ts_center)) > HINT_TIME_S:
                    continue

                # accetta: (stessa immagine) OPPURE (stessa banda e molto vicino nel tempo)
                if not same_img:
                    if not (same_band and ts_center is not None and abs(ts - float(ts_center)) <= HINT_TIME_S/2.0):
                        continue

                # --- filtro classi video-like (più permissivo / disattivabile) ---
                if not HINT_DISABLE_CLASS_FILTER:
                    cls_raw = d.get("cls", None)
                    cls_id  = str(cls_raw) if cls_raw is not None else None
                    name_l  = str(d.get("name","")).lower()
                    classmap_is_empty = (not CLASSMAP.get("by_id")) and (not CLASSMAP.get("by_name")) and (not CLASSMAP.get("video_ids"))
                    is_video = False
                    if not classmap_is_empty:
                        is_video = (cls_id in VIDEO_IDS) or (str(cls_raw) in VIDEO_IDS)
                    # estendi con nomi ammessi (es. "fpv analogico")
                    if (not is_video) and name_l:
                        is_video = any(tok.strip() and tok.strip() in name_l for tok in HINT_NAME_ALLOW)
                    if not is_video:
                        continue

                # --- preferisci w_mhz, altrimenti ricava da w*span ---
                w_mhz = float(d.get("w_mhz") or 0.0)
                if w_mhz > 0:
                    # se abbiamo una larghezza già in MHz, prendila direttamente
                    if w_mhz > max_w_mhz:
                        max_w_mhz = w_mhz
                else:
                    # fallback: usa w normalizzata con lo span
                    w = float(d.get("w", 0.0))
                    span_det = float(d.get("span_mhz") or 0.0)
                    if span_det > 0:
                        cand = w * span_det
                        if cand > max_w_mhz:
                            max_w_mhz = cand
                    else:
                        # niente span: su immagini _cum evitiamo il panettone,
                        # ma su immagini non _cum possiamo usare lo span di banda intera
                        img_lower = det_img.lower()
                        if "_cum" not in img_lower:
                            cand = w * band_span
                            if cand > max_w_mhz:
                                max_w_mhz = cand

            except Exception:
                continue
    except Exception:
        return None

    return max_w_mhz if max_w_mhz > 0 else None


def image_freq_near_f0(track_img, band_key, ts_center, tol_s, f0, center_tol_mhz, max_lines=2000):
    """Ritorna True se in detections.jsonl c'è, per la stessa immagine, una detection con freq_mhz vicina a f0."""
    if not track_img or not DET_PATH.exists() or band_key not in BANDS:
        return False
    try:
        with DET_PATH.open("rb") as f:
            try: f.seek(-args.dets_tail_kb*1024, os.SEEK_END)
            except Exception: pass
            tail = f.read().decode(errors="ignore").splitlines()[-max_lines:]
        for line in tail:
            try:
                d = json.loads(line)
                if os.path.basename(d.get("image","")) != track_img: 
                    continue
                if "freq_mhz" not in d: 
                    continue
                ts = float(d.get("ts", 0.0))
                if abs(ts - float(ts_center)) > tol_s: 
                    continue
                if abs(float(d["freq_mhz"]) - float(f0)) <= float(center_tol_mhz):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False

# ---------- Utility ----------
def _compute_expected_bw_for_iou(
    bw0,                  # banda dal trigger (MHz)
    used_bw,              # banda misurata ora (MHz)
    bw_p95,               # p95 (MHz) delle ultime misure
    last_label,           # etichetta ultima (es. "ANALOG video")
    dt_since_trigger,     # secondi dal trigger
    clip_limit            # limite superiore di banda usato per IoU
):
    expected_bw = float(bw0)

    # 1) Se è analogico o narrow, restringi subito l'aspettativa
    if (last_label or "").upper().startswith("ANALOG"):
        expected_bw = min(expected_bw, ANALOG_BW_MHZ)
    elif bw_p95 > 0 and bw_p95 <= NARROW_P95_MHZ:
        expected_bw = min(expected_bw, max(MIN_IOU_BW_MHZ, 1.2 * bw_p95))

    # 2) Adattamento progressivo verso la banda reale (se molto più stretta)
    if dt_since_trigger > ADAPT_AFTER_S and used_bw > 0 and used_bw < expected_bw * 0.5:
        alpha = min(1.0, (dt_since_trigger - ADAPT_AFTER_S) / ADAPT_TAU_S)
        expected_bw = expected_bw * (1.0 - alpha) + used_bw * alpha

    # 3) Applica clip
    expected_bw = max(MIN_IOU_BW_MHZ, min(expected_bw, clip_limit))
    return expected_bw

def load_tracks(min_len):
    if not TRACKS_CURR.exists(): return []
    try:
        data = json.loads(TRACKS_CURR.read_text() or "[]")
        return [t for t in data if t.get("len", 0) >= min_len]
    except Exception:
        return []

def parse_args():
    p = argparse.ArgumentParser(description="RF scan classifier (capability-first + gating)")
    p.add_argument("--min-track-len", type=int, default=1, help="Lunghezza minima track per essere considerato")
    p.add_argument("--poll-sec", type=float, default=0.25, help="Polling di tracks_current.json (s)")
    p.add_argument("--log-trim-mb", type=int, default=5, help="Trim del log assoc.log (MB)")
    p.add_argument("--dets-tail-kb", type=int, default=200, help="Dimensione tail letta da detections.jsonl (KB)")
    p.add_argument("--dets-tail-lines", type=int, default=2000, help="Righe massimo di detections lette dal tail")
    p.add_argument("--cap-margin", type=float, default=1.0, help="Margine per lo switch di famiglia")
    p.add_argument("--cap-hold-s", type=float, default=2.0, help="Tempo di tenuta prima dello switch (s)")
    p.add_argument("--cap-fps", type=float, default=1.0, help="Hz di aggiornamento stimato")
    p.add_argument("--mem-ttl-s", type=float, default=8.0, help="TTL memoria bande viste (s)")
    p.add_argument("--peak-prom-db", type=float, default=4.0, help="Prominenza minima per i picchi (dB)")
    p.add_argument("--peak-min-spacing", type=float, default=0.0, help="Spaziatura minima stimata (MHz, 0=auto)")
    # Gating dal trigger RFE
    p.add_argument("--gate-sec", type=float, default=3.0, help="Durata del gating post-trigger (s)")
    p.add_argument("--gate-center-mhz", type=float, default=12.0, help="Tolleranza sul centro rispetto a f0 del trigger (MHz)")
    # Squelch: silenzia detections non-gated per una finestra post-trigger (default = gate-sec).
    p.add_argument("--squelch-after-trigger-s", type=float, default=None, help="Finestra (s) di silenziamento delle detections NON-GATE dopo un trigger (default = gate-sec)")
    # Anti-falsi: vincola il gating quando il panettone è troppo ampio o è Wi-Fi 80 su 2.4
    p.add_argument("--gate-max-expansion", type=float, default=1.8, help="Massimo rapporto bw_raw/clip_limit per accettare il gating (es. 1.8 = +80%)")
    p.add_argument("--gate-allow-wifi80", action="store_true", default=False, help="Se presente, consente il gating anche se il segnale appare come Wi-Fi 80 su 2.4")
    return p.parse_args()

args = parse_args()

def _label_from_family(fam):
    if fam == "O4":     return "DJI (O4)"
    if fam == "O23":    return "DJI (O2/O3)"
    if fam == "HUBSAN": return "Hubsan LEAS"
    if fam == "AUTEL": return "Autel Skylink"
    if fam == "ANALOG": return "ANALOG video"
    if fam == "5GVIDEO": return "5GHz Video"
    if fam == "WIFI":   return "Wi‑Fi"
    return fam or "unknown"

def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SNAP.parent.mkdir(parents=True, exist_ok=True)

    cap = CapabilityClassifier(margin=args.cap_margin, hold_s=args.cap_hold_s, fps=args.cap_fps)
    band_mem = BandMemory(ttl_s=args.mem_ttl_s)

    last_seen = {}
    seen_img_ts = {}
    bw_hist = defaultdict(lambda: deque(maxlen=12))

    print("🔎 RF scan classifier (capability-first) avviato…", flush=True)
    while True:
        tracks = load_tracks(args.min_track_len)

        # subito dopo tracks = load_tracks(...)
        #n_raw = len(tracks)
        #log(f"📥 loaded tracks={n_raw}")

        #gi0, ga0 = read_last_trigger(args.gate_sec)
        #if gi0 and ga0 is not None:
        #    log(f"ℹ️ last_trigger band={gi0.get('band')} f0={gi0.get('f0_mhz')} age={ga0:.2f}s")

        # ---- Raggruppa per (img, ts) e scegli la traccia "più vicina al trigger" ----
        grouped = {}
        gi_focus, ga_focus = read_last_trigger(args.gate_sec)
        f0_focus = bw0_focus = None
        band_focus = None
        if gi_focus and ga_focus is not None and ga_focus <= args.gate_sec:
            try:
                band_focus = str(gi_focus.get("band"))
                f0_focus   = float(gi_focus.get("f0_mhz"))
                bw0_focus  = float(gi_focus.get("bw_hz", gi_focus.get("bw_mhz", 0.0))) / (1e6 if "bw_hz" in gi_focus else 1.0)
            except Exception:
                f0_focus = None

        for tr in tracks:  
            tid = tr.get("track_id")
            ts  = float(tr.get("last_seen", 0))
            img = tr.get("img")
            if img and seen_img_ts.get(img) == ts:
                continue
            if last_seen.get(tid) == ts:
                continue

            key = (img or f"tid:{tid}", ts)
            cur = grouped.get(key)
            take = False
            if cur is None:
                take = True
            else:
                # metrica di scelta
                f_new  = float(tr.get("center_freq_mhz", 0.0))
                f_cur  = float(cur.get("center_freq_mhz", 0.0))
                bw_new = float(tr.get("bandwidth_mhz", 0.0))
                bw_cur = float(cur.get("bandwidth_mhz", 0.0))
                b_new  = str(tr.get("band"))
                if f0_focus is not None and b_new == band_focus:
                    # preferisci la traccia col centro più vicino a f0; penalizza panettoni
                    clip_limit = max((bw0_focus or 0.0) * 1.4, (bw0_focus or 0.0) + 6.0) if (bw0_focus or 0.0) > 0 else None
                    def cost(f_c, bw_c):
                        df = abs(f_c - f0_focus)
                        penalty_bw = 1000.0 if (clip_limit and bw_c > clip_limit * args.gate_max_expansion) else 0.0
                        return df + penalty_bw
                    take = (cost(f_new, bw_new) < cost(f_cur, bw_cur))
                else:
                    # fallback: più larga vince (soft)
                    take = (bw_new > bw_cur)
            if take:
                grouped[key] = tr

        tracks = list(grouped.values())
        #log(f"🧮 grouped tracks kept={len(tracks)}")
        predictions = []
        used_imgs = []

        # ---- Processo le tracce scelte ----
        for tr in tracks:
            tid = tr.get("track_id")
            ts  = float(tr.get("last_seen", 0))
            img = tr.get("img")
            if last_seen.get(tid) == ts:
                continue

            band = str(tr.get("band"))
            f    = float(tr.get("center_freq_mhz", 0.0))
            bw   = float(tr.get("bandwidth_mhz", 0.0))
            hop  = float(tr.get("hop_rate_mhz_s", 0.0))

            bw_raw = bw  # misura grezza

            # Memoria bande per capability
            if band in ("24","52","58"):
                band_mem.mark(band)

            # --- HINT VTS (larghezza da immagine YOLO) ---
            vts_bw_est = estimate_vts_bw_from_image(img, band, args.dets_tail_kb, ts_center=ts, max_lines=args.dets_tail_lines)
            hint_used = False
            bw_used = bw_raw

            if vts_bw_est:
                # 1) Caso analogico/narrow: se vts_bw_est è nell'intervallo atteso, adottalo.
                analog_like = (vts_bw_est <= 2.2) and (vts_bw_est >= 0.5)
                if analog_like and vts_bw_est <= max(bw_raw, 1.5):
                    bw_used = vts_bw_est
                    hint_used = True
                # 2) Sottostima evidente: prendi vts se molto più grande della misura grezza
                elif vts_bw_est > bw_raw * 1.5:
                    bw_used = vts_bw_est
                    hint_used = True
                else:
                    # 3) Trigger recente + coerenza col centro: accetta vts anche se molto stretto
                    gi_tmp, ga_tmp = read_last_trigger(args.gate_sec)
                    try:
                        if gi_tmp and gi_tmp.get("band") == band and (ga_tmp is not None and ga_tmp <= args.gate_sec):
                            f0_tmp  = float(gi_tmp.get("f0_mhz", f))
                            bw0_tmp = float(gi_tmp.get("bw_hz", gi_tmp.get("bw_mhz", 0.0))) / (1e6 if "bw_hz" in gi_tmp else 1.0)
                            center_tol_tmp = max(args.gate_center_mhz, (bw0_tmp or 0.0) * GATE_CENTER_MULT)
                            if abs(f - f0_tmp) <= 1.2 * center_tol_tmp and vts_bw_est < max(0.5 * bw_raw, 0.6):
                                bw_used = vts_bw_est
                                hint_used = True
                    except Exception:
                        pass

            # Log HINT
            if hint_used:
                try:
                    src_tag = "detections_hint.jsonl" if _select_det_source() == DET_HINT_PATH else "detections.jsonl"
                    log(f"🧩 HINT✅ adottato: img={img} band={band} vts_bw={bw_used:.2f}MHz (da {src_tag})")
                except Exception: pass
            else:
                # log diagnostico quando l'HINT c'è ma non viene adottato
                if vts_bw_est:
                    try:
                        log(f"🧩 HINT✳︎ presente ma non adottato: vts_bw={vts_bw_est:.2f} bw_raw={bw_raw:.2f} img={img}")
                    except Exception:
                        pass
            # Euristica Wi‑Fi80: usa bw_raw
            wifi80_candidate = is_wifi80_band24(band, bw_raw, hop)

            # ====== CAPABILITY PRIMA DEL GATE ======
            bw_hist[tid].append(float(bw_used))
            seen = band_mem.get_seen()
            cap_out = cap.classify(list(bw_hist[tid]), seen)
            cap_family = cap_out["family"]
            cap_scores = cap_out["scores"]
            bw_p95 = cap_out["bw_p95"]
            bw_max = cap_out["bw_max"]
            last_label = _label_from_family(cap_family)

            # (opzionale) multi‑peaks
            mp = None
            try:
                freqs, psd = load_latest_rfe_csv_for_band(band)
                if freqs and psd and bw_used >= 10.0:
                    mp = count_multi_peaks_from_psd(freqs, psd,
                                                    center_mhz=f,
                                                    bw_mhz=bw_used,
                                                    prom_db=args.peak_prom_db,
                                                    min_spacing_mhz=(args.peak_min_spacing or None))
            except Exception:
                mp = None

            # ====== GATE dal last_trigger.json (usa bw_p95/last_label appena calcolati) ======
            gate = {"active": False}
            gate_info, gate_age = read_last_trigger(args.gate_sec)
            if gate_info and (gate_info.get("band") == band):
                dt = float(gate_age if gate_age is not None else 999.0)
                try:
                    f0 = float(gate_info.get("f0_mhz", f))
                    bw0 = float(gate_info.get("bw_hz", gate_info.get("bw_mhz", 0.0))) / (1e6 if "bw_hz" in gate_info else 1.0)
                except Exception:
                    f0 = f; bw0 = bw_used

                if dt <= args.gate_sec and bw0 > 0:
                    clip_limit = max(bw0 * 1.4, bw0 + 6.0)

                    expected_bw = _compute_expected_bw_for_iou(
                        bw0=bw0,
                        used_bw=bw_used,
                        bw_p95=bw_p95,
                        last_label=last_label,
                        dt_since_trigger=dt,   # ← niente variabile fantasma
                        clip_limit=clip_limit
                    )

                    # Usa una finestra "stretta" anche lato trigger quando il segnale è analogico/narrow.
                    bw_trig_for_iou = max(MIN_IOU_BW_MHZ, min(expected_bw, clip_limit))
                    bw_for_iou      = min(bw_used, expected_bw)

                    # Finestre comparabili per IoU (stretta vs stretta)
                    trk_lo  = f  - bw_for_iou/2.0
                    trk_hi  = f  + bw_for_iou/2.0
                    trig_lo = f0 - bw_trig_for_iou/2.0
                    trig_hi = f0 + bw_trig_for_iou/2.0
                    center_tol = max(args.gate_center_mhz, bw0 * GATE_CENTER_MULT)
                    delta_f = abs(f - f0)
                    iou = iou_1d(trk_lo, trk_hi, trig_lo, trig_hi)

                    # Debug ora può usare iou/bw_p95 etc.
                    _dbg = (f"(IoU={iou:.2f}) bw0={bw0:.2f} used={bw_used:.2f} p95={bw_p95:.2f} "
                            f"exp={expected_bw:.2f} bw_for_iou={bw_for_iou:.2f} clip={clip_limit:.2f}")

                    # Anti‑falsi
                    reject_reason = None
                    exp_ratio_raw = (bw_raw / clip_limit) if clip_limit > 0 else 999.0
                    exp_ratio_iou = (bw_for_iou / clip_limit) if clip_limit > 0 else 999.0
                    use_clipped_for_exp = (vts_bw_est is not None and vts_bw_est <= clip_limit and delta_f <= center_tol)
                    exp_ratio = exp_ratio_iou if use_clipped_for_exp else exp_ratio_raw

                    if (wifi80_candidate and not args.gate_allow_wifi80 and band == "24"):
                        reject_reason = "wifi80"
                    elif exp_ratio > args.gate_max_expansion:
                        reject_reason = f"expansion {exp_ratio:.2f}x>{args.gate_max_expansion:.2f}"

                    # Bypass per analogico/narrow: se Δf è vicino, accetta anche con IoU basso
                    analog_or_narrow = (last_label.upper().startswith("ANALOG") or (bw_p95 > 0 and bw_p95 <= NARROW_P95_MHZ))
                    iou_ok = (iou >= GATE_IOU_MIN) or (analog_or_narrow and delta_f <= center_tol)

                    # Gate finale
                    if (reject_reason is None) and (delta_f <= center_tol) and iou_ok:
                        if not hint_used:
                            bw_used = min(bw_used, clip_limit)
                        gate.update({
                            "active": True,
                            "dt": round(dt, 2), "f0": round(f0, 3), "bw0": round(bw0, 2),
                            "clip_limit": round(clip_limit, 2),
                            "bw_raw": round(bw_raw, 2), "bw_used": round(bw_used, 2),
                            "delta_f": round(delta_f, 2), "iou": round(iou, 3),
                        })
                        log(f"⛳ GATE T{tid} [{band}] dt={dt:.2f}s Δf={delta_f:.2f}MHz IoU={iou:.2f} "
                            f"f0={f0:.3f} bw0={bw0:.2f} clip<={clip_limit:.2f} bw_raw={bw_raw:.2f} → used={bw_used:.2f} {_dbg}")
                    else:
                        why = f"reason={reject_reason}" if reject_reason else "thresholds"
                        log(f"🟡 Gate miss T{tid} [{band}] dt={dt:.2f}s (≤{args.gate_sec}?) "
                            f"Δf={delta_f:.2f} (≤{center_tol:.2f}?) IoU={iou:.2f} (≥{GATE_IOU_MIN}?) "
                            f"bw0={bw0:.2f} bw={bw_used:.2f} (bw_for_iou={bw_for_iou:.2f} clip_limit={clip_limit:.2f} "
                            f"exp={exp_ratio:.2f}x) {why} {_dbg}")

            # ====== SOFT‑GATE & SQUELCH ======
            soft_gate = False
            gi, ga = read_last_trigger(args.gate_sec)
            if gi and gi.get("band")==band:
                try:
                    f0 = float(gi.get("f0_mhz", f))
                    center_tol = max(args.gate_center_mhz, (bw0 if 'bw0' in locals() else 0.0) * GATE_CENTER_MULT)
                    soft_gate = (ga is not None and ga <= args.gate_sec and abs(f - f0) <= 1.5*center_tol)
                    if not soft_gate and img and image_freq_near_f0(img, band, ts, HINT_TIME_S, f0, 1.2*center_tol):
                        soft_gate = True
                        try:
                            log(f"🧩 SOFT‑GATE per immagine: img={img} band={band} vicino a f0={f0:.3f}MHz")
                        except Exception:
                            pass
                except Exception:
                    pass

            if wifi80_candidate and not args.gate_allow_wifi80 and band == "24" and soft_gate:
                log("↪︎ soft_gate disattivato per Wi‑Fi80 candidate")
                soft_gate = False

            try:
                squelch_win = args.squelch_after_trigger_s if args.squelch_after_trigger_s is not None else args.gate_sec
                gi2, ga2 = read_last_trigger(squelch_win)
                if gi2 and gi2.get("band")==band and ga2 is not None and ga2 <= squelch_win and not (gate.get("active", False) or soft_gate):
                    if band == "24" and bw_raw >= 45.0:
                        log(f"🔇 Squelch post-trigger: T{tid} [{band}] f={f:.3f}MHz bw_raw={bw_raw:.2f} IoU miss → skip")
                        last_seen[tid] = ts
                        if img: used_imgs.append((img, ts))
                        continue
                    log(f"🔇 Squelch post-trigger (general): T{tid} [{band}] non‑gated entro {squelch_win:.1f}s → skip")
                    last_seen[tid] = ts
                    if img: used_imgs.append((img, ts))
                    continue
            except Exception:
                pass

            # ====== Caso esplicito Wi‑Fi80 non‑drone ======
            if wifi80_candidate and not (gate.get("active", False) or soft_gate):
                pred = {
                    "ts": ts, "track_id": tid, "band": band, "img": img,
                    "yolo_conf": tr.get("yolo_conf"),          
                    "tile_path": tr.get("tile_path"),         
                    "center_freq_mhz": round(f,3), "bandwidth_mhz": round(bw,3),
                    "bw_used_mhz": round(bw_used,3), "hop_rate_mhz_s": round(hop,3),
                    "label": "Wi‑Fi (80 MHz)", "family": "WIFI", "score": 0.99, "source": "heuristic",
                    "capability": {"scores": {}, "bw_p95": None, "bw_max": None, "bands_seen": sorted(list(seen))},
                    "multi_peaks": {}, "vts_bw_est_mhz": round(vts_bw_est,2) if vts_bw_est else None,
                    "gate": gate, "hint_used": bool(hint_used), "badges": ["wifi80","non-drone"]
                }
                with OUT_JSONL.open("a") as g:
                    g.write(json.dumps(pred, ensure_ascii=False) + "\n")
                log(f"🚫 Wi‑Fi80 heuristic: T{tid} [24] f={f:.3f}MHz bw_raw={bw_raw:.2f} hop={hop:.2f} → Wi‑Fi 80 MHz")
                predictions.append(pred)
                last_seen[tid] = ts
                if img: used_imgs.append((img, ts))
                continue

            # ====== Predizione finale ======
            label = last_label
            score = 1.0 / (1.0 + max(1e-6, cap_scores.get(cap_family, 0.0)))
            if mp and bw_p95 >= 28.0 and mp.get("n_peaks", 0) >= 4 and cap_family == "O4":
                score = min(0.99, score + 0.10)
            src = "capability+peaks" if mp else "capability"

            pred = {
                "ts": ts, "track_id": tid, "band": band, "img": img,
                "center_freq_mhz": round(f,3), "bandwidth_mhz": round(bw,3),
                "bw_used_mhz": round(bw_used,3), "hop_rate_mhz_s": round(hop,3),
                "label": label, "family": cap_family, "score": round(float(score), 3), "source": src,
                "capability": {
                    "scores": {k: round(v,3) for k,v in cap_scores.items()},
                    "bw_p95": round(bw_p95,2), "bw_max": round(bw_max,2),
                    "bands_seen": sorted(list(seen)),
                },
                "multi_peaks": (mp or {}),
                "vts_bw_est_mhz": round(vts_bw_est,2) if vts_bw_est else None,
                "gate": gate, "hint_used": bool(hint_used),
            }

            with OUT_JSONL.open("a") as g:
                g.write(json.dumps(pred, ensure_ascii=False) + "\n")

            # Δf vs trigger (se presente)
            delta_f = None
            try:
                gi, ga = read_last_trigger(args.gate_sec)
                if gi and gi.get("band") == band:
                    f0 = float(gi.get("f0_mhz"))
                    delta_f = abs(f - f0)
            except Exception:
                pass

            peaks_msg = f" peaks={mp.get('n_peaks')} pk={mp.get('peakiness'):.1f}" if mp else ""
            log(f"✅ RFscan T{tid} [{band}] {f:.3f}MHz bw_raw={bw:.3f} used={bw_used:.3f}"
                + (" +hint" if hint_used else "")
                + f" → {label} ({src} {pred['score']:.2f}) — seen={sorted(list(seen))} bw_p95={bw_p95:.1f} bw_max={bw_max:.1f}{peaks_msg}"
                + (f" Δf={delta_f:.2f}MHz" if delta_f is not None else "")
                + (" [GATE]" if gate.get('active') else ""))

            predictions.append(pred)
            last_seen[tid] = ts
            if img:
                used_imgs.append((img, ts))


        # segna immagini usate per debouncing
        for img, ts in used_imgs:
            seen_img_ts[img] = ts

        if predictions:
            latest = {}
            for p in predictions:
                latest[p["track_id"]] = p
            OUT_SNAP.write_text(json.dumps(list(latest.values()), ensure_ascii=False, indent=2))

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("👋 stop")
        sys.exit(0)
