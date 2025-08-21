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
CLASSMAP_PATH   = Path("/home/raffaello/dataset/yolo_custom/classmap.json")

LAST_TRIGGER_JSON = Path("/tmp/crpc_logs/last_trigger.json")
# nuovi parametri di gating
GATE_IOU_MIN = float(os.environ.get("GATE_IOU_MIN", "0.20"))   # soglia di overlap min
GATE_CENTER_MULT = float(os.environ.get("GATE_CENTER_MULT", "0.6"))  # moltiplica la tolleranza centre

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

VIDEO_IDS = derive_video_ids(CLASSMAP)
HINT_TIME_S = float(os.environ.get("HINT_TIME_S", "30"))


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

def estimate_vts_bw_from_image(track_img, band_key, tail_kb, ts_center=None, max_lines=2000):
    if not track_img or not DET_PATH.exists() or band_key not in BANDS:
        return None
    f_lo, f_hi = BANDS[band_key]; span = f_hi - f_lo
    max_w = 0.0
    try:
        with DET_PATH.open("rb") as f:
            try:
                f.seek(-tail_kb*1024, os.SEEK_END)
            except Exception:
                pass
            tail = f.read().decode(errors="ignore").splitlines()[-max_lines:]
        for line in tail:
            try:
                d = json.loads(line)
                if os.path.basename(d.get("image","")) != track_img:
                    continue
                if ts_center is not None:
                    ts = float(d.get("ts", 0.0))
                    if abs(ts - float(ts_center)) > HINT_TIME_S:
                        continue
                cls_id = str(d.get("cls"))
                if cls_id in VIDEO_IDS:
                    w = float(d.get("w", 0.0))
                    if w > max_w:
                        max_w = w
            except Exception:
                continue
    except Exception:
        return None
    if max_w <= 0:
        return None
    return max_w * span

# ---------- Utility ----------
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

        grouped = {}
        for tr in tracks:
            tid = tr.get("track_id")
            ts  = float(tr.get("last_seen", 0))
            img = tr.get("img")
            if img and seen_img_ts.get(img) == ts:
                continue
            if last_seen.get(tid) == ts:
                continue
            key = (img or f"tid:{tid}", ts)
            if (key not in grouped) or (tr.get("bandwidth_mhz", 0) > grouped[key].get("bandwidth_mhz", 0)):
                grouped[key] = tr

        tracks = list(grouped.values())
        predictions = []
        used_imgs = []

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

            # assegna bw_raw PRIMA di usarlo in qualunque euristica
            bw_raw = bw

            if band in ("24","52","58"):
                band_mem.mark(band)
            vts_bw_est = estimate_vts_bw_from_image(img, band, args.dets_tail_kb, ts_center=ts, max_lines=args.dets_tail_lines)
            hint_used = False
            if vts_bw_est and vts_bw_est > bw_raw * 1.5:
                bw_used = vts_bw_est
                hint_used = True
            else:
                bw_used = bw_raw

            # ora possiamo valutare il candidato Wi-Fi 80 usando bw_raw
            wifi80_candidate = is_wifi80_band24(band, bw_raw, hop)

            # --- GATING dal last_trigger.json (UNICO BLOCCO, migliorato con IoU) ---
            gate = {"active": False}
            gate_info, gate_age = read_last_trigger(args.gate_sec)
            if gate_info and (gate_info.get("band") == band):
                dt = float(gate_age if gate_age is not None else 999)
                try:
                    f0 = float(gate_info.get("f0_mhz", f))
                    bw0 = float(gate_info.get("bw_hz", gate_info.get("bw_mhz", 0.0))) / (1e6 if "bw_hz" in gate_info else 1.0)
                except Exception:
                    f0 = f; bw0 = bw
                if dt <= args.gate_sec and bw0 > 0:
                    trig_lo = f0 - bw0/2.0
                    trig_hi = f0 + bw0/2.0
                    clip_limit = max(bw0 * 1.4, bw0 + 6.0)
                    bw_for_iou = min(bw_used, clip_limit)
                    trk_lo = f - bw_for_iou/2.0
                    trk_hi = f + bw_for_iou/2.0
                    center_tol = max(args.gate_center_mhz, bw0 * GATE_CENTER_MULT)
                    delta_f = abs(f - f0)
                    iou = iou_1d(trk_lo, trk_hi, trig_lo, trig_hi)
 
                    # Regole anti-falso per il gating
                    reject_reason = None
                    exp_ratio = (bw_raw / clip_limit) if clip_limit > 0 else 999.0
                    if (wifi80_candidate and not args.gate_allow_wifi80 and band == "24"):
                        reject_reason = "wifi80"
                    elif exp_ratio > args.gate_max_expansion:
                        reject_reason = f"expansion {exp_ratio:.2f}x>{args.gate_max_expansion:.2f}"

                    if (reject_reason is None) and (delta_f <= center_tol) and (iou >= GATE_IOU_MIN):
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
                            f"f0={f0:.3f} bw0={bw0:.2f} clip<={clip_limit:.2f} bw_raw={bw_raw:.2f} → used={bw_used:.2f}")
                    else:
                        why = f"reason={reject_reason}" if reject_reason else "thresholds"
                        log(f"🟡 Gate miss T{tid} [{band}] dt={dt:.2f}s (≤{args.gate_sec}?) "
                            f"Δf={delta_f:.2f} (≤{center_tol:.2f}?) IoU={iou:.2f} (≥{GATE_IOU_MIN}?) "
                            f"bw0={bw0:.2f} bw={bw_used:.2f} (bw_for_iou={bw_for_iou:.2f} clip_limit={clip_limit:.2f} exp={exp_ratio:.2f}x) {why}")
  
            # Euristica Wi‑Fi 80 MHz su 2.4 (usa bw_raw pre-gate per non mascherare)
            soft_gate = False
            gi, ga = read_last_trigger(args.gate_sec)
            if gi and gi.get("band")==band:
                try:
                    f0 = float(gi.get("f0_mhz", f))
                    center_tol = max(args.gate_center_mhz, bw0 * GATE_CENTER_MULT) if 'bw0' in locals() else args.gate_center_mhz
                    soft_gate = (ga is not None and ga <= args.gate_sec and abs(f - f0) <= 1.5*center_tol)
                except Exception:
                    pass
            # Se è Wi-Fi80 candidato, non usare soft_gate per bloccare l'euristica Wi-Fi
            if wifi80_candidate and not args.gate_allow_wifi80 and band == "24" and soft_gate:
                log("↪︎ soft_gate disattivato per Wi-Fi80 candidate")
                soft_gate = False
            # --- SQUELCH POST‑TRIGGER (silenzia non‑gated entro finestra) ---
            try:
                squelch_win = args.squelch_after_trigger_s if args.squelch_after_trigger_s is not None else args.gate_sec
                # Rileggi il last_trigger con finestra = squelch_win (non con gate_sec)
                gi2, ga2 = read_last_trigger(squelch_win)
                if gi2 and gi2.get("band")==band and ga2 is not None and ga2 <= squelch_win and not (gate.get("active", False) or soft_gate):
                    # Caso specifico 2.4: panettoni larghi (Wi‑Fi‑like) → silenzia
                    if band == "24" and bw_raw >= 45.0:
                        log(f"🔇 Squelch post-trigger: T{tid} [{band}] f={f:.3f}MHz bw_raw={bw_raw:.2f} IoU miss → skip")
                        last_seen[tid] = ts
                        if img: used_imgs.append((img, ts))
                        continue
                    # Caso generale: entro finestra post‑trigger, non gated → silenzia
                    log(f"🔇 Squelch post-trigger (general): T{tid} [{band}] non‑gated entro {squelch_win:.1f}s → skip")
                    last_seen[tid] = ts
                    if img: used_imgs.append((img, ts))
                    continue
            except Exception:
                pass

            if wifi80_candidate and not (gate.get("active", False) or soft_gate):

                pred = {
                    "ts": ts,
                    "track_id": tid,
                    "band": band,
                    "img": img,
                    "center_freq_mhz": round(f,3),
                    "bandwidth_mhz": round(bw,3),
                    "bw_used_mhz": round(bw_used,3),
                    "hop_rate_mhz_s": round(hop,3),
                    "label": "Wi‑Fi (80 MHz)",
                    "family": "WIFI",
                    "score": 0.99,
                    "source": "heuristic",
                    "capability": {"scores": {}, "bw_p95": None, "bw_max": None, "bands_seen": sorted(list(band_mem.get_seen()))},
                    "multi_peaks": {},
                    "vts_bw_est_mhz": round(vts_bw_est,2) if vts_bw_est else None,
                    "gate": gate,
                    "hint_used": bool(hint_used),
                    "badges": ["wifi80","non-drone"]
                }
                with OUT_JSONL.open("a") as g:
                    g.write(json.dumps(pred, ensure_ascii=False) + "\n")
                log(f"🚫 Wi‑Fi80 heuristic: T{tid} [24] f={f:.3f}MHz bw_raw={bw_raw:.2f} hop={hop:.2f} → Wi‑Fi 80 MHz")
                predictions.append(pred)
                last_seen[tid] = ts
                if img:
                    used_imgs.append((img, ts))
                continue

            bw_hist[tid].append(float(bw_used))
            seen = band_mem.get_seen()

            cap_out = cap.classify(list(bw_hist[tid]), seen)
            cap_family = cap_out["family"]
            cap_scores = cap_out["scores"]
            bw_p95 = cap_out["bw_p95"]
            bw_max = cap_out["bw_max"]

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

            label = _label_from_family(cap_family)
            score = 1.0 / (1.0 + max(1e-6, cap_scores.get(cap_family, 0.0)))
            if mp and bw_p95 >= 28.0 and mp.get("n_peaks", 0) >= 4 and cap_family == "O4":
                score = min(0.99, score + 0.10)

            src = "capability+peaks" if mp else "capability"

            pred = {
                "ts": ts,
                "track_id": tid,
                "band": band,
                "img": img,
                "center_freq_mhz": round(f,3),
                "bandwidth_mhz": round(bw,3),
                "bw_used_mhz": round(bw_used,3),
                "hop_rate_mhz_s": round(hop,3),
                "label": label,
                "family": cap_family,
                "score": round(float(score), 3),
                "source": src,
                "capability": {
                    "scores": {k: round(v,3) for k,v in cap_scores.items()},
                    "bw_p95": round(bw_p95,2),
                    "bw_max": round(bw_max,2),
                    "bands_seen": sorted(list(seen)),
                },
                "multi_peaks": (mp or {}),
                "vts_bw_est_mhz": round(vts_bw_est,2) if vts_bw_est else None,
                "gate": gate,
                "hint_used": bool(hint_used),
            }

            with OUT_JSONL.open("a") as g:
                g.write(json.dumps(pred, ensure_ascii=False) + "\\n")

            # calcola Δf se poss.
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
