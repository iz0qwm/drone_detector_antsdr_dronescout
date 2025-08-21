#!/usr/bin/env python3
# RF scan classifier — versione capability-first (senza fingerprint DB)
# Adatto alla struttura CRPC con:
#  - tracks_current.json in /tmp/crpc_logs
#  - detections.jsonl per hint video in /tmp/crpc_logs
#  - latest_24/52/58.csv da RF Explorer in /tmp/rfe/scan
#
# Output invariati:
#  - rfscan.jsonl e rfscan_current.json in /tmp/crpc_logs
#  - log in /tmp/crpc_logs/assoc.log

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

BANDS = {
    "24": (2400.0, 2500.0),
    "58": (5725.0, 5875.0),
    "52": (5170.0, 5250.0),
}

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > args.log_trim_mb * 1024 * 1024:
            try:
                tail = LOG_PATH.read_text(errors="ignore").splitlines()[-500:]
                LOG_PATH.write_text("\\n".join(tail) + "\\n")
            except Exception:
                pass
        with LOG_PATH.open("a") as f:
            f.write(f"[{now_str()}] {msg}\\n")
    except Exception:
        pass
    print(msg, flush=True)

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
    p = argparse.ArgumentParser(description="RF scan classifier (capability-first)")
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
    return p.parse_args()

args = parse_args()

def _label_from_family(fam):
    if fam == "O4":     return "DJI (O4)"
    if fam == "O23":    return "DJI (O2/O3)"
    if fam == "HUBSAN": return "Hubsan-like"
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

            if band in ("24","52","58"):
                band_mem.mark(band)

            vts_bw_est = estimate_vts_bw_from_image(img, band, args.dets_tail_kb, ts_center=ts, max_lines=args.dets_tail_lines)
            if vts_bw_est and vts_bw_est > bw * 1.5:
                bw_used = vts_bw_est
            else:
                bw_used = bw

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
            }

            with OUT_JSONL.open("a") as g:
                g.write(json.dumps(pred, ensure_ascii=False) + "\\n")

            peaks_msg = f" peaks={mp.get('n_peaks')} pk={mp.get('peakiness'):.1f}" if mp else ""
            log(f"✅ RFscan T{tid} [{band}] {f:.3f}MHz bw={bw:.3f} (used={bw_used:.3f}) → {label} ({src} {pred['score']:.2f})"
                f" — seen={sorted(list(seen))} bw_p95={bw_p95:.1f} bw_max={bw_max:.1f}{peaks_msg}")

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
