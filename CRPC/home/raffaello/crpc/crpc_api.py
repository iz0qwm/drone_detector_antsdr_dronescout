#!/usr/bin/env python3
# CRPC API + Dashboard serving (v3, with integrated Waterfall)
# - /                -> serves dashboard.html
# - /dashboard       -> same as above
# - /api/status      -> system + pipeline status, band breakdown, latest tiles, eval (if GT provided)
# - /api/file/<name> -> serve files from tiles/tiles_done/log_dir
# - /api/detections  -> UI-friendly recent detections (multi-card)
# - /waterfall       -> simple HTML page with live waterfall (IQ→STFT)
# - /waterfall.png   -> current waterfall frame (PNG)
#
# Env:
#   CRPC_USER=raffaello
#   CRPC_LOG_DIR=/tmp/crpc_logs
#   CRPC_TILES_DIR=/tmp/tiles
#   CRPC_TILES_DONE_DIR=/tmp/tiles_done
#   CRPC_SWEEP24=/tmp/hackrf_sweeps_text/sweep_2400_2500.txt
#   CRPC_SWEEP58=/tmp/hackrf_sweeps_text/sweep_5725_5875.txt
#   CRPC_GT_CSV=/home/raffaello/crpc/ground_truth.csv   (opzionale, per Eval)
#   CRPC_EVAL_TAIL=6000                                 (quante righe di rfscan.jsonl considerare)
#   CRPC_FIFO=/tmp/hackrf.iq                            (FIFO IQ per la waterfall)
#
# Run:
#   apt-get install -y python3-flask python3-psutil python3-matplotlib python3-numpy
#   python3 crpc_api.py --host 0.0.0.0 --port 8080

import os, json, time, subprocess, io, threading
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request, Response, send_file, make_response
import psutil
from datetime import datetime

# --- in cima al file crpc_api.py ---
import csv, statistics as stats

# --- Optional deps for Waterfall ---
# We import numpy/matplotlib unconditionally because the user requested integrated waterfall.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config ----
USER = os.environ.get("CRPC_USER", "raffaello")
CRPC_DIR = Path(f"/home/{USER}/crpc")
BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = Path(os.environ.get("CRPC_LOG_DIR", "/tmp/crpc_logs"))
TILES = Path(os.environ.get("CRPC_TILES_DIR", "/tmp/tiles"))
TILES_DONE = Path(os.environ.get("CRPC_TILES_DONE_DIR", "/tmp/tiles_done"))
SWEEP24 = Path(os.environ.get("CRPC_SWEEP24", "/tmp/hackrf_sweeps_text/sweep_2400_2500.txt"))
SWEEP58 = Path(os.environ.get("CRPC_SWEEP58", "/tmp/hackrf_sweeps_text/sweep_5725_5875.txt"))
SWEEP52 = Path(os.environ.get("CRPC_SWEEP52", "/tmp/hackrf_sweeps_text/sweep_5170_5250.txt"))
GT_CSV = Path(os.environ.get("CRPC_GT_CSV", "/home/raffaello/crpc/ground_truth.csv"))
EVAL_TAIL = int(os.environ.get("CRPC_EVAL_TAIL", "6000"))
FIFO = Path(os.environ.get("CRPC_FIFO", "/tmp/hackrf_live.iq"))  # for Waterfall

DET = LOG_DIR / "detections.jsonl"
TRACKS_CURR = LOG_DIR / "tracks_current.json"
RFSCAN_CURR = LOG_DIR / "rfscan_current.json"
RFSCAN_JL = LOG_DIR / "rfscan.jsonl"
ASSOC = LOG_DIR / "associations.jsonl"   # optional legacy
ASSOC_LOG = LOG_DIR / "assoc.log"
# --- new: RFExplorer sweep JSONL ---
RFE_SWEEP_JL = LOG_DIR / "rfe_sweep.jsonl"

CAND_24 = ["/tmp/rfe/scan/latest_24.csv", "/tmp/rfe/scan/last_24.csv"]
CAND_58 = ["/tmp/rfe/scan/latest_58.csv", "/tmp/rfe/scan/last_58.csv"]
CAND_52 = ["/tmp/rfe/scan/latest_52.csv", "/tmp/rfe/scan/last_52.csv"]

DISABLE_EVAL = os.environ.get("CRPC_DISABLE_EVAL", "0") == "1"
DISABLE_JOURNAL = os.environ.get("CRPC_DISABLE_JOURNAL", "0") == "1"

SERVICES = ["crpc-tiles","crpc-tracker","crpc-yolo","crpc-rfscan","crpc-api","crpc-waterfall","rfe-dual-scan","rfe-csv-bridge","rfe-trigger","hackrf-controller"]

app = Flask(__name__, static_folder=str(BASE_DIR))

# piccola cache in-process (TTL 3s)
_CACHE = {}
def cache_get(key, ttl=3.0):
    now = time.time()
    v = _CACHE.get(key)
    if v and (now - v[0]) < ttl:
        return v[1]
    return None
def cache_set(key, val):
    _CACHE[key] = (time.time(), val)


def tail_lines(path: Path, n=20):
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read_size = min(block, size)
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
            lines = data.splitlines()[-n:]
            return [l.decode("utf-8","ignore") for l in lines]
    except Exception:
        return []

def tail_json(path: Path, max_lines=2000):
    out = []
    for line in tail_lines(path, max_lines):
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

def systemctl_is_active(name):
    try:
        out = subprocess.run(
            ["/bin/systemctl","--system","is-active",name],
            capture_output=True, text=True, timeout=0.7
        )
        s = (out.stdout or "").strip()
        return s if s else "unknown"
    except Exception:
        return "unknown"


def journal_warn_tail(units, n=50):
    if DISABLE_JOURNAL:
        return []
    cached = cache_get(("journal", tuple(units), n))
    if cached is not None:
        return cached
    try:
        cmd = ["journalctl","--no-pager","-n",str(n)]
        for u in units:
            cmd.extend(["-u", u])
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        lines = out.stdout.splitlines()
        import re
        rx = re.compile(r"(error|fail|no space|oom|timeout|cannot|traceback)", re.I)
        res = [ln for ln in lines if rx.search(ln)]
        cache_set(("journal", tuple(units), n), res)
        return res
    except Exception:
        return []

def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default

def count_png(p: Path):
    try:
        return len([x for x in p.glob("*.png")])
    except Exception:
        return 0

def latest_files(p: Path, n=3):
    try:
        files = sorted(p.glob("*.png"), key=lambda q: q.stat().st_mtime, reverse=True)[:n]
        return [x.name for x in files]
    except Exception:
        return []

def bytes_h(n):
    if n is None: return None
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def latest_files_info(p: Path, n=3):
    """Ritorna una lista di dict con name,url,ts_unix per gli ultimi PNG."""
    out = []
    try:
        files = sorted(p.glob("*.png"), key=lambda q: q.stat().st_mtime, reverse=True)[:n]
        for q in files:
            out.append({
                "name": q.name,
                "url": f"/api/file/{q.name}",
                "ts_unix": q.stat().st_mtime
            })
    except Exception:
        pass
    return out

# ---- Evaluation (live) ----
def family_from_label(lbl: str):
    if not lbl: return None
    s = str(lbl).lower()
    if "dji" in s or "mavic" in s or "mini " in s or "air " in s: return "DJI"
    if "frsky" in s or "fr sky" in s or "taranis" in s or "accst" in s or "access" in s: return "FrSky"
    if "flysky" in s or "fly sky" in s or "fs-ia" in s or "fs i6" in s: return "FlySky"
    if "autel" in s or "evo" in s: return "Autel"
    if "parrot" in s or "anafi" in s: return "Parrot"
    if "fimi" in s or "xiaomi" in s: return "FIMI"
    if "elrs" in s or "expresslrs" in s or "radiomaster" in s: return "ELRS"
    if "tbs" in s or "crossfire" in s or "tracer" in s: return "TBS"
    return lbl

def load_ground_truth(gt_csv: Path):
    if not gt_csv.exists():
        return None
    rows = gt_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    import csv
    from io import StringIO
    r = csv.DictReader(StringIO("\n".join(rows)))
    hdr = [h.strip() for h in (r.fieldnames or [])]
    items = [row for row in r]
    if {"track_id","true_label"}.issubset(hdr) or {"track_id","true_family"}.issubset(hdr):
        gt_by_tid = {}
        for row in items:
            try:
                tid = int(row["track_id"])
            except Exception:
                continue
            tl  = (row.get("true_label") or "").strip() or None
            tf  = (row.get("true_family") or "").strip() or None
            if tf is None and tl is not None:
                tf = family_from_label(tl)
            gt_by_tid[tid] = {"true_label": tl, "true_family": tf}
        return {"mode":"per_track","gt_by_tid": gt_by_tid}
    if {"ts_start","ts_end"}.issubset(hdr):
        intervals = []
        for row in items:
            try:
                t0 = float(row["ts_start"]); t1 = float(row["ts_end"])
            except Exception:
                continue
            tl = (row.get("true_label") or "").strip() or None
            tf = (row.get("true_family") or "").strip() or None
            if tf is None and tl is not None:
                tf = family_from_label(tl)
            intervals.append({"t0": t0, "t1": t1, "true_label": tl, "true_family": tf})
        return {"mode":"per_window","intervals": intervals}
    return None

def evaluate_live(preds, gt, domain="family"):
    if not gt:
        return {"available": False}
    y_true, y_pred = [], []
    if gt["mode"] == "per_track":
        gmap = gt["gt_by_tid"]
        for d in preds:
            tid = int(d.get("track_id", -1))
            if tid in gmap:
                if domain == "family":
                    y_true.append(gmap[tid]["true_family"] or "UNK")
                    y_pred.append(d.get("family") or d.get("label") or "UNK")
                else:
                    y_true.append(gmap[tid]["true_label"] or "UNK")
                    y_pred.append(d.get("label") or "UNK")
    else:
        intervals = gt["intervals"]
        for d in preds:
            ts = float(d.get("ts", 0.0))
            matched = [w for w in intervals if (w["t0"] <= ts <= w["t1"])]
            if not matched: continue
            w = matched[0]
            if domain == "family":
                y_true.append(w["true_family"] or "UNK")
                y_pred.append(d.get("family") or d.get("label") or "UNK")
            else:
                y_true.append(w["true_label"] or "UNK")
                y_pred.append(d.get("label") or "UNK")
    labels = sorted(set(y_true) | set(y_pred))
    cm = {a:{b:0 for b in labels} for a in labels}
    for t,p in zip(y_true,y_pred):
        cm[t][p] += 1
    total = len(y_true)
    correct = sum(cm[l].get(l,0) for l in labels)
    acc = (correct/total) if total else 0.0
    prec = {}; rec = {}; support = {}
    for l in labels:
        tp = cm[l].get(l,0)
        fp = sum(cm[x].get(l,0) for x in labels if x!=l)
        fn = sum(cm[l].get(x,0) for x in labels if x!=l)
        prec[l] = tp/ (tp+fp) if (tp+fp)>0 else 0.0
        rec[l]  = tp/ (tp+fn) if (tp+fn)>0 else 0.0
        support[l] = sum(cm[l].values())
    return {
        "available": True,
        "domain": domain,
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": acc,
        "n_samples": total,
        "precision": prec,
        "recall": rec,
        "support": support,
    }

# ---- Routes ----
@app.route("/")
@app.route("/dashboard")
@app.route("/dashboard.html")
def serve_dashboard():
    # serve the bundled dashboard.html from the same folder
    f = BASE_DIR / "dashboard.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return Response("<h1>Dashboard not found</h1>", mimetype="text/html")

@app.route("/api/status")
def api_status():
    t0 = time.time()

    # services (cache 3s)
    svc_cached = cache_get(("services", tuple(SERVICES)))
    if svc_cached is None:
        svc_cached = {s: systemctl_is_active(s) for s in SERVICES}
        cache_set(("services", tuple(SERVICES)), svc_cached)
    services = svc_cached

    # disk /tmp
    tmp_usage = None
    try:
        du = psutil.disk_usage("/tmp")
        tmp_usage = {"total": du.total, "used": du.used, "free": du.free, "percent": du.percent}
    except Exception:
        pass

    # tiles
    tiles = {
        "queue": count_png(TILES),
        "done": count_png(TILES_DONE),
        "latest_queue": latest_files(TILES, 3),
        "latest_done": latest_files(TILES_DONE, 3)
    }

    # tails & logs
    sweep24_tail = tail_lines(SWEEP24, n=3)
    sweep58_tail = tail_lines(SWEEP58, n=3)
    sweep52_tail = tail_lines(SWEEP52, n=3)
    det_tail = tail_lines(DET, n=5)
    assoc_tail = tail_lines(ASSOC, n=5)
    assoc_log_tail = tail_lines(ASSOC_LOG, n=10)
    warnings_tail = journal_warn_tail(SERVICES, n=80)

    # current JSONs
    tracks = read_json(TRACKS_CURR, [])
    rfscan_cur = read_json(RFSCAN_CURR, [])

    # band breakdown from rfscan_current
    by_band = {"24": {}, "58": {}, "52": {}, "UNK": {}}
    for r in rfscan_cur:
        b = r.get("band") or "UNK"
        fam = r.get("family") or "UNK"
        by_band.setdefault(b, {})
        by_band[b][fam] = by_band[b].get(fam, 0) + 1

    # live eval from tail of rfscan.jsonl (respect flags + cache on mtime)
    eval_obj = {"available": False}
    try:
        if (not DISABLE_EVAL) and GT_CSV.exists() and RFSCAN_JL.exists():
            key = ("eval", RFSCAN_JL.stat().st_mtime)
            ev_cached = cache_get(key)
            if ev_cached is None:
                tail_preds = tail_json(RFSCAN_JL, max_lines=EVAL_TAIL)
                gt = load_ground_truth(GT_CSV)
                ev_cached = evaluate_live(tail_preds, gt, domain="family")
                cache_set(key, ev_cached)
            eval_obj = ev_cached
    except Exception:
        # se qualcosa va storto nell'eval, non bloccare lo status
        eval_obj = {"available": False, "error": "eval_failed"}

    # preview images (absolute URLs via /api/file)
    previews = {
        "queue": tiles["latest_queue"],          # compatibilità retro
        "done": tiles["latest_done"],            # compatibilità retro
        "queue_info": latest_files_info(TILES, 3),
        "done_info": latest_files_info(TILES_DONE, 3),
    }


    resp = {
        "ts": time.time(),
        "services": services,
        "tmp_usage": tmp_usage,
        "tiles": tiles,
        "sweeps": {"24": sweep24_tail, "58": sweep58_tail, "52": sweep52_tail},
        "logs": {
            "detections_tail": det_tail,
            "associations_tail": assoc_tail,
            "assoc_log_tail": assoc_log_tail,
            "warnings_tail": warnings_tail
        },
        "tracks_current": tracks,
        "rfscan_current": rfscan_cur,
        "breakdown_by_band": by_band,
        "eval": eval_obj,
        "previews": previews
    }

    app.logger.info(
        "status in %.1f ms (eval:%s journal:%s)",
        (time.time() - t0) * 1000,
        not DISABLE_EVAL,
        not DISABLE_JOURNAL
    )
    return jsonify(resp)

@app.route("/api/file/<path:fname>")
def api_file(fname):
    for base in (TILES, TILES_DONE, LOG_DIR):
        p = base / fname
        if p.exists():
            return send_from_directory(str(base), fname)
    return ("Not found", 404)


# --- Detections helpers & route ---
def _latest_tile_info():
    try:
        f = sorted(TILES_DONE.glob("*.png"), key=lambda q: q.stat().st_mtime, reverse=True)
        if not f: return (None, None)
        top = f[0]
        return (top.name, top.stat().st_mtime)
    except Exception:
        return (None, None)

def _tile_info_from_name(name: str):
    if not name: return (None, None)
    p = TILES_DONE / name
    if not p.exists():  # prova anche nella coda
        p = TILES / name
        if not p.exists():
            return (name, None)
    try:
        return (name, p.stat().st_mtime)
    except Exception:
        return (name, None)

def _iso_utc(ts: float):
    try:
        import datetime as dt
        return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

def _format_signal_badges(r: dict):
    badges = []
    for k in ("signal_type", "kind", "mode"):
        v = (r.get(k) or "").strip()
        if v:
            badges.extend([x.strip() for x in v.replace(",", " ").split() if x.strip()])
    if isinstance(r.get("tags"), list):
        badges.extend([str(x).strip() for x in r["tags"] if str(x).strip()])
    seen = set(); out = []
    for b in badges:
        key = b.lower()
        if key not in seen:
            seen.add(key)
            out.append(b.upper())
    return out[:4]

@app.route("/api/detections")
def api_detections():
    now = time.time()
    rfscan_cur = read_json(RFSCAN_CURR, [])
    out = []
    for r in rfscan_cur:
        # freq in MHz
        cf = r.get("center_freq_mhz") or r.get("freq") or r.get("frequency")
        try:
            cf = float(cf)
            freq_mhz = round(cf/1e6, 3) if cf > 10000 else round(cf, 3)
        except Exception:
            freq_mhz = None
        bw = r.get("bandwidth_mhz") or r.get("bw") or r.get("bandwidth")
        try:
            bw = round(float(bw), 3) if bw is not None else None
        except Exception:
            pass
        band = r.get("band") or "UNK"
        # immagine + ts
        tile_name = r.get("tile") or r.get("img")
        if tile_name:
            tile_name, tile_mtime = _tile_info_from_name(tile_name)
        else:
            tile_name, tile_mtime = _latest_tile_info()
        tile_url = f"/api/file/{tile_name}" if tile_name else None
        det_ts = None
        for k in ("ts", "timestamp"):
            v = r.get(k)
            if v is not None:
                try:
                    det_ts = float(v); break
                except Exception:
                    pass
        if det_ts is None:
            det_ts = tile_mtime
        out.append({
            "title": "Drone detected",
            "freq_mhz": freq_mhz,
            "band": band,
            "badges": _format_signal_badges(r),
            "ts_unix": det_ts,
            "ts_iso": _iso_utc(det_ts) if det_ts else None,
            "thumbnail": tile_url,
            "tile_name": tile_name,
            "bw_mhz": bw,
            "snr_db": r.get("snr") or r.get("snr_db"),
            "src": r.get("source") or r.get("src") or "rfscan",
            "track_id": r.get("track_id"),
        })
    out.sort(key=lambda x: (x["ts_unix"] is None, -(x["ts_unix"] or 0)))
    return jsonify({"detections": out, "ts": now})

@app.route("/api/uav_status")
def api_uav_status():
    """
    Ritorna se c'è un UAV 'attivo adesso' usando solo tempi lato server.
    Regole:
      - tile *_live.png  → finestra breve (8s)
      - tile *_cum_*     → finestra più ampia (120s)
      - fallback: tracks_current.last_seen <= 8s
    """
    now = time.time()
    LIVE_W = 8
    CUM_W  = 120

    rfscan_cur = read_json(RFSCAN_CURR, [])  # stesso file che usi in /api/detections
    active_bands = set()
    recent_found = False

    for r in rfscan_cur:
        band = r.get("band") or "UNK"

        # prendi tile e timestamp (come fai in /api/detections)
        tile_name = r.get("tile") or r.get("img")
        if tile_name:
            tile_name, tile_mtime = _tile_info_from_name(tile_name)
        else:
            tile_name, tile_mtime = _latest_tile_info()

        det_ts = None
        for k in ("ts", "timestamp"):
            v = r.get(k)
            if v is not None:
                try:
                    det_ts = float(v); break
                except Exception:
                    pass
        if det_ts is None:
            det_ts = tile_mtime  # fallback come in /api/detections

        if det_ts is None:
            continue

        name = str(tile_name or "")
        is_live = name.endswith("_live.png")
        is_cum  = "_cum_" in name
        window  = LIVE_W if is_live else (CUM_W if is_cum else 20)

        if (now - det_ts) <= window:
            recent_found = True
            if band: active_bands.add(str(band))

    # fallback robusto: tracks_current.last_seen (stesso clock server)
    if not recent_found:
        tracks = read_json(TRACKS_CURR, [])
        for t in tracks:
            try:
                if float(t.get("last_seen", 1e9)) <= LIVE_W:
                    recent_found = True
                    b = t.get("band"); 
                    if b: active_bands.add(str(b))
            except Exception:
                pass

    return jsonify({
        "active": bool(recent_found),
        "bands": sorted(active_bands),
        "ts": now,
        "n_rfscan": len(rfscan_cur)
    })

def _ts_to_epoch(v):
    if v is None: return 0.0
    try:
        return float(v)  # già epoch?
    except Exception:
        pass
    try:
        s = str(v).strip().replace("Z","")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0

def _pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _read_csv_spectrum_from(path):
    if not path: return None
    try:
        ts = os.path.getmtime(path)
        freqs, pwr = [], []
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    fm = float(row.get("freq_mhz") or row.get("frequency") or row.get("freq"))
                    db = float(row.get("power_dbm") or row.get("dbm") or row.get("amp_dbm") or row.get("amp"))
                    freqs.append(fm); pwr.append(db)
                except: 
                    continue
        if len(freqs) < 4: 
            return None
        z = sorted(zip(freqs, pwr), key=lambda x: x[0])
        freqs, pwr = list(map(list, zip(*z)))
        # y‑axis stabile
        if len(pwr) >= 100:
            qs = stats.quantiles(pwr, n=100)
            lo, hi = qs[4], qs[95]
        else:
            lo, hi = min(pwr), max(pwr)
        if hi - lo < 5: hi = lo + 5
        yaxis = {"min": float(round(lo - 1, 1)), "max": float(round(hi + 1, 1))}
        return {"freqs_mhz": freqs, "pwr_dbm": pwr, "ts": ts, "yaxis": yaxis, "path": path}
    except:
        return None

def _find_peaks(freqs, pwr, min_db_over_floor=6.0, min_sep_bins=3, top_n=8):
    if not freqs or not pwr: return []
    floor = stats.median(pwr)
    cand = []
    for i in range(1, len(pwr)-1):
        if pwr[i] > pwr[i-1] and pwr[i] > pwr[i+1] and (pwr[i]-floor) >= min_db_over_floor:
            cand.append((i, pwr[i]))
    cand.sort(key=lambda x: x[1], reverse=True)
    picked = []
    for idx, _ in cand:
        if all(abs(idx-j) >= min_sep_bins for j,_ in picked):
            picked.append((idx, pwr[idx]))
        if len(picked) >= top_n: break
    return [{"freq_mhz": float(freqs[i]), "dbm": float(pwr[i])} for i,_ in picked]

@app.route("/api/spectrum")
def api_spectrum():
    res = {"latest": {}, "peaks": []}
    b24 = _pick_existing(CAND_24)
    b58 = _pick_existing(CAND_58)
    b52 = _pick_existing(CAND_52)

    d24 = _read_csv_spectrum_from(b24)
    d58 = _read_csv_spectrum_from(b58)
    d52 = _read_csv_spectrum_from(b52)
    if d24: 
        res["latest"]["24"] = d24
        for p in _find_peaks(d24["freqs_mhz"], d24["pwr_dbm"]):
            p.update(band="24", ts=d24["ts"])
            res["peaks"].append(p)
    if d58:
        res["latest"]["58"] = d58
        for p in _find_peaks(d58["freqs_mhz"], d58["pwr_dbm"]):
            p.update(band="58", ts=d58["ts"])
            res["peaks"].append(p)
    if d52:
        res["latest"]["52"] = d52
        for p in _find_peaks(d52["freqs_mhz"], d52["pwr_dbm"]):
            p.update(band="52", ts=d52["ts"])
            res["peaks"].append(p)        

    res["peaks"].sort(key=lambda p: p.get("ts", 0))
    # no‑cache per il browser
    rsp = make_response(jsonify(res))
    rsp.headers["Cache-Control"] = "no-store"
    return rsp


# --- Waterfall (integrated) ---
# Parameters aligned with your standalone script
NFFT = 4096
HOP  = NFFT // 4
CMAP = "turbo"
FIG_W, FIG_H = 1800, 1200
SPEC_W, SPEC_H = 1395, 920
DPI = 100
DURATION_S = 0.10
FMAX_LABEL = 5e7
T_PIX = 480
F_PIX = 920

_img01 = np.zeros((F_PIX, T_PIX), np.float32)
_wf_started = False
_wf_lock = threading.Lock()

def _wf_iq_reader():
    """Reads IQ int8 from FIFO and updates the global _img01 waterfall image."""
    global _img01
    # wait for FIFO
    while not FIFO.exists():
        time.sleep(0.1)
    # open unbuffered
    f = open(FIFO, "rb", buffering=0)
    win = np.hanning(NFFT).astype(np.float32)
    buf = np.zeros(NFFT, np.complex64)
    filled = 0
    while True:
        need = (NFFT - filled) * 2
        raw = f.read(need)
        if not raw:
            time.sleep(0.001); continue
        iq = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        if iq.size % 2 == 1:
            iq = iq[:-1]
        I = iq[0::2]; Q = iq[1::2]
        c = (I + 1j*Q) / 128.0
        take = min(c.size, NFFT - filled)
        buf[filled:filled+take] = c[:take]
        filled += take
        c = c[take:]
        if filled < NFFT:
            continue

        x = buf * win
        spec = np.fft.fft(x, n=NFFT)
        psd = np.abs(spec)**2
        half = psd[:NFFT//2]
        eps = 1e-12
        db = 10*np.log10(np.maximum(half, eps))
        db = db - np.median(db)
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
        with _wf_lock:
            _img01 = np.hstack([_img01[:, 1:], col01.reshape(F_PIX, 1)])
        if HOP < NFFT:
            buf[:-HOP] = buf[HOP:]
            filled = NFFT - HOP
        else:
            filled = 0

def _wf_render_png():
    with _wf_lock:
        m = _img01.copy()
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
    from matplotlib.ticker import FuncFormatter
    ax.set_yticks(yt); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(round(y/1e7))}"))
    ax.text(0.0, FMAX_LABEL*1.015, "1e7", ha="left", va="bottom", fontsize=14)
    ax.set_title("Sweep", fontsize=16, pad=8)
    fig.savefig(bio, format="png", facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    bio.seek(0)
    return bio

def _ensure_wf_thread():
    global _wf_started
    if not _wf_started:
        threading.Thread(target=_wf_iq_reader, daemon=True).start()
        _wf_started = True

@app.get("/waterfall")
def wf_page():
    _ensure_wf_thread()
    return """<!doctype html><title>HackRF Waterfall</title>
<style>body{background:#111;color:#eee;font-family:sans-serif} img{width:96vw;max-width:1800px;border:2px solid #333;border-radius:8px}</style>
<h2>HackRF Waterfall (IQ → STFT)</h2>
<div class="muted" style="color:#8b949e">Fonte FIFO: %s</div>
<img id="wf" src="/waterfall.png">
<script>setInterval(()=>{const i=document.getElementById('wf'); i.src='/waterfall.png?t='+Date.now()}, 200);</script>
""" % (str(FIFO),)

@app.get("/waterfall.png")
def wf_png():
    _ensure_wf_thread()
    return send_file(_wf_render_png(), mimetype="image/png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
