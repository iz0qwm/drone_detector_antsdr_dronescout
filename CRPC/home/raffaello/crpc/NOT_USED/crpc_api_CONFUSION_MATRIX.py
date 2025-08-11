#!/usr/bin/env python3
# CRPC API + Dashboard serving (v2)
# - /          -> serves dashboard.html
# - /dashboard -> same as above
# - /api/status: system + pipeline status, band breakdown, latest tiles, eval (if GT provided)
# - /api/file/<name>: serve files from tiles/tiles_done/log_dir
#
# Env:
#   CRPC_USER=raffaello
#   CRPC_LOG_DIR=/tmp/crpc_logs
#   CRPC_TILES_DIR=/tmp/tiles
#   CRPC_TILES_DONE_DIR=/tmp/tiles_done
#   CRPC_SWEEP24=/tmp/hackrf_sweeps_text/sweep_2400_2500.txt
#   CRPC_SWEEP58=/tmp/hackrf_sweeps_text/sweep_5725_5875.txt
#   CRPC_GT_CSV=/home/raffaello/crpc/ground_truth.csv  (opzionale, per Eval)
#   CRPC_EVAL_TAIL=6000  (quante righe recenti di rfscan.jsonl considerare)
#
# Run:
#   apt-get install -y python3-flask python3-psutil
#   python3 crpc_api.py --host 0.0.0.0 --port 8080

import os, json, time, subprocess
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request, Response
import psutil

# ---- Config ----
USER = os.environ.get("CRPC_USER", "raffaello")
CRPC_DIR = Path(f"/home/{USER}/crpc")
BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = Path(os.environ.get("CRPC_LOG_DIR", "/tmp/crpc_logs"))
TILES = Path(os.environ.get("CRPC_TILES_DIR", "/tmp/tiles"))
TILES_DONE = Path(os.environ.get("CRPC_TILES_DONE_DIR", "/tmp/tiles_done"))
SWEEP24 = Path(os.environ.get("CRPC_SWEEP24", "/tmp/hackrf_sweeps_text/sweep_2400_2500.txt"))
SWEEP58 = Path(os.environ.get("CRPC_SWEEP58", "/tmp/hackrf_sweeps_text/sweep_5725_5875.txt"))
GT_CSV = Path(os.environ.get("CRPC_GT_CSV", "/home/raffaello/crpc/ground_truth.csv"))
EVAL_TAIL = int(os.environ.get("CRPC_EVAL_TAIL", "6000"))

DET = LOG_DIR / "detections.jsonl"
TRACKS_CURR = LOG_DIR / "tracks_current.json"
RFSCAN_CURR = LOG_DIR / "rfscan_current.json"
RFSCAN_JL = LOG_DIR / "rfscan.jsonl"
ASSOC = LOG_DIR / "associations.jsonl"   # optional legacy
ASSOC_LOG = LOG_DIR / "assoc.log"

DISABLE_EVAL = os.environ.get("CRPC_DISABLE_EVAL", "0") == "1"
DISABLE_JOURNAL = os.environ.get("CRPC_DISABLE_JOURNAL", "0") == "1"

SERVICES = ["crpc-api","crpc-prepare","crpc-sweep","crpc-tiles","crpc-yolo","crpc-tracker","crpc-rfscan"]

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
            while size > 0 and data.count(b"\\n") <= n:
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
        return json.loads(path.read_text())
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
    r = csv.DictReader(StringIO("\\n".join(rows)))
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
    det_tail = tail_lines(DET, n=5)
    assoc_tail = tail_lines(ASSOC, n=5)
    assoc_log_tail = tail_lines(ASSOC_LOG, n=10)
    warnings_tail = journal_warn_tail(SERVICES, n=80)

    # current JSONs
    tracks = read_json(TRACKS_CURR, [])
    rfscan_cur = read_json(RFSCAN_CURR, [])

    # band breakdown from rfscan_current
    by_band = {"24": {}, "58": {}, "UNK": {}}
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
        "queue": tiles["latest_queue"],
        "done": tiles["latest_done"]
    }

    resp = {
        "ts": time.time(),
        "services": services,
        "tmp_usage": tmp_usage,
        "tiles": tiles,
        "sweeps": {"24": sweep24_tail, "58": sweep58_tail},
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

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)