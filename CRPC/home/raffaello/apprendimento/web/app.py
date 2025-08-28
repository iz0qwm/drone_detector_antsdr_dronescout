#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
import subprocess, json, pathlib, os
from werkzeug.exceptions import HTTPException
from datetime import datetime

# --- ADD: import per spettro ---
import csv, statistics as stats

# --- ADD: candidati CSV RFE (24/58/52) ---
CAND_24 = ["/tmp/rfe/scan/latest_24.csv", "/tmp/rfe/scan/last_24.csv"]
CAND_58 = ["/tmp/rfe/scan/latest_58.csv", "/tmp/rfe/scan/last_58.csv"]
CAND_52 = ["/tmp/rfe/scan/latest_52.csv", "/tmp/rfe/scan/last_52.csv"]

APP_ROOT = pathlib.Path("/home/raffaello/apprendimento")
BIN = APP_ROOT / "bin"
DATA = APP_ROOT / "data"
LOCK = pathlib.Path("/tmp/apprendimento.lock")  # <— PRIMA ERA /run, dava PermissionError
PY = "/usr/bin/python3"

app = Flask(__name__)

def resolve_session_dir(s: str) -> pathlib.Path | None:
    p = pathlib.Path(s)
    if p.is_absolute():
        return p if p.exists() else None
    # cerca sotto data/recordings/*/*/session_...
    cand = list((DATA/"recordings").rglob(s))
    if not cand:
        return None
    # se più risultati, piglia il più recente
    return max(cand, key=lambda x: x.stat().st_mtime)

def run_cmd(cmd):
    """Esegue e restituisce dict con rc/stdout/stderr senza alzare eccezioni."""
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
    except Exception as e:
        return {"rc": 127, "out": "", "err": f"{type(e).__name__}: {e}"}

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
        # y-axis stabile
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


@app.errorhandler(Exception)
def handle_any_error(e):
    if isinstance(e, HTTPException):
        return jsonify(ok=False, error=e.name, code=e.code, detail=str(e)), e.code
    return jsonify(ok=False, error="InternalServerError", code=500, detail=str(e)), 500

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def api_health():
    return jsonify(ok=True, service="learn-api", lock=str(LOCK), lock_exists=LOCK.exists())

@app.post("/api/stop_live")
def stop_live():
    res = run_cmd([str(BIN/"stop_live.sh")])
    try:
        LOCK.write_text("1")
    except Exception as e:
        res["err"] += f"\nlock_err: {e}"
    # RESTITUISCO SEMPRE JSON, anche se rc!=0
    return jsonify(ok=(res["rc"]==0), **res)

@app.post("/api/start_live")
def start_live():
    try:
        if LOCK.exists():
            LOCK.unlink()
    except Exception:
        pass
    res = run_cmd([str(BIN/"start_live.sh")])
    return jsonify(ok=(res["rc"]==0), **res)

@app.post("/api/hw_check")
def hw_check():
    hackrf = run_cmd(["hackrf_info"])
    latest = list(pathlib.Path("/tmp/rfe/scan").glob("latest_*.csv"))
    return jsonify(ok=True, hw={
        "hackrf": hackrf["rc"] == 0,
        "rfe": len(latest) > 0,
        "notes": [hackrf["err"]] if hackrf["rc"] != 0 else []
    })

@app.post("/api/record")
def record():
    try:
        j = request.get_json(force=True)
    except Exception:
        return jsonify(ok=False, error="JSON mancante o invalido"), 400

    # default comodi se il form è vuoto
    band = str(j.get("band", "24"))
    center_map = {"24": 2400.0, "52": 5200.0, "58": 5800.0}
    center = float(j.get("center_mhz", center_map.get(band, 2400.0)))
    bw = float(j.get("bw_mhz", 20.0))
    dur = int(j.get("duration_s", 15))
    drone_id = j.get("drone_id", "Unknown")

    cmd = [
        PY, str(BIN/"learn_recorder.py"),
        "--band", band,
        "--drone-id", drone_id,
        "--center-mhz", str(center),
        "--bw-mhz", str(bw),
        "--duration-s", str(dur),
        "--out-root", str(DATA)
    ]

    res = run_cmd(cmd)
    ok = (res["rc"] == 0)
    # se lo script stampa JSON valido su stdout, ritornalo; altrimenti pacchetto standard
    if ok:
        try:
            payload = json.loads(res["out"] or "{}")
            payload.setdefault("ok", True)
            return jsonify(payload)
        except Exception:
            return jsonify(ok=True, out=res["out"], err=res["err"])
    return jsonify(ok=False, out=res["out"], err=res["err"]), 500

@app.post("/api/extract")
def extract():
    j = request.get_json(force=True)
    sess_dir = resolve_session_dir(j["session_dir"])
    if not sess_dir:
        return jsonify(ok=False, error="session_dir non trovata", given=j["session_dir"]), 400

    res = run_cmd([PY, str(BIN/"learn_extractor.py"),
               "--session-dir", str(sess_dir),
               "--label", j["label"]])

    try:
        payload = json.loads(res["out"] or "{}")
    except Exception:
        payload = {"ok": res["rc"]==0, "out": res["out"], "err": res["err"]}
    return jsonify(payload)

@app.post("/api/train")
def train():
    res = run_cmd([PY, str(BIN/"learn_trainer.py")])
    try:
        payload = json.loads(res["out"] or "{}")
    except Exception:
        payload = {"ok": res["rc"]==0, "out": res["out"], "err": res["err"]}
    return jsonify(payload)


@app.post("/api/promote")
def promote():
    res = run_cmd([str(BIN/"promote_model.sh")])
    ok = (res["rc"] == 0)
    return jsonify(ok=ok, out=res["out"], err=res["err"]), (200 if ok else 500)

# Servi i file (tiles, ecc.) in modo sicuro partendo dalla radice APP_ROOT
@app.get("/api/file/<path:relpath>")
def api_file(relpath):
    base = APP_ROOT.resolve()
    target = (base / relpath).resolve()
    if not str(target).startswith(str(base)):
        abort(403)
    if target.is_dir():
        abort(404)
    return send_from_directory(target.parent, target.name)

@app.route("/api/last_session", methods=["GET","POST"])
def last_session():
    base = (DATA/"recordings")
    cand = list(base.rglob("session_*"))
    if not cand:
        return jsonify(ok=False, error="nessuna session"), 404
    last = max(cand, key=lambda p: p.stat().st_mtime)
    return jsonify(ok=True, session=str(last))

@app.post("/api/quick_test")
def quick_test():
    j = request.get_json(force=True)
    # risolve relativo/assoluto
    def resolve_session_dir(s: str):
        p = pathlib.Path(s)
        if p.is_absolute():
            return p if p.exists() else None
        cand = list((DATA/"recordings").rglob(s))
        return max(cand, key=lambda x: x.stat().st_mtime) if cand else None

    sess_dir = resolve_session_dir(j["session_dir"])
    if not sess_dir:
        return jsonify(ok=False, error="session_dir non trovata", given=j["session_dir"]), 400

    use = (j.get("use") or "current").lower()  # "current" | "staging"
    model_path = APP_ROOT / "models" / ( "rfscan_current.pkl" if use=="current" else "rfscan_staging.pkl" )

    res = run_cmd([PY, str(APP_ROOT/"bin/quick_classify.py"),
                "--session-dir", str(sess_dir),
                "--model", str(model_path)])

    # se il comando fallisce o non stampa nulla, torna l'errore (non {} vuoto)
    if res["rc"] != 0 or not (res["out"] or "").strip():
        return jsonify(ok=False, rc=res["rc"], out=res["out"], err=res["err"])

    try:
        payload = json.loads(res["out"])
    except Exception:
        payload = {"ok": False, "out": res["out"], "err": res["err"]}

    return jsonify(payload)

@app.post("/api/rfe/start")
def rfe_start():
    res = run_cmd([str(BIN/"start_rfe.sh")])
    return jsonify(ok=(res["rc"]==0), **res)

@app.post("/api/rfe/stop")
def rfe_stop():
    res = run_cmd([str(BIN/"stop_rfe.sh")])
    return jsonify(ok=(res["rc"]==0), **res)


@app.get("/api/spectrum")
def api_spectrum():
    """
    Restituisce lo spettro corrente. Parametri opzionali:
      - band=24|52|58  (se omesso: tutte)
    """
    want = (request.args.get("band") or "").strip()
    res = {"latest": {}, "peaks": []}

    pairs = []
    if want in ("24",""):
        pairs.append(("24", _pick_existing(CAND_24)))
    if want in ("58",""):
        pairs.append(("58", _pick_existing(CAND_58)))
    if want in ("52",""):
        pairs.append(("52", _pick_existing(CAND_52)))

    for b, path in pairs:
        data = _read_csv_spectrum_from(path)
        if not data: 
            continue
        res["latest"][b] = data
        for p in _find_peaks(data["freqs_mhz"], data["pwr_dbm"]):
            p.update(band=b, ts=data["ts"])
            res["peaks"].append(p)

    res["peaks"].sort(key=lambda p: p.get("ts", 0))
    return jsonify(res)

# === PROD EXT: costanti comode ===
YOLO_DATASET = pathlib.Path("/home/raffaello/dataset/yolo_custom")
YOLO_RUNS    = pathlib.Path("/home/raffaello/yolo_runs")
YOLO_RUNNAME = "yolo_custom"
YOLO_ARCH    = "yolov8n.pt"  # cambia qui se serve
MODELS_DIR   = APP_ROOT / "models"
LOGS_DIR     = APP_ROOT / "logs"

# --- Run dir helpers (alias/symlink) ---
def _find_latest_run_dir():
    """Restituisce la cartella più recente tra yolo_custom, yolo_custom2, yolo_custom3, ..."""
    pattern = f"{YOLO_RUNNAME}*"
    dirs = [p for p in YOLO_RUNS.glob(pattern) if p.is_dir()]
    if not dirs:
        return YOLO_RUNS / YOLO_RUNNAME
    return max(dirs, key=lambda p: p.stat().st_mtime)

def _ensure_run_alias(target: pathlib.Path):
    """
    Garantisce che YOLO_RUNS/YOLO_RUNNAME sia un symlink che punta a 'target'.
    Se esiste una dir reale con quel nome, la rinomina in *_old_YYYYmmdd-HHMMSS.
    """
    link = YOLO_RUNS / YOLO_RUNNAME
    try:
        if link.is_symlink():
            link.unlink()
        elif link.exists() and link.is_dir() and link.resolve() != target.resolve():
            backup = link.with_name(link.name + "_old_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            link.replace(backup)
        if target.exists() and target != link:
            os.symlink(str(target), str(link))
        return True, str(link), str(target)
    except Exception as e:
        return False, str(link), f"{type(e).__name__}: {e}"

def _effective_run_dir():
    """
    Cartella effettiva da usare per pesi/classmap:
    - se esiste il link YOLO_RUNNAME, lo risolve;
    - altrimenti prende l'ultima yolo_custom* e prova a creare il link.
    """
    link = YOLO_RUNS / YOLO_RUNNAME
    if link.exists():
        try:
            return link.resolve()
        except Exception:
            return link  # nel dubbio, usa così com'è
    latest = _find_latest_run_dir()
    _ensure_run_alias(latest)
    return latest

def run_cmd_logged(cmd, log_name):
    """Esegue un comando, salva out/err in logs/<log_name>-YYYYmmdd-HHMMSS.log e ritorna rc/out/err + log_path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"{log_name}-{ts}.log"
    res = run_cmd(cmd)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(map(str, cmd)) + "\n\n")
            f.write("== STDOUT ==\n" + (res["out"] or "") + "\n")
            f.write("== STDERR ==\n" + (res["err"] or "") + "\n")
            f.write(f"== RC == {res['rc']}\n")
    except Exception as e:
        res["err"] += f"\n[log_write_error] {e}"
    res["log_path"] = str(log_path)
    return res

@app.post("/api/make_yolo_dataset")
def api_make_yolo_dataset():
    base = str(DATA / "recordings")
    out  = str(YOLO_DATASET)
    cmd = [PY, str(APP_ROOT/"bin"/"appre_to_prod"/"01_make_yolo_from_recordings.py"), "--base", base, "--out", out, "--stratified"]
    res = run_cmd_logged(cmd, "01_make_yolo")
    ok = (res["rc"] == 0)
    return jsonify(ok=ok, **res), (200 if ok else 500)

@app.post("/api/yolo_train")
def api_yolo_train():
    data_yaml = str(YOLO_DATASET / "data.yaml")
    cmd = [str(APP_ROOT/"bin"/"appre_to_prod"/"02_train_yolo.sh"), data_yaml, str(YOLO_RUNS), YOLO_RUNNAME, YOLO_ARCH]
    res = run_cmd_logged(cmd, "02_train_yolo")

    # NEW: aggiorna/crea alias yolo_custom -> ultimo run reale (es. yolo_custom2)
    try:
        latest = _find_latest_run_dir()
        ok_link, link_path, target_path = _ensure_run_alias(latest)
        res["alias_ok"] = ok_link
        res["alias_link"] = link_path
        res["alias_target"] = target_path
    except Exception as e:
        res["alias_ok"] = False
        res["alias_err"] = f"{type(e).__name__}: {e}"

    ok = (res["rc"] == 0)
    return jsonify(ok=ok, **res), (200 if ok else 500)


@app.post("/api/yolo_classmap")
def api_yolo_classmap():
    #data_yaml = str(YOLO_DATASET / "data.yaml")
    run_dir = _effective_run_dir()  # NEW
    #out_json  = str(run_dir / "classmap.json")
    cmd = [PY, str(APP_ROOT/"bin"/"appre_to_prod"/"02b_make_classmap.py")]
    res = run_cmd_logged(cmd, "02b_classmap")
    ok = (res["rc"] == 0)
    return jsonify(ok=ok, run_dir=str(run_dir), **res), (200 if ok else 500)

@app.post("/api/yolo_smoketest")
def api_yolo_smoketest():
    run_dir = _effective_run_dir()  # NEW
    weights = str(run_dir / "weights" / "best.pt")
    data_yaml = str(YOLO_DATASET / "data.yaml")
    src_glob = request.json.get("src_glob") if request.is_json else None
    src_glob = src_glob or "/tmp/tiles/*_cum_*.png"
    cmd = [str(APP_ROOT/"bin"/"appre_to_prod"/"03_predict_smoke_test.sh"), weights, data_yaml, src_glob]
    res = run_cmd_logged(cmd, "03_smoketest")
    ok = (res["rc"] == 0)
    return jsonify(ok=ok, run_dir=str(run_dir), weights=weights, **res), (200 if ok else 500)

@app.post("/api/yolo_deploy")
def api_yolo_deploy():
    models = MODELS_DIR; models.mkdir(parents=True, exist_ok=True)
    run_dir = _effective_run_dir()  # NEW
    new_w = run_dir / "weights" / "best.pt"
    if not new_w.exists():
        return jsonify(ok=False, error="weights non trovati", path=str(new_w), run_dir=str(run_dir)), 400
    
    # backup vecchio
    cur = models / "yolo_current.pt"
    prev = models / "yolo_prev.pt"
    try:
        if cur.exists():
            if prev.exists(): prev.unlink()
            cur.replace(prev)
        # copia/symlink atomico (uso copy per compatibilità)
        import shutil, time
        shutil.copy2(str(new_w), str(cur))
        # VERSION
        (MODELS_DIR/"VERSION").write_text(json.dumps({
            "ts": datetime.utcnow().isoformat()+"Z",
            "source": str(YOLO_RUNS/YOLO_RUNNAME/"weights"/"best.pt"),
            "arch": YOLO_ARCH
        }, indent=2))
        # TODO (opzionale): restart servizio e health-check qui
        ok = True
        return jsonify(ok=True, deployed=str(cur), backup=str(prev), run_dir=str(run_dir))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500



@app.get("/api/drone_ids")
def api_drone_ids():
    """
    Raccoglie i nomi dei drone_id già presenti in data/recordings/<band>/<drone_id>/session_*
    Parametri opzionali:
      - band=24|52|58 per filtrare
      - q=substring (case-insensitive) per filtrare per testo
    """
    want_band = (request.args.get("band") or "").strip()
    q = (request.args.get("q") or "").strip().lower()

    base = (DATA / "recordings")
    if not base.exists():
        return jsonify(ok=True, ids=[])

    info = {}  # name -> {name,bands,set(sessions),last_ts}
    for band_dir in base.iterdir():
        if not band_dir.is_dir():
            continue
        band = band_dir.name
        if want_band and want_band != band:
            continue
        for drone_dir in band_dir.iterdir():
            if not drone_dir.is_dir():
                continue
            name = drone_dir.name
            if q and q not in name.lower():
                continue
            sessions = [p for p in drone_dir.glob("session_*") if p.is_dir()]
            if not sessions:
                continue
            last_ts = max(p.stat().st_mtime for p in sessions)
            item = info.setdefault(name, {"name": name, "bands": set(), "sessions": 0, "last_ts": 0.0})
            item["bands"].add(band)
            item["sessions"] += len(sessions)
            item["last_ts"] = max(item["last_ts"], last_ts)

    # normalizza e ordina (recente->vecchio)
    ids = []
    for v in info.values():
        ids.append({
            "name": v["name"],
            "bands": sorted(v["bands"]),
            "sessions": v["sessions"],
            "last_ts": v["last_ts"],
            "last_iso": datetime.utcfromtimestamp(v["last_ts"]).isoformat() + "Z"
        })
    ids.sort(key=lambda x: x["last_ts"], reverse=True)
    return jsonify(ok=True, ids=ids)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=False)
