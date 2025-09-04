#!/usr/bin/env python3
import os, time, json, requests, socket, threading, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from math import isfinite
from pathlib import Path

# === LOGGING ===
LOG_DIR  = Path("/tmp/crpc_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _mk_handler(name):
    h = RotatingFileHandler(LOG_DIR / name, maxBytes=5_000_000, backupCount=3)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    return h

logger = logging.getLogger("crpc_uploader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(_mk_handler("uploader.log"))

ALERTS_JSONL   = LOG_DIR / "alerts.jsonl"
POSITIONS_JSONL= LOG_DIR / "positions.jsonl"

def _append_jsonl(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"JSONL write error {path}: {e}")

# === CONFIG ===
PROJECT_ID = "tutto-sui-droni-community"
API_KEY    = "AIzaSyAs13Jwj4ZOd9SS9W7C7UxeJy62wS6qphQ"  # come sulla .5
FIRESTORE  = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"
LOCAL_API  = "http://127.0.0.1:8080"   # crpc_api.py sulla .6
RECEIVER_ID = os.environ.get("CRPC_RECEIVER_ID", socket.gethostname())  # es: 'crpc-19216816'

RSSI_TABLE = [
    {"max_dbm": -80, "radius_m": 1200, "color": "#f2cb05"},
    {"max_dbm": -70, "radius_m": 700,  "color": "#ff8c00"},
    {"max_dbm": -60, "radius_m": 400,  "color": "#ff4d4d"},
    {"max_dbm": -50, "radius_m": 220,  "color": "#d12a2a"},
]

RECEIVER_SOURCE_URL = os.environ.get("RECEIVER_SOURCE_URL", "http://192.168.1.5:8080/receiver")
POS_FALLBACK_JSON = "/etc/crpc_pos.json" # { "lat": 41.9, "lon": 12.5, "alt": 60 }

# ---- Firestore helpers
def _doc_url(coll, doc_id):     return f"{FIRESTORE}/{coll}?documentId={doc_id}&key={API_KEY}"
def _patch_url(coll, doc_id):   return f"{FIRESTORE}/{coll}/{doc_id}?key={API_KEY}"
def _sub_url(coll, doc_id, sub):return f"{FIRESTORE}/{coll}/{doc_id}/{sub}?key={API_KEY}"

def _now_iso(): return datetime.utcnow().isoformat()+"Z"
def _fields(**k):
    out = {"fields":{}}
    for a,v in k.items():
        if isinstance(v, (int,)) and not isinstance(v, bool):
            out["fields"][a] = {"integerValue": v}
        elif isinstance(v, float):
            out["fields"][a] = {"doubleValue": v}
        elif isinstance(v, bool):
            out["fields"][a] = {"booleanValue": v}
        elif v is None:
            out["fields"][a] = {"nullValue": None}
        else:
            out["fields"][a] = {"stringValue": str(v)}
    return out

def upsert(coll, doc_id, fields_dict):
    data = _fields(**fields_dict)
    try:
        r = requests.post(_doc_url(coll, doc_id), json=data, timeout=5)
        created = (r.status_code == 200)
        if r.status_code == 409:  # exists → PATCH
            r = requests.patch(_patch_url(coll, doc_id), json=data, timeout=5)
            created = False
        ok = (r.status_code // 100 == 2)
        if not ok:
            logger.error(f"[FS] ERR {coll}/{doc_id}: {r.status_code} {r.text}")
        else:
            logger.info(f"[FS] {'CREATED' if created else 'UPDATED'} {coll}/{doc_id}")
        return ok, created, (r.text if not ok else "")
    except Exception as e:
        logger.exception(f"[FS] EXC upsert {coll}/{doc_id}: {e}")
        return False, False, str(e)

def add_point(coll, doc_id, subcoll, fields_dict):
    try:
        r = requests.post(_sub_url(coll, doc_id, subcoll), json=_fields(**fields_dict), timeout=5)
        ok = (r.status_code // 100 == 2)
        if not ok:
            logger.error(f"[FS] ERR ADD {coll}/{doc_id}/{subcoll}: {r.status_code} {r.text}")
        else:
            logger.info(f"[FS] ADDED point {coll}/{doc_id}/{subcoll}")
        return ok, (r.text if not ok else "")
    except Exception as e:
        logger.exception(f"[FS] EXC add_point {coll}/{doc_id}/{subcoll}: {e}")
        return False, str(e)

# ---- CRPC API fetchers
def get_json(url, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok: return r.json()
    except Exception:
        pass
    return None

def api_detections(): return get_json(f"{LOCAL_API}/api/detections") or {}
def api_uav_status(): return get_json(f"{LOCAL_API}/api/uav_status") or {}
def api_spectrum():   return get_json(f"{LOCAL_API}/api/spectrum") or {}

# ---- RSSI & distanza
def df_tolerance_mhz(bw_mhz):
    try:
        bw = float(bw_mhz or 6.0)
    except Exception:
        bw = 6.0
    return max(0.2, min(3.0, 0.20 * bw))  # 20% della BW, clamped

def best_peak_rssi(spec, freq_mhz, band=None, bw_mhz=None):
    peaks = (spec or {}).get("peaks") or []
    if not (isfinite(freq_mhz) and peaks): return None
    tol = df_tolerance_mhz(bw_mhz)
    best = None; best_df = 1e9
    for p in peaks:
        f = p.get("freq_mhz") or p.get("frequency")
        a = p.get("power_dbm") or p.get("dbm") or p.get("amp") or p.get("amp_dbm")
        try:
            f = float(f)
        except Exception:
            continue
        if band is not None and str(p.get("band")) not in (None, "", str(band)):
            continue
        df = abs(f - freq_mhz)
        if df <= tol:
            if df < best_df or (abs(df - best_df) < 1e-6 and (a is not None) and a > (best or -1e9)):
                best, best_df = a, df
    return best

def radius_from_rssi(rssi_dbm):
    tbl = sorted(RSSI_TABLE, key=lambda x: x["max_dbm"])
    if rssi_dbm <= tbl[0]["max_dbm"]:
        return tbl[0]["radius_m"], tbl[0]["color"]
    if rssi_dbm >= tbl[-1]["max_dbm"]:
        return tbl[-1]["radius_m"], tbl[-1]["color"]
    for i in range(1, len(tbl)):
        a, b = tbl[i-1], tbl[i]
        if a["max_dbm"] <= rssi_dbm <= b["max_dbm"]:
            f = (rssi_dbm - a["max_dbm"]) / (b["max_dbm"] - a["max_dbm"] + 1e-9)
            rad = a["radius_m"] + (b["radius_m"] - a["radius_m"]) * f
            return rad, b["color"]
    return 600.0, "#ff8c00"

# ---- Posizione del ricevitore
_last_pos_sent = 0
def read_pos():
    try:
        r = requests.get(RECEIVER_SOURCE_URL, timeout=2.5)
        if not r.ok:
            return (None, None, None, False, "receiver-endpoint")
        j = r.json()
        lat = j.get("lat"); lon = j.get("lon"); alt = j.get("alt"); fix_ok = bool(j.get("fix_ok", False))
        if not fix_ok or lat is None or lon is None:
            lat = j.get("last_lat"); lon = j.get("last_lon"); alt = j.get("last_alt", alt); fix_ok = False
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
        alt = float(alt) if alt is not None else 0.0
        return (lat, lon, alt, fix_ok, "bridge-192.168.1.5")
    except Exception:
        return (None, None, None, False, "receiver-error")

def push_receiver_position():
    global _last_pos_sent
    lat, lon, alt, ok, src = read_pos()
    ts = int(time.time()*1000)

    ok_up, created, err = upsert("crpc_receivers", RECEIVER_ID, dict(
        lat=lat if lat is not None else 0.0,
        lon=lon if lon is not None else 0.0,
        alt=alt if alt is not None else 0.0,
        fix_ok=bool(ok), source=src, ts_iso=_now_iso(), host=RECEIVER_ID, online=True,
    ))
    _append_jsonl(POSITIONS_JSONL, {
        "ts": ts, "receiverId": RECEIVER_ID, "event": "receiver_upsert",
        "ok": ok_up, "created": created, "lat": lat, "lon": lon, "alt": alt, "fix_ok": ok, "source": src,
        **({"error": err} if not ok_up else {})
    })

    if (time.time() - _last_pos_sent) > 10 and lat and lon:
        ok_pt, err_pt = add_point("crpc_receivers", RECEIVER_ID, "positions", dict(
            lat=lat, lon=lon, timestamp=ts
        ))
        _append_jsonl(POSITIONS_JSONL, {
            "ts": ts, "receiverId": RECEIVER_ID, "event": "positions_add",
            "ok": ok_pt, "lat": lat, "lon": lon, **({"error": err_pt} if not ok_pt else {})
        })
        _last_pos_sent = time.time()

# ---- Alerts (ring + distanza) dal CRPC
_last_alert_sent = {}  # chiave (band,freq,label) → ts
def push_alert_if_any():
    det = api_detections()
    uav = api_uav_status()
    now_ts = int(time.time()*1000)

    if not (uav and uav.get("active")):
        _append_jsonl(ALERTS_JSONL, {
            "ts": now_ts, "receiverId": RECEIVER_ID, "event": "skipped_uav_inactive"
        })
        return

    arr = det.get("detections") or []
    if not arr:
        return
    last = arr[0]
    band = str(last.get("band") or "")
    freq = float(last.get("freq_mhz") or 0.0)
    label = last.get("label") or last.get("model") or ""
    family = last.get("family") or last.get("brand") or ""
    bw_mhz = last.get("bw_mhz") or last.get("bandwidth_mhz")

    spec = api_spectrum()
    rssi = best_peak_rssi(spec, freq_mhz=freq, band=band, bw_mhz=bw_mhz)

    # default se non abbiamo un RSSI valido
    DEFAULT_RADIUS = 600.0
    DEFAULT_COLOR  = "#ff8c00"

    if rssi is not None:
        radius_m, color = radius_from_rssi(float(rssi))
    else:
        radius_m, color = DEFAULT_RADIUS, DEFAULT_COLOR

    key = (band, round(freq,3), str(label))
    last_ts = _last_alert_sent.get(key, 0)
    if time.time() - last_ts < 8:   # anti-duplicato
        _append_jsonl(ALERTS_JSONL, {
            "ts": now_ts, "receiverId": RECEIVER_ID, "event": "suppressed_duplicate",
            "band": band, "freq_mhz": round(freq,3), "label": label
        })
        return

    _last_alert_sent[key] = time.time()

    fields = dict(
        receiverId=RECEIVER_ID,
        band=band,
        freq_mhz=float(freq),
        family=family,
        label=label,
        ts_iso=_now_iso(),
        radius_m=float(radius_m),   # <-- ora SEMPRE presente
        color=color                 # <-- ora SEMPRE presente
    )
    if rssi is not None:
        fields["rssi_dbm"] = float(rssi)

    ok_up, created, err = upsert("crpc_alerts", f"{RECEIVER_ID}-{int(time.time())}", fields)
    _append_jsonl(ALERTS_JSONL, {
        "ts": now_ts, "receiverId": RECEIVER_ID, "event": "alert_upsert",
        "ok": ok_up, "created": created, **fields, **({"error": err} if not ok_up else {})
    })
    if ok_up:
        logger.info(f"[ALERT] {band}@{freq:.3f} MHz {label} rssi={rssi} → radius≈{radius_m:.0f}m")


def main():
    logger.info(f"[CRPC→FS] avvio uploader per receiverId={RECEIVER_ID}")
    print(f"[CRPC→FS] avvio uploader per receiverId={RECEIVER_ID}")
    while True:
        try:
            push_receiver_position()
            push_alert_if_any()
        except Exception as e:
            logger.exception(f"[ERR] loop: {e}")
            print("[ERR]", e)
        time.sleep(2)

if __name__ == "__main__":
    main()
