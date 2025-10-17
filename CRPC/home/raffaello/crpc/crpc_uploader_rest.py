#!/usr/bin/env python3
import os, time, json, requests, socket, threading, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from math import isfinite
from pathlib import Path
import re

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
API_KEY    = ""  # come sulla .5
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

def _slug(s: str) -> str:
    s = (s or "").strip()
    out = "".join(ch if ch.isalnum() else "-" for ch in s)
    return re.sub("-{2,}", "-", out).strip("-")[:64]

def _safe_id(s: str) -> str:
    # solo [a-z0-9-] per evitare regole restrittive; niente punti
    s = s.lower()
    return re.sub(r"[^a-z0-9-]", "-", s)[:128]

def _freq_bucket_mhz(freq: float, bw_mhz=None) -> int:
    # bin a 1 MHz (robusto contro micro-shift). Se vuoi più “larghi”, usa 2 o 5.
    try:
        return int(round(float(freq)))
    except Exception:
        return 0


def _patch_url_mask(coll, doc_id, field_names):
    base = _patch_url(coll, doc_id)  # .../documents/{coll}/{doc}?key=API_KEY
    # una entry updateMask.fieldPaths per ogni chiave che stai aggiornando
    mask = "".join(f"&updateMask.fieldPaths={name}" for name in field_names)
    return base + mask + "&currentDocument.exists=true"


def upsert(coll, doc_id, fields_dict):
    data = _fields(**fields_dict)
    try:
        r = requests.post(_doc_url(coll, doc_id), json=data, timeout=5)
        created = (r.status_code == 200)
        if r.status_code == 409:  # exists → PATCH (merge)
            r = requests.patch(
                _patch_url_mask(coll, doc_id, list(fields_dict.keys())),
                json=data, timeout=5
            )
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


# subito dopo i getter API
def api_df(): return get_json(f"{LOCAL_API}/api/df") or {}

def push_df_live():
    """Aggiorna df_bearing/df_confidence/df_sector del receiver, se disponibili, ogni giro."""
    try:
        dfwrap = api_df() or {}
        d = dfwrap.get("df") or {}
        bearing = d.get("bearing_deg")
        conf = d.get("confidence")
        if bearing is None:
            return
        sector = _bearing_to_sector(bearing)
        upsert("crpc_receivers", RECEIVER_ID, dict(
            df_bearing_deg=float(bearing),
            df_confidence=(None if conf is None else float(conf)),
            df_sector=(sector or None),
            ts_iso=_now_iso(),
            online=True
        ))
    except Exception:
        pass


# geo: punto destinazione da lat/lon (gradi), bearing (gradi), distanza (m)
from math import radians, degrees, sin, cos, asin, atan2
EARTH_R = 6371000.0
def dest_point(lat, lon, bearing_deg, dist_m):
    if lat is None or lon is None or bearing_deg is None: 
        return (None, None)
    lat1 = radians(float(lat)); lon1 = radians(float(lon))
    brg = radians(float(bearing_deg)); d = float(dist_m)/EARTH_R
    lat2 = asin(sin(lat1)*cos(d) + cos(lat1)*sin(d)*cos(brg))
    lon2 = lon1 + atan2(sin(brg)*sin(d)*cos(lat1), cos(d)-sin(lat1)*sin(lat2))
    return (degrees(lat2), (degrees(lon2)+540)%360-180)


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

# --- nuova configurazione “on-change” dei decimali
POS_DECIMALS = int(os.environ.get("CRPC_POS_DECIMALS", 5))  # 5 ~ 1.1 m
_last_pos_key = None
_last_pos_point_ts = 0

def _pos_key(lat, lon, decimals=POS_DECIMALS):
    if lat is None or lon is None:
        return None
    return (round(float(lat), decimals), round(float(lon), decimals))

def _bearing_to_sector(brg_deg: float) -> str:
    # 8 settori da 45° centrati su N/NE/E/...
    if brg_deg is None:
        return None
    brg = float(brg_deg) % 360.0
    sectors = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int((brg + 22.5) // 45) % 8
    return sectors[idx]


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

# opzionale: abilita/disable da env (default OFF)
RECEIVER_POSITIONS_ENABLED = bool(int(os.environ.get("RECEIVER_POSITIONS_ENABLED", "0")))

def push_receiver_position():
    global _last_pos_key, _last_pos_point_ts

    lat, lon, alt, ok, src = read_pos()
    ts = int(time.time()*1000)

    # 1) snapshot live sempre aggiornato (solo documento padre)
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

    # 2) (DISABLED di default) trail nella subcollection positions solo se abilitato
    if RECEIVER_POSITIONS_ENABLED:
        key = _pos_key(lat, lon)
        if key is None:
            return
        if key != _last_pos_key:
            ok_pt, err_pt = add_point("crpc_receivers", RECEIVER_ID, "positions", dict(
                lat=key[0], lon=key[1], timestamp=ts
            ))
            _append_jsonl(POSITIONS_JSONL, {
                "ts": ts, "receiverId": RECEIVER_ID, "event": "positions_add_onchange",
                "ok": ok_pt, "lat": key[0], "lon": key[1], **({"error": err_pt} if not ok_pt else {})
            })
            _last_pos_key = key
            _last_pos_point_ts = time.time()


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

    # ... dopo aver calcolato radius_m,color,rssi ...
    df = api_df()
    df_obj = (df or {}).get("df") or {}
    bearing = df_obj.get("bearing_deg")
    conf    = df_obj.get("confidence")
    # clamp conf (0..1)
    try:
        conf = max(0.0, min(1.0, float(conf))) if conf is not None else None
    except Exception:
        conf = None

    # calcola punto finale della freccia (dal ricevitore verso il bearing)
    lat0, lon0, alt0, okfix, src = read_pos()
    lat1, lon1 = dest_point(lat0, lon0, bearing, radius_m if bearing is not None else 0)

    fields = dict(
        receiverId=RECEIVER_ID,
        band=band,
        freq_mhz=float(freq),
        family=family,
        label=label,
        ts_iso=_now_iso(),
        radius_m=float(radius_m),
        color=color,
        # --- NEW: DF ---
        bearing_deg=(None if bearing is None else float(bearing)),
        df_confidence=(None if conf is None else float(conf)),
        rx_lat=lat0, rx_lon=lon0,
        end_lat=lat1, end_lon=lon1,
    )
    if rssi is not None:
        fields["rssi_dbm"] = float(rssi)


    # --- DocId stabile per firma (receiver+band+freq_0.1MHz+label) ---
    # ... hai già fields popolato qui sopra ...
    # BIN frequenza e docId stabile (niente punti)
    freq_key = _freq_bucket_mhz(freq, bw_mhz)     # es. 2473
    doc_id_raw = f"{RECEIVER_ID}-{band}-{freq_key}-{_slug(label)}"
    doc_id = _safe_id(doc_id_raw)

    ts_iso = _now_iso()
    fields.update({
        "ts_iso": ts_iso,          # ultimo avvistamento (HUD ordinata per ts_iso)
        "last_seen_iso": ts_iso,   # comodo per TTL e analitiche
        "freq_bucket_mhz": freq_key
    })

    ok_up, created, err = upsert("crpc_alerts", doc_id, fields)
    _append_jsonl(ALERTS_JSONL, {
        "ts": now_ts, "receiverId": RECEIVER_ID, "event": "alert_upsert",
        "ok": ok_up, "created": created, "doc_id": doc_id, **fields, **({"error": err} if not ok_up else {})
    })
    if ok_up:
        logger.info(f"[ALERT] {band}@{freq:.3f} MHz {label} rssi={rssi} → radius≈{fields['radius_m']:.0f}m (doc={doc_id})")




def main():
    logger.info(f"[CRPC→FS] avvio uploader per receiverId={RECEIVER_ID}")
    print(f"[CRPC→FS] avvio uploader per receiverId={RECEIVER_ID}")
    while True:
        try:
            push_receiver_position()
            push_df_live()
            push_alert_if_any()
        except Exception as e:
            logger.exception(f"[ERR] loop: {e}")
            print("[ERR]", e)
        time.sleep(2)

if __name__ == "__main__":
    main()
