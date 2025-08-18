#!/usr/bin/env python3
# RF scan classifier (indipendente dal bridge) — PATCH 2025-08-10
# Migliorie:
# 1) Fix seen_img_ts: niente variabili fuori scope; dedup per immagine/ts dentro il loop
# 2) VIDEO_IDS data-driven da yolo_classmap.json (fallback su euristica)
# 3) Log/telemetria extra: used_vts_for_knn, img_cls_id, img_votes
# 4) CLI per pesi/soglie/tail lettura detection
#
# I/O invariati:
#   - IN:  /tmp/crpc_logs/tracks_current.json (dallo spectral tracker)
#   - OUT: /tmp/crpc_logs/rfscan.jsonl (append) e rfscan_current.json (snapshot)
#   - LOG: /tmp/crpc_logs/assoc.log

import os, time, json, math, csv, datetime, sys, argparse
from pathlib import Path
from collections import Counter

# --- alias nomi modello ---
ALIASES = {
  "AIR3S": "DJI AIR3S",
  "Air3S": "DJI AIR3S",
  "DJI_AIR3S": "DJI AIR3S",
}
def norm_name(s):
    return ALIASES.get(str(s).upper().replace("_"," ").strip(), s)


# === IO ===
TRACKS_CURR = Path("/tmp/crpc_logs/tracks_current.json")
OUT_JSONL   = Path("/tmp/crpc_logs/rfscan.jsonl")
OUT_SNAP    = Path("/tmp/crpc_logs/rfscan_current.json")
LOG_PATH    = Path("/tmp/crpc_logs/assoc.log")

# === Detections YOLO (per image-hint) ===
DET_PATH        = Path("/tmp/crpc_logs/detections.jsonl")
#CLASSMAP_PATH   = Path("/home/raffaello/dataset/rf_fingerprint/yolo_classmap.json")
CLASSMAP_PATH   = Path("/home/raffaello/dataset/yolo_custom/classmap.json")

# === Fingerprint DB / Modello ===
FPRINT_DB   = Path(os.environ.get("FPRINT_DB", "/home/raffaello/dataset/rf_fingerprint/fingerprint_db_full.csv"))
#MODEL_PATH  = Path(os.environ.get("RF_MODEL", "/home/raffaello/models/rf_model.pkl"))
MODEL_PATH  = Path(os.environ.get("RF_MODEL", "/home/raffaello/apprendimento/models/served/rfscan.pkl"))

# === Bande e normalizzazione (MHz) ===
BANDS = {
    "24": (2400.0, 2500.0),
    "58": (5725.0, 5875.0),
    "52": (5170.0, 5250.0),
}

# --- globals per il modello ---
model_features = None

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    print(msg, flush=True)

def family_from_label(lbl: str):
    if not lbl: return None
    s = norm_name(lbl) 
    s = str(lbl).lower()
    if "dji" in s or "mavic" in s or "mini " in s or "air " in s: return "DJI"
    if "frsky" in s or "fr sky" in s or "taranis" in s or "accst" in s or "access" in s: return "FrSky"
    if "flysky" in s or "fly sky" in s or "fs-ia" in s or "fs i6" in s: return "FlySky"
    if "autel" in s or "evo" in s: return "Autel"
    if "parrot" in s or "anafi" in s: return "Parrot"
    if "fimi" in s or "xiaomi" in s: return "FIMI"
    if "elrs" in s or "expresslrs" in s or "radiomaster" in s: return "ELRS"
    if "tbs" in s or "crossfire" in s or "tracer" in s: return "TBS"
    return lbl  # fallback

# ---------- Fingerprint CSV (KNN) ----------
K_NEIGHBORS   = 5
W_FREQ        = 1.0
W_BW          = 0.7
W_HOP         = 0.5

class FingerDB:
    def __init__(self, path: Path):
        self.rows = []
        self.ok = False
        if not path.exists():
            return
        try:
            with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                r = csv.DictReader(f)
                hdr = [h.strip() for h in (r.fieldnames or [])]
                need = {"type","center_freq"}
                if not need.issubset(set(hdr)):
                    self.ok = False
                    return
                for row in r:
                    try:
                        cf = float(row.get("center_freq", "nan"))
                        if math.isnan(cf):
                            continue
                        fmhz = cf * 1000.0 if cf < 10 else cf  # 5.8 -> 5800 MHz
                        bw_rf  = row.get("fhs_bw", "")
                        bw_vts = row.get("vts_bw", "")
                        bw_rf  = float(bw_rf)  if bw_rf  not in ("", "nan", None) else None
                        bw_vts = float(bw_vts) if bw_vts not in ("", "nan", None) else None
                        fhsdc = row.get("fhsdc", "")
                        hop   = float(fhsdc) if fhsdc not in ("", "nan", None) else 0.0
                        label = (row.get("type") or "unknown").strip()
                        band_guess = (
                            "24" if 2400 <= fmhz <= 2500 else
                            "58" if 5725 <= fmhz <= 5875 else
                            ("52" if 5170 <= fmhz <= 5250 else None)
                        )
                        self.rows.append({
                            "band": band_guess,
                            "f": fmhz,
                            "bw_rf": bw_rf,
                            "bw_vts": bw_vts,
                            "hop": hop,
                            "label": label
                        })
                    except Exception:
                        continue
            self.ok = len(self.rows) > 0
        except Exception:
            self.ok = False

    def band_span(self, band_key):
        if band_key not in BANDS: return (1.0, 1.0)
        lo, hi = BANDS[band_key]; return (lo, hi - lo)

    def dist(self, band_key, f, bw_track, hop_track, row):
        lo, span = self.band_span(band_key)
        if span <= 0: span = 1.0
        df  = abs((f - row["f"])) / span
        dbw_opts = []
        if row.get("bw_rf")  is not None: dbw_opts.append(abs(bw_track - row["bw_rf"])  / span)
        if row.get("bw_vts") is not None: dbw_opts.append(abs(bw_track - row["bw_vts"]) / span)
        dbw = min(dbw_opts) if dbw_opts else 1.0
        dh = abs((hop_track or 0.0) - (row.get("hop") or 0.0))
        return W_FREQ*df + W_BW*dbw + W_HOP*dh

    def knn(self, band_key, f, bw, hop, k=K_NEIGHBORS):
        cands = [r for r in self.rows if (r["band"] == band_key or band_key is None)] or self.rows
        scored = [(self.dist(band_key, f, bw, hop, r), r) for r in cands]
        scored.sort(key=lambda x: x[0])
        top = scored[:max(1, k)]
        if not top:
            return {"label": None, "score": 0.0, "top": []}
        maxd = max(d for d, _ in top) or 1.0
        votes = {}
        for d, r in top:
            conf = 1.0 / (1.0 + (d / (maxd or 1.0)))
            votes[r["label"]] = votes.get(r["label"], 0.0) + conf
        best_label, best_score = max(votes.items(), key=lambda kv: kv[1])
        total = sum(votes.values()) or 1.0
        return {
            "label": best_label,
            "score": round(best_score/total, 3),
            "top": [
                {
                    "label": r["label"],
                    "d": round(d,3),
                    "f": r["f"],
                    "bw_rf": r.get("bw_rf"),
                    "bw_vts": r.get("bw_vts"),
                    "hop": r.get("hop")
                } for d, r in top
            ]
        }

# ---------- Modello sklearn ----------
def _safe_num(x, default=0.0):
    try:
        if x is None: return float(default)
        xx = float(x)
        if xx != xx:  # NaN
            return float(default)
        return xx
    except Exception:
        return float(default)

def build_runtime_features(track, img_hint=None, defaults=None):
    """
    Restituisce un dict con le 7 feature che il modello si aspetta,
    senza NaN. Usa 'img_hint' per stimare la video BW se disponibile.
    """
    defaults = defaults or {"snr": 15.0, "fhsdt": 0.0, "fhspp": 0.0, "file_size": 0.0}
    # center_freq in GHz "compatti"
    center_freq = _safe_num(track.get("center_freq_mhz"), 0.0) / 1000.0

    # bandwidth: se YOLO vede la portante video larga, usala
    bw_track = _safe_num(track.get("bandwidth_mhz"), 0.0)
    vts_bw = None
    if img_hint and isinstance(img_hint, dict):
        # es: img_hint.get("video_bw_mhz") oppure "bbox_bw_mhz"
        for k in ("video_bw_mhz", "bbox_bw_mhz", "vts_bw"):
            if k in img_hint:
                vts_bw = _safe_num(img_hint.get(k), None)
                break
    fhs_bw = vts_bw if (vts_bw and vts_bw > 1.5 * max(bw_track, 1e-6)) else bw_track

    # hop rate (MHz/s)
    fhsdc = _safe_num(track.get("hop_rate_mhz_s"), 0.0)
    # dwell time / peaks per period: se non li hai, 0
    fhsdt = _safe_num(track.get("hop_dwell_ms"), defaults["fhsdt"])
    fhspp = _safe_num(track.get("hop_peaks_per_s"), defaults["fhspp"])

    # SNR: prova dal track, poi calcola p95-floor, altrimenti default
    snr = track.get("snr_db")
    if snr is None:
        p95 = track.get("p95_dbm"); floor = track.get("floor_dbm")
        if p95 is not None and floor is not None:
            snr = (float(p95) - float(floor))
    snr = _safe_num(snr, defaults["snr"])

    # file_size: se non lo calcoli, 0
    file_size = _safe_num(track.get("tile_file_kb"), defaults["file_size"])

    feats = {
        "fhs_bw": _safe_num(fhs_bw, 0.0),
        "fhsdt":  _safe_num(fhsdt, 0.0),
        "fhsdc":  _safe_num(fhsdc, 0.0),
        "fhspp":  _safe_num(fhspp, 0.0),
        "file_size": _safe_num(file_size, 0.0),
        "snr": _safe_num(snr, defaults["snr"]),
        "center_freq": _safe_num(center_freq, 0.0),
    }
    return feats

def try_load_model(path: Path):
    def _unwrap(obj):
        # ritorna (model, features or None)
        if isinstance(obj, dict):
            m = obj.get("model") or obj.get("clf") or obj.get("estimator") or obj.get("pipeline")
            feats = obj.get("features")
            if hasattr(m, "predict"): 
                return m, feats
            # fallback: cerca qualunque value "predicibile"
            for v in obj.values():
                if hasattr(v, "predict"):
                    return v, feats
            return None, feats
        if hasattr(obj, "predict"):
            return obj, None
        return None, None
    try:
        import joblib
        if path.exists():
            return _unwrap(joblib.load(str(path)))
    except Exception:
        pass
    try:
        import pickle
        if path.exists():
            with open(path, "rb") as f:
                return _unwrap(pickle.load(f))
    except Exception:
        pass
    return None, None



def hop_stats_from_history(hist):
    if not hist or len(hist) < 2:
        return None, None, 0.0
    hist = sorted(hist, key=lambda x: x["t"])
    dt_list, dfdt_list = [], []
    freqs = []
    for a, b in zip(hist[:-1], hist[1:]):
        dt = max(1e-6, float(b["t"]) - float(a["t"]))
        df = float(b["f"]) - float(a["f"])
        dt_list.append(dt)
        dfdt_list.append(abs(df) / dt)
        freqs.extend([a["f"], b["f"]])
    fhsdt = sum(dt_list)/len(dt_list)
    fhsdc = sum(dfdt_list)/len(dfdt_list)
    fhspp = (max(freqs) - min(freqs)) if freqs else 0.0
    return fhsdt, fhsdc, fhspp

def vectorize_for_model(features, track):
    import numpy as np
    def _num(x, d=0.0):
        try:
            if x is None: return float(d)
            return float(x)
        except Exception:
            return float(d)

    PSD_FEATS = ["mean_dbm","p95_dbm","floor_dbm","crest_db","center_mhz","bw_mhz"]

    if features and set(features) == set(PSD_FEATS):
        p95    = _num(track.get("p95_dbm"), 0.0)
        floor  = _num(track.get("floor_dbm"), 0.0)
        mean   = track.get("mean_dbm")
        if mean is None:
            mean = (p95 + floor) / 2.0
        mean   = _num(mean, 0.0)
        crest  = _num(p95 - floor, 0.0)
        center = _num(track.get("center_freq_mhz"), 0.0)
        bw     = _num(track.get("bandwidth_mhz"), 0.0)

        row = {
            "mean_dbm":   mean,
            "p95_dbm":    p95,
            "floor_dbm":  floor,
            "crest_db":   crest,
            "center_mhz": center,
            "bw_mhz":     bw,
        }
        X = np.array([[row[name] for name in features]], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # fallback: modello runtime-7
    feats = build_runtime_features(track, img_hint=None, defaults={"snr": 15.0})
    cols = ['fhs_bw','fhsdt','fhsdc','fhspp','file_size','snr','center_freq']
    X = np.array([[feats.get(c, 0.0) for c in cols]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def model_predict(model, f=None, bw=None, hop=None, track=None):
    """
    Compatibile con la call vecchia (f,bw,hop) e con quella a 'track'.
    Recupera le feature dal modello (model._features) o dalla globale.
    """
    try:
        feats = getattr(model, "_features", None)
        if feats is None:
            feats = globals().get("model_features", None)

        tr = dict(track or {})
        if not tr:
            tr = {"center_freq_mhz": f, "bandwidth_mhz": bw, "hop_rate_mhz_s": hop}

        X = vectorize_for_model(feats, tr)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            cls   = model.classes_[int(proba.argmax())]
            return str(cls), float(proba.max())
        else:
            y = model.predict(X)[0]
            return str(y), 0.60
    except Exception as e:
        try:
            with open("/tmp/crpc_logs/assoc.log","a") as lf:
                lf.write(f"[{now_str()}] [model_predict] errore: {e}\n")
                lf.write(f"   feats={feats} X={X.tolist() if 'X' in locals() else 'n/a'} tr_keys={list((track or {}).keys())}\n")
        except Exception:
            pass
        return None, 0.0


# ---------- Image hint ----------
def load_classmap():
    try:
        return json.loads(CLASSMAP_PATH.read_text())
    except Exception:
        return {"by_id": {}, "by_name": {}}

CLASSMAP = load_classmap()

def derive_video_ids(classmap):
    """Preferisci elenco esplicito se presente; altrimenti euristica per nomi."""
    # 1) campo esplicito, se presente
    explicit = classmap.get("video_ids")
    if isinstance(explicit, list) and all(isinstance(x, (str,int)) for x in explicit):
        return {str(x) for x in explicit}
    # 2) euristica: nomi che contengono 'image_transmission' o 'video'
    by_id = classmap.get("by_id", {})
    vids = set()
    for cid, name in by_id.items():
        s = str(name).lower()
        if ("image_transmission" in s) or ("video" in s) or ("vts" in s) or ("square" in s):
            vids.add(str(cid))
    if vids:
        return vids
    # 3) fallback legacy
    return {"3","12"}

VIDEO_IDS = derive_video_ids(CLASSMAP)

# ---------- Image hint (migliorato: peso per conf + finestra temporale) ----------
HINT_TIME_S = float(os.environ.get("HINT_TIME_S", "30"))  # era 3.0
 

# distanza massima in secondi tra detection YOLO e track ts

def image_hint_for_track(track_img, tail_kb, window_lines, ts_center=None):
    """
    Aggrega le classi YOLO per la stessa immagine (track['img']),
    pesando per 'conf' e filtrando per finestra temporale ±HINT_TIME_S se ts_center è passato.
    Ritorna (proto, score[0..1], cls_id, votes_dict).
    """
    if not track_img or not DET_PATH.exists():
        return None, 0.0, None, {}
    votes = Counter()
    total_w = 0.0
    try:
        with DET_PATH.open("rb") as f:
            try:
                f.seek(-tail_kb*1024, os.SEEK_END)
            except Exception:
                pass
            tail = f.read().decode(errors="ignore").splitlines()[-window_lines:]
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
                conf = float(d.get("conf", 0.0))
                if conf <= 0.01:
                    continue
                votes[cls_id] += conf
                total_w += conf
            except Exception:
                continue
    except Exception:
        return None, 0.0, None, {}

    if not votes:
        return None, 0.0, None, {}
    cls_id, w = max(votes.items(), key=lambda kv: kv[1])
    proto = CLASSMAP.get("by_id", {}).get(cls_id)
    if not proto:
        return None, 0.0, None, dict(votes)
    score = float(w / max(1e-9, total_w))
    return proto, score, cls_id, dict(votes)

def estimate_vts_bw_from_image(track_img, band_key, tail_kb, ts_center=None):
    """Stima bandwidth video (MHz) dalla bbox più larga tra classi 'video' (con filtro temporale)."""
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
            tail = f.read().decode(errors="ignore").splitlines()[-args.dets_tail_lines:]
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
    p = argparse.ArgumentParser(description="RF scan classifier (model/csv/img fusion)")
    p.add_argument("--w-model", type=float, default=0.5, help="Peso voto modello sklearn")
    p.add_argument("--w-csv",   type=float, default=0.3, help="Peso voto KNN su fingerprint CSV")
    p.add_argument("--w-img",   type=float, default=0.2, help="Peso voto image-hint YOLO")
    p.add_argument("--min-track-len", type=int, default=1, help="Lunghezza minima track per essere considerato")
    p.add_argument("--fprint-min", type=float, default=0.05, help="Soglia score minima per stampare ✅")
    p.add_argument("--poll-sec", type=float, default=0.25, help="Polling di tracks_current.json (s)")
    p.add_argument("--log-trim-mb", type=int, default=5, help="Trim del log assoc.log (MB)")
    p.add_argument("--dets-tail-kb", type=int, default=200, help="Dimensione tail letta da detections.jsonl (KB)")
    p.add_argument("--dets-tail-lines", type=int, default=2000, help="Righe massimo di detections lette dal tail")
    return p.parse_args()

args = parse_args()

def main():
    last_seen = {}          # track_id -> ts ultimo classificato
    seen_img_ts = {}        # img basename -> ts ultimo processato

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SNAP.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    fdb = FingerDB(FPRINT_DB)
    if fdb.ok:
        log(f"📚 Fingerprint DB caricato: {len(fdb.rows)} righe ({FPRINT_DB.name})")
    else:
        log("⚠️ Fingerprint DB non trovato/vuoto: proseguo solo con il modello (se presente).")

    # Model
    # Caricamento modello
    global model_features
    model, model_features = try_load_model(MODEL_PATH)
    if model:
        try:
            # attacca le features direttamente all'oggetto modello
            setattr(model, "_features", model_features)
        except Exception:
            pass
        log(f"🧠 Modello RF caricato: {MODEL_PATH.name}  (features={model_features or 'runtime-7'})")
    else:
        log("ℹ️ Nessun modello RF (.pkl) trovato: userò solo il CSV/immagine.")


    log("🔎 RF scan classifier attivo (senza bridge)…")
    while True:
        tracks = load_tracks(args.min_track_len)

        # Collassa per immagine: tieni il track con BW max per la stessa immagine/ts
        grouped = {}
        for tr in tracks:
            tid = tr.get("track_id")
            ts  = float(tr.get("last_seen", 0))
            img = tr.get("img")

            # evita doppioni stessa immagine allo stesso timestamp
            if img and seen_img_ts.get(img) == ts:
                continue
            # evita riclassifiche dello stesso track_id/ts
            if last_seen.get(tid) == ts:
                continue

            key = img or f"tid:{tid}"
            if key not in grouped or tr.get("bandwidth_mhz", 0) > grouped[key].get("bandwidth_mhz", 0):
                grouped[key] = tr

        tracks = list(grouped.values())
        predictions = []
        used_imgs = []  # per aggiornare seen_img_ts

        for tr in tracks:
            tid = tr.get("track_id")
            ts  = float(tr.get("last_seen", 0))
            img = tr.get("img")
            if last_seen.get(tid) == ts:
                continue

            band = tr.get("band")
            f    = float(tr.get("center_freq_mhz", 0.0))
            bw   = float(tr.get("bandwidth_mhz", 0.0))
            hop  = float(tr.get("hop_rate_mhz_s", 0.0))

            # --- Euristica: Wi-Fi 80 MHz in 2.4 GHz ---
            def is_wifi80_band24(band, bw_mhz, hop_rate):
                try:
                    b = str(band)
                    bw = float(bw_mhz or 0.0)
                    hop = float(hop_rate or 0.0)
                except Exception:
                    return False
                # 2.4 GHz: Wi-Fi 80 MHz risulta 75–95 MHz nel nostro stimatore
                # hop molto basso/stabile aiuta a distinguerlo da salti più “vivaci”
                return (b == "24") and (50.0 <= bw <= 95.0) and (abs(hop) <= 2.0)

            #if is_wifi80_band24(tr.get("band"), tr.get("bandwidth_mhz"), tr.get("hop_rate_mhz_s")):
                # o lo scarti del tutto:
                #continue
            # ... dentro il loop for tr in tracks: subito dopo aver letto band/f/bw/hop ...
            wifi80 = is_wifi80_band24(band, bw, hop)
            if wifi80:
                # marchia il track come Wi-Fi e riduci l’impatto in fusione
                tr["_heuristic_wifi80"] = True

            # 1) CSV KNN (con eventuale BW video da immagine)
            fp_label, fp_score, top = None, 0.0, []
            used_vts_for_knn = False
            vts_bw_est = None
            if fdb.ok:
                vts_bw_est = estimate_vts_bw_from_image(img, band, args.dets_tail_kb, ts_center=ts)
                bw_for_knn = bw
                if vts_bw_est and vts_bw_est > bw * 1.5:
                    log(f"   ⤷ vts_bw_est ~ {vts_bw_est:.2f} MHz (da immagine), usata per KNN al posto di {bw:.2f}")
                    bw_for_knn = vts_bw_est
                    used_vts_for_knn = True
                kn = fdb.knn(band, f, bw_for_knn, hop, k=K_NEIGHBORS)
                fp_label, fp_score, top = kn["label"], kn["score"], kn["top"]

            # 2) Model
            mdl_label, mdl_score = None, 0.0
            if model:
                mdl_label, mdl_score = model_predict(model, tr)

            # 3) Image hint
            img_label, img_score, img_cls_id, img_votes = image_hint_for_track(img, args.dets_tail_kb, args.dets_tail_lines, ts_center=ts)

            img_w = None
            try:
                img_w = float(img_votes.get("w_mhz_best", 0.0) if isinstance(img_votes, dict) else 0.0)
            except Exception:
                img_w = None

            # Se siamo in banda 24 e la bbox è “realistica”, usa quella come bw del track
            if tr.get("band") in (24, "24"):
                if img_w and 2.0 <= img_w <= 12.0:
                    tr["bandwidth_mhz"] = img_w
                    
            # --- NORMALIZZA I NOMI QUI ---
            if mdl_label: mdl_label = norm_name(mdl_label)
            if fp_label:  fp_label  = norm_name(fp_label)
            if img_label: img_label = norm_name(img_label)

            # 4) Fusione
            fam_model = family_from_label(mdl_label)
            fam_csv   = family_from_label(fp_label)
            fam_img   = family_from_label(img_label)

            w_model = args.w_model
            w_csv   = args.w_csv
            w_img   = args.w_img
            b = tr.get("band")
            if str(b) == "24":
                # in 2.4 il CSV è più discriminante sui modelli DJI; YOLO lo usiamo come “famiglia”
                w_csv, w_model, w_img = 0.55, 0.30, 0.15
            if tr.get("len", 0) < 3:
               w_model = 0.35
               w_csv   = 0.35
               w_img   = 0.45
            if used_vts_for_knn:
               w_csv += 0.15

            cands = {f for f in [fam_model, fam_csv, fam_img] if f}
            if cands:
               def vote_for(fam):
                  v = 0.0
                  if fam == fam_model: v += w_model * (mdl_score or 0.0)
                  if fam == fam_csv:   v += w_csv   * (fp_score  or 0.0)
                  if fam == fam_img:   v += w_img   * (img_score or 0.0)
                  return v
               fam_best = max(cands, key=vote_for)
               score = vote_for(fam_best)
               label = next((l for l in [fp_label, mdl_label, img_label] if family_from_label(l)==fam_best), fam_best)
               src   = "model/csv/img(fam)"
            else:
               label, score, src = None, 0.0, None

            # --- Forzatura: non segnalare come drone i Wi-Fi 80 MHz 2.4 ---
            if tr.get("_heuristic_wifi80"):
                label = "Wi-Fi (80 MHz)"
                score = 0.99
                src   = "heuristic"

            pred = {
                "ts": ts,
                "track_id": tid,
                "band": band,
                "img": tr.get("img"),
                "center_freq_mhz": round(f,3),
                "bandwidth_mhz": round(bw,3),
                "hop_rate_mhz_s": round(hop,3),
                "label": label,
                "family": family_from_label(label),
                "score": round(float(score), 3),
                "source": src,
                "model": {"label": mdl_label, "score": round(mdl_score,3)},
                "csv":   {"label": fp_label,  "score": round(fp_score,3), "top": top, "used_vts_for_knn": used_vts_for_knn},
                "image_hint": {"label": img_label, "score": round(img_score or 0.0,3), "cls_id": img_cls_id, "votes": img_votes},
                "vts_bw_est_mhz": round(vts_bw_est, 2) if (vts_bw_est is not None) else None,
                "bandwidth_used_for_csv_mhz": round((vts_bw_est if used_vts_for_knn else bw), 2),
                "badges": (["wifi80","non-drone"] if tr.get("_heuristic_wifi80") else [])
            }

            log(f"   ⤷ votes  model={mdl_label}:{mdl_score:.2f}  csv={fp_label}:{fp_score:.2f}  img={img_label}:{(img_score or 0):.2f}")
            with OUT_JSONL.open("a") as g:
                g.write(json.dumps(pred, ensure_ascii=False) + "\n")

            if label and (score >= args.fprint_min) and (label != "Wi-Fi (80 MHz)"):
                log(f"✅ RFscan T{tid} [{band}] {f:.3f}MHz bw={bw:.3f} → {label} ({src} {score:.2f})")
            else:
                log(f"ℹ️  RFscan T{tid} [{band}] {f:.3f}MHz bw={bw:.3f} → (debole) "
                    f"model={mdl_label}:{mdl_score:.2f} csv={fp_label}:{fp_score:.2f} img={img_label}:{img_score or 0:.2f})")


            predictions.append(pred)
            last_seen[tid] = ts
            if img:  # <-- FIX: aggiornamento dentro il loop per ogni immagine usata
                used_imgs.append((img, ts))

        # aggiorna seen_img_ts per tutte le immagini usate in questo batch
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