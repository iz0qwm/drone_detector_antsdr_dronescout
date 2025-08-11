#!/usr/bin/env python3
import json, time, os, math, csv
from pathlib import Path
import datetime

# === IO ===
TRACKS_CURR = Path("/tmp/crpc_logs/tracks_current.json")
RID_FEED    = Path("/tmp/crpc_logs/rid.jsonl")
ASSOC_OUT   = Path("/tmp/crpc_logs/associations.jsonl")

# === Fingerprint DB / Modello ===
FPRINT_DB   = Path(os.environ.get("FPRINT_DB", "/home/raffaello/dataset/rf_fingerprint/fingerprint_db_full.csv"))
MODEL_PATH  = Path(os.environ.get("RF_MODEL", "/home/raffaello/models/rf_model.pkl"))  # opzionale

# === Bande e proto map (grezze) ===
PROTO_BANDS = {
    "DJI":   [(2400, 2500), (5725, 5875)],
    "ELRS":  [(900, 940),   (2400, 2500)],
    "PARROT":[(2400, 2500)],
    "AUTEL": [(2400, 2500), (5725, 5875)],
    "FIMI":  [(2400, 2500)],
}
BANDS = {"24": (2400.0, 2500.0), "58": (5725.0, 5875.0)}

# === Pesi/soglie ===
TIME_MAX_S   = 5.0
BAND_BONUS   = 0.2
SLEEP_S      = 0.2

K_NEIGHBORS  = 5
W_FREQ       = 1.0     # peso delta frequenza (normalizzato su span banda)
W_BW         = 0.7     # peso delta banda
W_HOP        = 0.5     # peso delta hop rate
ALPHA_FPRINT = 0.5     # mescola: final_score = assoc_score*(1-ALPHA) + fprint_score*ALPHA
FPRINT_MIN   = 0.35    # soglia minima fingerprint per considerarlo informativo
FINAL_MIN    = 0.55    # soglia finale per accettare l’associazione

# === Utils proto ===

def _iso_to_epoch(s):
    try:
        # gestisci Z/offset; fallback senza microsecondi
        return datetime.datetime.fromisoformat(
            s.replace("Z","+00:00")
        ).timestamp()
    except Exception:
        return None

def rid_ts(evt, fallback=None):
    # priorità: ts (epoch), poi timestamp ISO8601, poi time_ms, poi now, poi fallback
    for k in ("ts", "time_s", "time"):
        if k in evt:
            try:
                return float(evt[k])
            except Exception:
                pass
    if "time_ms" in evt:
        try:
            return float(evt["time_ms"]) / 1000.0
        except Exception:
            pass
    if "timestamp" in evt:
        t = _iso_to_epoch(str(evt["timestamp"]))
        if t is not None:
            return t
    # ultima spiaggia: ora o fallback
    return fallback if fallback is not None else time.time()

def guess_proto(model: str):
    m = (model or "").lower()
    if "dji" in m or "mavic" in m or "mini" in m or "air" in m:
        return "DJI"
    if "elrs" in m:
        return "ELRS"
    if "parrot" in m or "anafi" in m:
        return "PARROT"
    if "autel" in m or "evo" in m:
        return "AUTEL"
    if "fimi" in m or "xiaomi" in m:
        return "FIMI"
    return None

def band_compatible(proto, f_mhz):
    if not proto or proto not in PROTO_BANDS: 
        return False
    for lo, hi in PROTO_BANDS[proto]:
        if lo <= f_mhz <= hi:
            return True
    return False

# === Tracks ===
def load_tracks():
    if not TRACKS_CURR.exists():
        return []
    try:
        data = json.loads(TRACKS_CURR.read_text() or "[]")
        return [t for t in data if t.get("len", 0) >= 1]
    except Exception:
        return []

# === RID tail ===
def rid_tail(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r", buffering=1, errors="ignore") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(SLEEP_S); continue
            try:
                evt = json.loads(line); yield evt
            except Exception:
                continue

# === Associazione base (tempo+banda) ===
def score_assoc(track, rid_evt):
    ts_rid = rid_ts(rid_evt, fallback=track.get("last_seen", time.time()))
    dt = abs(float(ts_rid) - float(track.get("last_seen", 0)))
    if dt > TIME_MAX_S:
        return 0.0
    base = max(0.0, 1.0 - dt / TIME_MAX_S)
    proto = guess_proto(rid_evt.get("model", "")) or rid_evt.get("proto")
    band_ok = band_compatible(proto, float(track.get("center_freq_mhz", 0.0)))
    return base + (BAND_BONUS if band_ok else 0.0)

# === Fingerprint DB (CSV KNN) ===
class FingerDB:
    def __init__(self, path: Path):
        self.ok = False
        self.rows = []  # dict: band_key, f_center_mhz, bw_mhz, hop_rate_mhz_s, label
        if not path.exists():
            return
        try:
            with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                r = csv.DictReader(f)
                for row in r:
                    # prova a leggere nomi generici; adatta qui se i campi sono diversi
                    fmhz = float(row.get("center_freq_mhz", row.get("center_freq", "nan")))
                    bw   = float(row.get("bandwidth_mhz",  row.get("bandwidth", "nan")))
                    hop  = float(row.get("hop_rate_mhz_s", row.get("hop_rate", "0") or 0))
                    band_guess = "24" if 2400 <= fmhz <= 2500 else ("58" if 5725 <= fmhz <= 5875 else None)
                    label = row.get("class_name") or row.get("label") or row.get("proto") or "unknown"
                    if math.isnan(fmhz) or math.isnan(bw):
                        continue
                    self.rows.append({"band": band_guess, "f": fmhz, "bw": bw, "hop": hop, "label": label})
            self.ok = len(self.rows) > 0
        except Exception:
            self.ok = False

    def band_span(self, band_key):
        if band_key not in BANDS: return (1.0, 1.0)  # evita div/0
        lo, hi = BANDS[band_key]; return (lo, hi - lo)

    def dist(self, band_key, f, bw, hop, row):
        # normalizza su span di banda: frequenza e bw comparabili tra 2.4 e 5.8
        lo, span = self.band_span(band_key)
        if span <= 0: span = 1.0
        df  = abs((f - row["f"])) / span
        dbw = abs((bw - row["bw"])) / span
        dh  = abs(hop - row["hop"])  # hop di solito già piccolo, lasciamo grezzo
        # distanza pesata (L1)
        return W_FREQ*df + W_BW*dbw + W_HOP*dh

    def knn(self, band_key, f, bw, hop, k=K_NEIGHBORS):
        cands = [r for r in self.rows if (r["band"] == band_key or band_key is None)]
        if not cands: 
            cands = self.rows  # fallback
        scored = [(self.dist(band_key, f, bw, hop, r), r) for r in cands]
        scored.sort(key=lambda x: x[0])
        top = scored[:max(1, k)]
        # score in [0,1]: mappa distanza→confidenza semplice
        # conf = 1 / (1 + dist_norm); dist_norm ~ usa la distanza del peggiore nei top come scala
        if not top: 
            return {"label": None, "score": 0.0, "top": []}
        maxd = max(d for d, _ in top) or 1.0
        votes = {}
        for d, r in top:
            conf = 1.0 / (1.0 + (d / (maxd or 1.0)))
            votes[r["label"]] = votes.get(r["label"], 0.0) + conf
        best_label, best_score = max(votes.items(), key=lambda kv: kv[1])
        # normalizza su somma dei conf dei top
        total = sum(votes.values()) or 1.0
        return {
            "label": best_label,
            "score": round(best_score/total, 3),
            "top": [{"label": r["label"], "d": round(d,3)} for d, r in top]
        }

# === Modello sklearn opzionale ===
def try_load_model(path: Path):
    # Prova joblib, poi pickle
    try:
        import joblib
        if path.exists():
            return joblib.load(str(path))
    except Exception:
        pass
    try:
        import pickle
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

def model_predict(model, band_key, f, bw, hop):
    # adatta ai feature-name del tuo pipeline
    try:
        import numpy as np
        X = np.array([[f, bw, hop]])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            cls   = model.classes_[int(proba.argmax())]
            return str(cls), float(proba.max())
        else:
            y = model.predict(X)[0]
            return str(y), 0.6
    except Exception:
        return None, 0.0

# === MAIN ===
def main():
    print(f"🔗 Associatore CRPC+Fingerprint  tracks={TRACKS_CURR}  rid={RID_FEED}")
    # preload fingerprint DB e modello
    fdb = FingerDB(FPRINT_DB)
    if fdb.ok:
        print(f"📚 Fingerprint DB: {len(fdb.rows)} righe da {FPRINT_DB}")
    else:
        print("⚠️  Fingerprint DB non trovato/vuoto (uso solo regole base)")

    model = try_load_model(MODEL_PATH)
    if model:
        print(f"🧠 Modello sklearn caricato: {MODEL_PATH}")
    else:
        print("ℹ️  Nessun modello sklearn (opzionale)")

    for rid_evt in rid_tail(RID_FEED):
        rid_id = rid_evt.get("id") or rid_evt.get("drone_id") or rid_evt.get("icao") or "unknown"
        ts = rid_ts(rid_evt)
        tracks = load_tracks()
        if not tracks:
            continue

        # 1) trova best track per regole base
        best, best_s = None, 0.0
        for tr in tracks:
            s = score_assoc(tr, rid_evt)
            if s > best_s:
                best_s, best = s, tr

        if not best:
            continue

        # 2) fingerprint score (se DB o modello disponibili)
        fprint_label, fprint_score, top_matches = None, 0.0, []
        if best:
            f   = float(best.get("center_freq_mhz", 0.0))
            bw  = float(best.get("bandwidth_mhz", 0.0))
            hop = float(best.get("hop_rate_mhz_s", 0.0))
            band_key = best.get("band")

            # 2a) KNN su CSV
            if fdb.ok:
                kn = fdb.knn(band_key, f, bw, hop, k=K_NEIGHBORS)
                fprint_label = kn["label"]; fprint_score = kn["score"]; top_matches = kn["top"]

            # 2b) Modello sklearn (se presente) → fai media con knn
            if model:
                m_label, m_score = model_predict(model, band_key, f, bw, hop)
                if m_label:
                    if not fprint_label:
                        fprint_label, fprint_score = m_label, m_score
                    else:
                        # se le label coincidono, rafforza; altrimenti, piglia la migliore
                        if str(m_label) == str(fprint_label):
                            fprint_score = round(min(1.0, 0.5*fprint_score + 0.5*m_score), 3)
                        elif m_score > fprint_score:
                            fprint_label, fprint_score = m_label, m_score

        # 3) combina punteggi
        combined = best_s if fprint_score < FPRINT_MIN else ((1-ALPHA_FPRINT)*best_s + ALPHA_FPRINT*fprint_score)

        if combined >= FINAL_MIN:
            assoc = {
                "ts": ts,
                "track_id": best["track_id"],
                "center_freq_mhz": best["center_freq_mhz"],
                "bandwidth_mhz":   best["bandwidth_mhz"],
                "hop_rate_mhz_s":  best["hop_rate_mhz_s"],
                "rid_id": rid_id,
                "model": rid_evt.get("model"),
                "proto_guess": guess_proto(rid_evt.get("model", "")) or rid_evt.get("proto"),
                "assoc_score": round(best_s, 3),
                "fingerprint_label": fprint_label,
                "fingerprint_score": round(fprint_score, 3),
                "top_matches": top_matches,   # primi K dal CSV con distanza
                "final_score": round(combined, 3),
            }
            with ASSOC_OUT.open("a") as f:
                f.write(json.dumps(assoc, ensure_ascii=False) + "\n")
            print(f"✅ Assoc T{best['track_id']} ↔ RID {rid_id}  final={assoc['final_score']} (assoc={best_s:.2f}, fp={fprint_score:.2f})")
        else:
            # non abbastanza confidente
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("👋 stop")

