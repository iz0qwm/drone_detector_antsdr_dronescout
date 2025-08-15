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

# === IO ===
TRACKS_CURR = Path("/tmp/crpc_logs/tracks_current.json")
OUT_JSONL   = Path("/tmp/crpc_logs/rfscan.jsonl")
OUT_SNAP    = Path("/tmp/crpc_logs/rfscan_current.json")
LOG_PATH    = Path("/tmp/crpc_logs/assoc.log")

# === Detections YOLO (per image-hint) ===
DET_PATH        = Path("/tmp/crpc_logs/detections.jsonl")
CLASSMAP_PATH   = Path("/home/raffaello/dataset/rf_fingerprint/yolo_classmap.json")

# === Fingerprint DB / Modello ===
FPRINT_DB   = Path(os.environ.get("FPRINT_DB", "/home/raffaello/dataset/rf_fingerprint/fingerprint_db_full.csv"))
MODEL_PATH  = Path(os.environ.get("RF_MODEL", "/home/raffaello/models/rf_model.pkl"))

# === Bande e normalizzazione (MHz) ===
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
def try_load_model(path: Path):
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

def model_predict(model, f, bw, hop, track=None):
    import pandas as pd
    try:
        names = ['fhs_bw', 'fhsdt', 'fhsdc', 'fhspp', 'file_size', 'snr', 'center_freq']
        row = {col: float('nan') for col in names}
        row['center_freq'] = float(f)
        row['fhs_bw']      = float(bw)
        fhsdt = fhsdc = fhspp = None
        file_size = None
        snr = None
        if isinstance(track, dict):
            hist = track.get("history")
            fhsdt, fhsdc, fhspp = hop_stats_from_history(hist)
            if (fhsdc is None or fhsdc == 0.0) and hop is not None:
                try:
                    fhsdc = float(hop)
                    if fhsdc > 0:
                        fhsdt = 1.0 / max(1e-6, fhsdc)
                except Exception:
                    pass
            p = track.get("tile_path")
            if p and os.path.exists(p):
                try: file_size = os.path.getsize(p)
                except Exception: pass
            if track.get("yolo_conf") is not None:
                try: snr = float(track['yolo_conf']) * 30.0
                except Exception: pass
        def _f(x):
            try:    return float(x)
            except: return float('nan')
        row['fhsdt']     = _f(fhsdt)
        row['fhsdc']     = _f(fhsdc)
        row['fhspp']     = _f(fhspp if fhspp is not None else 0.0)
        row['file_size'] = _f(file_size if file_size is not None else 0.0)
        row['snr']       = _f(snr if snr is not None else 10.0)
        X = pd.DataFrame([row], columns=names)
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

def image_hint_for_track(track_img, tail_kb, window_lines):
    """
    Aggrega le classi YOLO per la stessa immagine (track['img']).
    Ritorna (proto, score[0..1], cls_id, votes_dict).
    """
    if not track_img or not DET_PATH.exists():
        return None, 0.0, None, {}
    counts = Counter()
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
                cls_id = str(d.get("cls"))
                counts[cls_id] += 1
            except Exception:
                continue
    except Exception:
        return None, 0.0, None, {}
    if not counts:
        return None, 0.0, None, {}
    cls_id, n = counts.most_common(1)[0]
    proto = CLASSMAP.get("by_id", {}).get(cls_id)
    if not proto:
        return None, 0.0, None, dict(counts)
    total = sum(counts.values())
    return proto, float(n / max(1, total)), cls_id, dict(counts)

def estimate_vts_bw_from_image(track_img, band_key, tail_kb):
    """Stima bandwidth video (MHz) dalla bbox più larga tra classi 'video'."""
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
    model = try_load_model(MODEL_PATH)
    if model:
        log(f"🧠 Modello RF caricato: {MODEL_PATH.name}")
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

            # 1) CSV KNN (con eventuale BW video da immagine)
            fp_label, fp_score, top = None, 0.0, []
            used_vts_for_knn = False
            vts_bw_est = None
            if fdb.ok:
                vts_bw_est = estimate_vts_bw_from_image(img, band, args.dets_tail_kb)
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
                mdl_label, mdl_score = model_predict(model, f, bw, hop, track=tr)

            # 3) Image hint
            img_label, img_score, img_cls_id, img_votes = image_hint_for_track(img, args.dets_tail_kb, args.dets_tail_lines)

            # 4) Fusione
            fam_model = family_from_label(mdl_label)
            fam_csv   = family_from_label(fp_label)
            fam_img   = family_from_label(img_label)

            w_model = args.w_model
            w_csv   = args.w_csv
            w_img   = args.w_img
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
                "bandwidth_used_for_csv_mhz": round((vts_bw_est if used_vts_for_knn else bw), 2)
            }

            log(f"   ⤷ votes  model={mdl_label}:{mdl_score:.2f}  csv={fp_label}:{fp_score:.2f}  img={img_label}:{(img_score or 0):.2f}")
            with OUT_JSONL.open("a") as g:
                g.write(json.dumps(pred, ensure_ascii=False) + "\n")

            if label and score >= args.fprint_min:
                log(f"✅ RFscan T{tid} [{band}] {f:.3f}MHz bw={bw:.3f} → {label} ({src} {score:.2f})")
            else:
                log(f"ℹ️  RFscan T{tid} [{band}] {f:.3f}MHz bw={bw:.3f} → (debole) "
                    f"model={mdl_label}:{mdl_score:.2f} csv={fp_label}:{fp_score:.2f} img={img_label}:{img_score or 0:.2f}")

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