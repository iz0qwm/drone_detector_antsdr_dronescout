#!/usr/bin/env python3
import argparse, json, pathlib, sys
import numpy as np
import joblib

APP_ROOT = pathlib.Path("/home/raffaello/apprendimento")
DATA = APP_ROOT / "data"

def load_last_feature(session_dir: pathlib.Path):
    """
    Trova la feature per questa sessione usando meta.json (affidabile),
    altrimenti fa fallback cercando session_{ts}.jsonl sotto data/features.
    """
    meta_p = session_dir / "meta.json"
    if not meta_p.exists():
        return None, "meta.json non trovato nella sessione"

    try:
        meta = json.loads(meta_p.read_text())
    except Exception as e:
        return None, f"meta.json invalido: {e}"

    ts = meta.get("ts_utc") or session_dir.name.replace("session_","")
    band = str(meta.get("band") or session_dir.parts[-3])
    drone_id = str(meta.get("drone_id") or session_dir.parts[-2])

    feat_file = DATA / "features" / band / drone_id / f"session_{ts}.jsonl"
    if not feat_file.exists():
        # fallback: cerca ovunque
        cand = list((DATA/"features").rglob(f"session_{ts}.jsonl"))
        if not cand:
            return None, f"feature file non trovato per ts={ts}"
        feat_file = cand[0]

    try:
        row = json.loads(feat_file.read_text().splitlines()[0])
    except Exception as e:
        return None, f"feature file illeggibile: {e}"

    return row, None

def vectorize(row: dict, feature_names):
    """
    Ritorna sempre (X, err). Se mancano/sono non numeriche, err descrive quali.
    """
    vals, missing = [], []
    for k in feature_names:
        if k in row:
            try:
                vals.append(float(row[k]))
            except Exception:
                missing.append(k)
        else:
            missing.append(k)
    if missing:
        return None, f"feature mancanti o non numeriche: {missing}"
    return np.array([vals]), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--model", default=str(APP_ROOT/"models/rfscan_current.pkl"))
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    sess = pathlib.Path(args.session_dir)
    if not sess.exists():
        print(json.dumps({"ok": False, "error": f"session_dir non esiste: {sess}"}))
        sys.exit(0)

    # carica modello
    try:
        model_obj = joblib.load(args.model)  # {model, features}
        clf, feats = model_obj["model"], model_obj["features"]
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"modello non disponibile o illeggibile: {e}", "model": args.model}))
        sys.exit(0)

    # carica feature
    row, err = load_last_feature(sess)
    if err:
        print(json.dumps({"ok": False, "error": err, "session_dir": str(sess)}))
        sys.exit(0)

    # vettorizza
    X, err = vectorize(row, feats)
    if err:
        print(json.dumps({"ok": False, "error": err, "have_keys": list(row.keys()), "need": feats}))
        sys.exit(0)

    # predici (con probabilità se disponibili)
    try:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)
            ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:args.topk]
            out = [{"label": c, "prob": float(p)} for c, p in ranking]
        else:
            pred = clf.predict(X)[0]
            out = [{"label": str(pred), "prob": None}]
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"errore in predizione: {e}"}))
        sys.exit(0)

    print(json.dumps({
        "ok": True,
        "session_dir": str(sess),
        "model": args.model,
        "features_used": feats,
        "result": out
    }))
    sys.exit(0)

if __name__ == "__main__":
    main()
