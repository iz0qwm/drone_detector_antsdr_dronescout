#!/usr/bin/env python3
import json, pathlib, argparse, glob
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np, sys

FEATS = ["mean_dbm","p95_dbm","floor_dbm","crest_db","center_mhz","bw_mhz"]

def parse_lines(path):
    try:
        for line in path.read_text().splitlines():
            yield json.loads(line)
    except FileNotFoundError:
        return

def load_dataset(manifest_path):
    X, y = [], []
    if manifest_path.exists():
        for r in parse_lines(manifest_path): 
            try:
                X.append([float(r[k]) for k in FEATS])
                y.append(r["label"])
            except Exception:
                pass
    else:
        # fallback: raccogli tutti i session_*.jsonl sotto data/features
        base = pathlib.Path("/home/raffaello/apprendimento/data/features")
        files = list(base.rglob("session_*.jsonl"))
        for f in files:
            for r in parse_lines(f):
                try:
                    X.append([float(r[k]) for k in FEATS])
                    y.append(r["label"])
                except Exception:
                    pass
    return np.array(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/raffaello/apprendimento/data/manifests/dataset.jsonl")
    ap.add_argument("--out", default="/home/raffaello/apprendimento/models/rfscan_staging.pkl")
    args = ap.parse_args()

    X,y = load_dataset(pathlib.Path(args.manifest))
    if len(y) == 0:
        print(json.dumps({"ok":False,"error":"Nessun campione nel dataset. Esegui almeno una registrazione + estrazione."}))
        sys.exit(0)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X,y)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":clf,"features":FEATS}, args.out)
    print(json.dumps({"ok":True, "n":int(len(y)), "classes":sorted(list(set(y))), "model":args.out}))
    sys.exit(0)

if __name__ == "__main__":
    main()
