#!/usr/bin/env python3
import joblib, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "/home/raffaello/apprendimento/models/rfscan_current.pkl"
dst = sys.argv[2] if len(sys.argv) > 2 else "/home/raffaello/apprendimento/models/rfscan_pure.pkl"

print(f"[unwrap_model] loading {src}")
obj = joblib.load(src)

est = None
if isinstance(obj, dict):
    # prova chiavi note
    for k in ("clf", "model", "pipeline", "estimator"):
        if k in obj and (hasattr(obj[k], "predict") or hasattr(obj[k], "predict_proba")):
            est = obj[k]
            print(f"[unwrap_model] found estimator under key '{k}'")
            break
    # fallback: primo valore che ha predict
    if est is None:
        for k,v in obj.items():
            if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                est = v
                print(f"[unwrap_model] found estimator under key '{k}' (fallback)")
                break
else:
    est = obj

if est is None:
    print("[unwrap_model] ERROR: no estimator found")
    sys.exit(1)

joblib.dump(est, dst)
print(f"[unwrap_model] saved pure estimator to {dst}")
