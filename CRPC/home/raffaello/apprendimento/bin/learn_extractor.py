#!/usr/bin/env python3
import json, argparse, pathlib, numpy as np, sys

def psd_stats(iq_path, sr):
    raw = np.fromfile(iq_path, dtype=np.int8)
    if raw.size < 2:
        return None
    I, Q = raw[0::2].astype(np.float32), raw[1::2].astype(np.float32)
    x = (I + 1j*Q) / 128.0
    nfft = 65536 if x.size >= 65536 else 1<<int(np.ceil(np.log2(max(1024, x.size))))
    S = np.abs(np.fft.fftshift(np.fft.fft(x, nfft)))**2
    S_db = 10*np.log10(S+1e-12)
    mean_db, p95_db = float(S_db.mean()), float(np.percentile(S_db,95))
    floor_db = float(np.percentile(S_db, 10))
    crest_db = p95_db - floor_db
    return dict(mean_dbm=mean_db, p95_dbm=p95_db, floor_dbm=floor_db, crest_db=crest_db)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    d = pathlib.Path(args.session_dir)
    meta_p = d / "meta.json"
    if not meta_p.exists():
        print(json.dumps({"ok":False,"error":"meta.json assente","session_dir":str(d)}))
        sys.exit(0)

    meta = json.loads(meta_p.read_text())
    iq = next(d.glob("iq_*.iq"), None)
    if iq is None:
        print(json.dumps({"ok":False,"error":"file IQ mancante","session_dir":str(d)}))
        sys.exit(0)

    sr = int(float(meta["bw_mhz"]) * 1e6)
    f = psd_stats(iq, sr)
    if f is None:
        print(json.dumps({"ok":False,"error":"IQ vuoto o troppo corto"})); sys.exit(0)

    vec = {
        "band": meta["band"],
        "center_mhz": meta["center_mhz"],
        "bw_mhz": meta["bw_mhz"],
        "label": args.label,
        "drone_id": meta["drone_id"],
        **f
    }
    out_dir = pathlib.Path("/home/raffaello/apprendimento/data/features")/meta["band"]/meta["drone_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    of = out_dir/f"session_{meta['ts_utc']}.jsonl"
    of.write_text(json.dumps(vec)+"\n")

    man = pathlib.Path("/home/raffaello/apprendimento/data/manifests/dataset.jsonl")
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("a") as fh: fh.write(json.dumps(vec)+"\n")

    print(json.dumps({"ok":True,"feature":vec,"feature_file":str(of),"manifest":str(man)}))
    sys.exit(0)

if __name__ == "__main__":
    main()
