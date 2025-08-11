#!/usr/bin/env python3
"""
rf_eval.py — Valutazione del classificatore RF/YOLO fuso

Input:
  1) rfscan.jsonl         (default: /tmp/crpc_logs/rfscan.jsonl)
  2) ground_truth.csv     CSV con verità a terra per track_id o per finestra temporale

Formati supportati per ground_truth.csv:
  A) Per-track (semplice):
     track_id,true_label,true_family
     - Se 'true_family' manca, viene derivata da true_label con family_from_label().

  B) Per-finestra temporale (più flessibile):
     ts_start,ts_end,true_label,true_family
     - ts_* = epoch second (float); si applica a TUTTE le predizioni con ts in [ts_start, ts_end].

Output:
  - Report testo con accuracy, precision/recall per famiglia, confusion matrix (famiglia)
  - stats.json con metriche grezze

Uso:
  python3 rf_eval.py --pred /tmp/crpc_logs/rfscan.jsonl --gt ./ground_truth.csv
"""

import json, csv, argparse, sys, math
from pathlib import Path
from collections import defaultdict, Counter

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
    return lbl

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="/tmp/crpc_logs/rfscan.jsonl", help="File predizioni JSONL")
    p.add_argument("--gt", required=True, help="Ground truth CSV")
    p.add_argument("--min_score", type=float, default=0.0, help="Considera solo predizioni con score >= min_score")
    p.add_argument("--by", choices=["family","label"], default="family", help="Dominio della confusion matrix")
    p.add_argument("--out", default="rf_eval_stats.json", help="Output JSON con metriche")
    return p.parse_args()

def load_predictions(path, min_score):
    preds = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
                from rf_eval import family_from_label  # o copia la funzione nel file, se serve
                fam = d.get("family")
                if not fam and d.get("label"):
                    d["family"] = family_from_label(d["label"])

            except Exception:
                continue
            if d.get("score", 0.0) < min_score:
                continue
            preds.append(d)
    return preds

def load_ground_truth(path):
    # prova a capire lo schema dal header
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        hdr = [h.strip() for h in (r.fieldnames or [])]
        rows = [row for row in r]

    if {"track_id","true_label"}.issubset(hdr) or {"track_id","true_family"}.issubset(hdr):
        gt_by_tid = {}
        for row in rows:
            tid = int(row["track_id"])
            tl  = (row.get("true_label") or "").strip() or None
            tf  = (row.get("true_family") or "").strip() or None
            if tf is None and tl is not None:
                tf = family_from_label(tl)
            gt_by_tid[tid] = {"true_label": tl, "true_family": tf}
        return {"mode":"per_track","gt_by_tid": gt_by_tid}

    if {"ts_start","ts_end"}.issubset(hdr):
        intervals = []
        for row in rows:
            try:
                t0 = float(row["ts_start"]); t1 = float(row["ts_end"])
            except Exception:
                continue
            tl = (row.get("true_label") or "").strip() or None
            tf = (row.get("true_family") or "").strip() or None
            if tf is None and tl is not None:
                tf = family_from_label(tl)
            intervals.append({"t0": t0, "t1": t1, "true_label": tl, "true_family": tf})
        return {"mode":"per_window","intervals": intervals}

    raise ValueError("CSV ground truth non riconosciuto. Usa schema per-track o per-finestra.")

def evaluate(preds, gt, domain="family"):
    # mappa predizioni da valutare
    y_true = []
    y_pred = []

    if gt["mode"] == "per_track":
        gmap = gt["gt_by_tid"]
        for d in preds:
            tid = int(d.get("track_id", -1))
            if tid in gmap:
                if domain == "family":
                    y_true.append(gmap[tid]["true_family"] or "UNK")
                    y_pred.append(d.get("family") or "UNK")
                else:
                    y_true.append(gmap[tid]["true_label"] or "UNK")
                    y_pred.append(d.get("label") or "UNK")
    else:
        # per-finestra: assegna in base a ts
        intervals = gt["intervals"]
        for d in preds:
            ts = float(d.get("ts", 0.0))
            matched = [w for w in intervals if (w["t0"] <= ts <= w["t1"])]
            if not matched:
                continue
            w = matched[0]  # se più d'una, prendi la prima
            if domain == "family":
                y_true.append(w["true_family"] or "UNK")
                y_pred.append(d.get("family") or "UNK")
            else:
                y_true.append(w["true_label"] or "UNK")
                y_pred.append(d.get("label") or "UNK")

    labels = sorted(set(y_true) | set(y_pred))
    cm = {a:{b:0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # accuracy
    total = len(y_true)
    correct = sum(cm[l].get(l,0) for l in labels)
    acc = correct / total if total else 0.0

    # precision/recall per classe
    prec = {}; rec = {}; support = {}
    for l in labels:
        tp = cm[l].get(l,0)
        fp = sum(cm[x].get(l,0) for x in labels if x != l)
        fn = sum(cm[l].get(x,0) for x in labels if x != l)
        prec[l] = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec[l]  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        support[l] = sum(cm[l].values())

    return {
        "domain": domain,
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": acc,
        "n_samples": total,
        "precision": prec,
        "recall": rec,
        "support": support,
    }

def main():
    args = parse_args()
    preds = load_predictions(args.pred, args.min_score)
    gt = load_ground_truth(args.gt)
    stats = evaluate(preds, gt, domain=args.by)

    # stampa leggibile
    print(f"=== RF Eval ({args.by}) ===")
    print(f"samples: {stats['n_samples']} | accuracy: {stats['accuracy']:.3f}\n")

    # header
    labels = stats["labels"]
    print("Confusion matrix:")
    header = ["true\\pred"] + labels
    print("\t".join(header))
    for t in labels:
        row = [t] + [str(stats["confusion_matrix"][t][p]) for p in labels]
        print("\t".join(row))
    print("\nPrecision/Recall:")
    for l in labels:
        print(f"- {l:>10s}  prec={stats['precision'][l]:.3f}  rec={stats['recall'][l]:.3f}  support={stats['support'][l]}")

    # salva JSON
    out = Path(args.out)
    out.write_text(json.dumps(stats, indent=2))
    print(f"\n📁 Salvato: {out.resolve()}")

if __name__ == "__main__":
    main()