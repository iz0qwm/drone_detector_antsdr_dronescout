#!/usr/bin/env python3
import csv, math
from pathlib import Path

# Path del CSV
FPRINT_DB = Path("/home/raffaello/dataset/rf_fingerprint/fingerprint_db_full.csv")

# Colonne candidate
CAND_FREQ  = ["center_freq_mhz", "center_freq", "f_center_mhz", "freq_mhz", "f_mhz"]
CAND_BW    = ["bandwidth_mhz", "bandwidth", "bw_mhz", "bw", "vts_bw"]
CAND_HOP   = ["hop_rate_mhz_s", "hop_rate", "hop_mhz_s", "hop", "hop_rate_hz", "hop_rate_khz", "fhsdt"]
CAND_LABEL = ["class_name", "label", "proto", "class", "name", "type"]

if not FPRINT_DB.exists():
    print(f"❌ File non trovato: {FPRINT_DB}")
    exit(1)

with FPRINT_DB.open("r", newline="", encoding="utf-8", errors="ignore") as f:
    r = csv.DictReader(f)
    headers = [h.strip() for h in (r.fieldnames or [])]
    print(f"📄 Header ({len(headers)} colonne): {headers}")

    def pick(cols):
        for c in cols:
            if c in headers:
                return c
        return None

    kf  = pick(CAND_FREQ)
    kbw = pick(CAND_BW)
    kh  = pick(CAND_HOP)
    kl  = pick(CAND_LABEL)

    print(f"🔍 Colonna frequenza: {kf}")
    print(f"🔍 Colonna banda    : {kbw}")
    print(f"🔍 Colonna hop      : {kh}")
    print(f"🔍 Colonna label    : {kl}")

    rows = []
    for i, row in enumerate(r, start=1):
        try:
            fmhz = float(row.get(kf, "nan"))
            bw   = float(row.get(kbw, "nan"))
            hopv = row.get(kh, None)
            hop  = float(hopv) if hopv not in (None, "", "nan") else 0.0
            label= (row.get(kl) if kl else "unknown") or "unknown"
            if math.isnan(fmhz) or math.isnan(bw):
                continue
            band_guess = "24" if 2400 <= fmhz <= 2500 else ("58" if 5725 <= fmhz <= 5875 else None)
            rows.append({"band": band_guess, "f": fmhz, "bw": bw, "hop": hop, "label": label})
        except Exception as e:
            print(f"⚠️ Riga {i} saltata: {e}")

    print(f"✅ Righe valide lette: {len(rows)}")
    if rows:
        print("📌 Prime 3 righe:")
        for r in rows[:3]:
            print(r)

