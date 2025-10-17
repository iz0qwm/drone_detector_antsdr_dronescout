#!/usr/bin/env python3
import os, time, json, math
from pathlib import Path

SCANS_DIR = Path("/tmp/rfe/scan")
STATE = Path("/tmp/crpc_logs/df_state.json")
F0_FILE = Path("/tmp/crpc_logs/last_trigger.json")   # (se presente) f0 candidato
SECTOR_STATE = Path("/tmp/rfe/sector_state.json")    # chi sta usando rfe_dual_scan.sh
ALPHA = 0.6
BEARING_OFFSET_DEG = float(os.getenv("DF_BEARING_OFFSET_DEG", "0"))  # opzionale
BAND_RANGES = {
    "24": (2400.0, 2483.6),
    "52": (5150.0, 5350.0),
    "58": (5725.0, 5875.0),
}
EPS = 1e-9

# EMA per N,E,S,O
P = {"N": None, "E": None, "S": None, "O": None}
seen = set()
sector_order = ["N","E","S","O"]

def _read_sector():
    try:
        d = json.loads(SECTOR_STATE.read_text())
        s = d.get("sector","N")
        return s if s in P else "N"
    except Exception:
        return "N"

def _read_f0():
    # Se ho un trigger recente uso quello; altrimenti prendo il picco dal CSV corrente
    try:
        d = json.loads(F0_FILE.read_text())
        return float(d.get("f0_mhz"))
    except Exception:
        return None

def _parse_csv(path: Path):
    rows = []
    try:
        with path.open() as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except: pass
    except Exception:
        return []
    return rows

def _nearest_power(rows, f0_mhz):
    if not rows or f0_mhz is None: return None
    return min(rows, key=lambda x: abs(x[0]-f0_mhz))[1]

def _write_state(bearing_deg, conf, f0, band, sector, levels_dict):
    out = {
        "ts": time.time(),
        "bearing_deg": round(bearing_deg,1) if bearing_deg is not None else None,
        "confidence": round(conf,3) if conf is not None else None,
        "levels": {k: (None if v is None else round(v,1)) for k,v in levels_dict.items()},
        "f0_mhz": (None if f0 is None else round(f0,3)),
        "band": band,
        "sector": sector,      # ora coerente con bearing/livello massimo
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(out))

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10.0)

def _sector_from_bearing(bearing_deg):
    """Restituisce N/E/S/O coerente con il bearing (0°=Nord, 90°=Est)."""
    if bearing_deg is None:
        return "N"
    a = bearing_deg % 360.0
    if (315 <= a < 360) or (0 <= a < 45):
        return "N"
    elif 45 <= a < 135:
        return "E"
    elif 135 <= a < 225:
        return "S"
    else:
        return "O"


def _band_of_freq(f_mhz):
    if f_mhz is None:
        return None
    for b,(lo,hi) in BAND_RANGES.items():
        if lo <= f_mhz <= hi:
            return b
    return None

def _bearing_from_levels(Pdict):
    # 0°=Nord, 90°=Est, 180°=Sud, 270°=Ovest
    # θ = atan2(E-O, N-S)
    X = (Pdict["E"] - Pdict["O"])
    Y = (Pdict["N"] - Pdict["S"])
    theta_deg = (math.degrees(math.atan2(X, Y)) + 360.0) % 360.0
    return theta_deg


def main():
    print("[DF] pseudo-doppler lento avviato.")
    while True:
        # scorri le 3 bande possibili, usa il CSV più recente
        for band in ("24","52","58"):
            link = SCANS_DIR / f"latest_{band}.csv"
            if not link.is_symlink(): 
                continue
            target = link.resolve()
            sig = (target.stat().st_ino, target.stat().st_size, int(target.stat().st_mtime))
            if sig in seen: 
                continue
            seen.add(sig)
            rows = _parse_csv(target)
            if not rows: 
                continue

            f0 = _read_f0()
            # fallback: se non abbiamo f0, usa il picco dello sweep corrente
            if f0 is None and rows:
                f0 = max(rows, key=lambda x: x[1])[0]

            sector = _read_sector()     # quale settore era attivo in questo sweep
            pow_dbm = _nearest_power(rows, f0)
            if pow_dbm is None: 
                continue

            # EMA per il settore corrente
            prev = P[sector]
            P[sector] = pow_dbm if prev is None else (ALPHA*prev + (1-ALPHA)*pow_dbm)

            # se ho campioni su tutti e 4 i settori, calcolo bearing+conf
            if all(P[s] is not None for s in sector_order):
                bearing = _bearing_from_levels(P)
                sector_for_json = _sector_from_bearing(bearing)


                # usa potenze lineari per il contrasto
                vals_dbm = list(P.values())
                vals_mw = [dbm_to_mw(v) for v in vals_dbm]
                conf = (max(vals_mw) - min(vals_mw)) / (sum(vals_mw) + EPS)

                _write_state(bearing, conf, f0, band, sector_for_json, P)
                print(f"[DF] f0≈{f0:.3f} MHz band={band} → bearing={bearing:5.1f}°  conf={conf:.3f}  sec={sector_for_json}")

        time.sleep(0.2)

if __name__ == "__main__":
    main()
