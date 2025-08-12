#!/usr/bin/env python3
import os, time, json, subprocess, math
from pathlib import Path
from datetime import datetime

SCANS_DIR = Path("/tmp/rfe/scan")
LOG_JSONL = Path("/tmp/crpc_logs/rfe_sweep.jsonl")   # il cruscotto può “seguire” questo
PEAK_THRESHOLD_DBM = -80.0                            # soglia picco (tuning)
MIN_FREQ_GAP_MHZ = 2.0                                # evita doppi trigger su picchi vicini
TRIGGER_HACKRF = True                                 # se False: solo logga
HACKRF_CAPTURE_SEC = 0.25                             # durata cattura
HACKRF_SAMPLERATE = 20000000                          # 20 Msps
HACKRF_IF_GAIN = 40
HACKRF_LNA_GAIN = 40
HACKRF_AMP = 1

last_triggered = {"24": [], "58": []}  # lista di freq già triggerate di recente

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def write_jsonl(obj):
    with LOG_JSONL.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_csv(path: Path):
    """
    Supporta output tipico di rfexplorerDetailedScan:
    Righe con frequenza e livello (dBm).
    Ignora linee commento/headers.
    Ritorna lista [(freq_mhz, power_dbm), ...]
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "," not in line:
                continue
            parts = [x.strip() for x in line.split(",")]
            # cerchiamo due colonne numeriche (freq MHz, dBm)
            nums = []
            for p in parts:
                try: nums.append(float(p))
                except: pass
            if len(nums) >= 2:
                freq, dbm = nums[0], nums[1]
                rows.append((freq, dbm))
    return rows

def max_peak(points):
    if not points:
        return None
    # max per potenza
    return max(points, key=lambda x: x[1])

def similar_to_any(freq, lst, tol):
    return any(abs(freq - f) < tol for f in lst)

def trigger_hackrf(center_mhz: float):
    # costruisce filename e comanda hackrf_transfer
    TS = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = Path("/tmp/rfe/captures") / f"hackrf_{center_mhz:.3f}MHz_{TS}.iq"
    # calcola numero campioni = sec * samplerate
    nsamp = int(HACKRF_SAMPLERATE * HACKRF_CAPTURE_SEC)
    cmd = [
        "hackrf_transfer",
        "-r", str(out),
        "-f", str(int(center_mhz * 1e6)),
        "-s", str(HACKRF_SAMPLERATE),
        "-n", str(nsamp),
        "-a", str(HACKRF_AMP),
        "-l", str(HACKRF_LNA_GAIN),
        "-g", str(HACKRF_IF_GAIN),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=HACKRF_CAPTURE_SEC+2)
        return str(out)
    except Exception as e:
        return f"ERROR {e}"

def process_band(band_tag: str, csv_path: Path):
    points = parse_csv(csv_path)
    if not points:
        return
    # log “spettro” leggero (puoi campionare per non esagerare)
    write_jsonl({
        "ts": now_iso(),
        "source": "RFExplorer",
        "band": band_tag,
        "type": "spectrum",
        "points": [[round(f,3), round(d,1)] for f,d in points]
    })
    # peak detection
    fpk = max_peak(points)
    if not fpk: return
    f_mhz, p_dbm = fpk
    write_jsonl({
        "ts": now_iso(),
        "source": "RFExplorer",
        "band": band_tag,
        "type": "peak",
        "freq_mhz": round(f_mhz,3),
        "power_dbm": round(p_dbm,1)
    })
    # trigger (con hysteresis su frequenze vicine)
    if p_dbm >= PEAK_THRESHOLD_DBM and not similar_to_any(f_mhz, last_triggered[band_tag], MIN_FREQ_GAP_MHZ):
        cap_path = None
        if TRIGGER_HACKRF:
            cap_path = trigger_hackrf(f_mhz)
        write_jsonl({
            "ts": now_iso(),
            "source": "RFExplorer",
            "band": band_tag,
            "type": "trigger",
            "freq_mhz": round(f_mhz,3),
            "power_dbm": round(p_dbm,1),
            "hackrf_capture": cap_path
        })
        # memorizza per evitare retrigger troppo vicini
        last_triggered[band_tag].append(f_mhz)
        # tieni lista corta
        if len(last_triggered[band_tag]) > 20:
            last_triggered[band_tag] = last_triggered[band_tag][-20:]

def main():
    print("[RFE-BRIDGE] Avvio lettore CSV → JSONL (+trigger HackRF opz.)")
    latest24 = SCANS_DIR / "latest_24.csv"
    latest58 = SCANS_DIR / "latest_58.csv"
    seen = set()  # evita di rileggere stessi file
    while True:
        for band_tag, link in (("24", latest24), ("58", latest58)):
            try:
                if link.is_symlink():
                    target = link.resolve()
                    if target not in seen and target.exists():
                        process_band(band_tag, target)
                        seen.add(target)
            except Exception as e:
                write_jsonl({
                    "ts": now_iso(),
                    "source": "RFExplorer",
                    "type": "error",
                    "band": band_tag,
                    "error": str(e),
                })
        time.sleep(0.5)

if __name__ == "__main__":
    main()

