# capability_classifier.py
# Adattato alla struttura CRPC:
# - legge PSD dall'ultimo CSV RFE: /tmp/rfe/scan/latest_24.csv|52|58
# - memoria bande per compensare la scansione a "slot" dell'RF Explorer
# - classificazione capability-driven (banda + vts_bw stimata) con isteresi
# - contatore "multi‑picchi" leggero per OcuSync (opzionale)

import time
from math import log

# --- capability matrix per famiglie ---

#
# mu
# È la larghezza di banda video media tipica (MHz) che ci aspettiamo per quella famiglia.
# O4 → ~30 MHz (perché OcuSync 4 di solito lavora intorno a 30–40)
# O2/3 → ~16 MHz (tipico intorno a 15–20)
# Hubsan-like → ~14 MHz
# Wi-Fi fallback → ~20 MHz
#
# sig
# È la deviazione standard “attesa” (σ) per quella larghezza di banda.
# Serve per calcolare lo scostamento z = (bw_obs – mu)/σ.
# In pratica dice quanto sei “tollerante” a variazioni di banda:
# Valore più piccolo ⇒ famiglia “stretta”: se l’osservato si discosta, penalizzi molto.
# Valore più grande ⇒ famiglia “larga”: accetti oscillazioni più ampie.
#
# peak
# È la larghezza massima plausibile (MHz) che quel protocollo può raggiungere.
# Se l’osservato va oltre, scatta una penalità (P_peak).
# Esempi:
# O4 può arrivare a 40–45 MHz → peak=48
# O2/3 si ferma a 20–22 MHz → peak=24
# Hubsan sotto i 20–22 MHz → peak=22
# Wi-Fi può occupare canali 40 MHz → peak=40

FAMS = {
    # O4 (Mini 4 Pro, Air 3S, Avata 2, ecc.)
    "O4":     dict(s24=True, s52=True, s58=True, mu=30.0, sig=8.0,  peak=48.0, prior=0.35),
    # O2/O3 (Avata 1, Air 2/2S, ecc.)
    "O23":    dict(s24=True, s52=False, s58=True, mu=12.0, sig=4.0,  peak=24.0, prior=0.30),
    # AUTEL Skylink 2/3.0)
    "AUTEL":  dict(s24=True, s52=True, s58=True, mu=20.0, sig=4.0,  peak=42.0, prior=0.27),
    # Hubsan‑like (5.2 presente ma BW ≤ ~10 tipico)
    "HUBSAN": dict(s24=True, s52=False, s58=True, mu=9.0, sig=3.0,  peak=11.0, prior=0.10),
    # FPV Analogico ( Solo 5.8 BW di 3-4 MHz tipico )
    "ANALOG":   dict(s24=False, s52=False, s58=True, mu=4.0, sig=4.0,  peak=6.0, prior=0.25),
    # Video WiFi 5GHz - Droni cinesi ( Solo 5.8 BW di 10 MHz tipico )
    "5GVIDEO":   dict(s24=False, s52=False, s58=True, mu=10.0, sig=3.0,  peak=12.0, prior=0.15),
    # Fallback Wi‑Fi (non drone)
    "WIFI":   dict(s24=True, s52=False, s58=True, mu=20.0, sig=6.0,  peak=40.0, prior=0.25),
}

def _robust_p95(xs):
    if not xs: return 0.0
    sx = sorted(xs)
    k = int(0.95*(len(sx)-1))
    return float(sx[k])

NEG_MISS_K = 1.0  # tuning: 0.7–1.5

def _score_family(bw_p95, bw_max, bands_seen, fam):
    s24, s52, s58 = fam["s24"], fam["s52"], fam["s58"]
    mu, sig, peak, prior = fam["mu"], max(fam["sig"],1.0), fam["peak"], max(fam["prior"],1e-3)
    penalty = 0.0
    # hard-penalty: hai visto una banda che la famiglia NON supporta
    if ("24" in bands_seen) and not s24: penalty += 1000
    if ("52" in bands_seen) and not s52: penalty += 1000
    if ("58" in bands_seen) and not s58: penalty += 1000

    # soft-penalty: NON hai visto bande che la famiglia si aspetta
    miss = 0
    if s52 and "52" not in bands_seen: miss += 1
    if s58 and "58" not in bands_seen: miss += 1
    penalty += NEG_MISS_K * miss

    z = (bw_p95 - mu) / sig
    L_bw = z*z
    over = max(0.0, bw_max - peak)
    P_peak = (over/4.0)**2
    bonus52 = -2.0 if ("52" in bands_seen and s52) else 0.0
    base_score = penalty + L_bw + P_peak - log(prior) + bonus52

    # --- BONUS Wi-Fi: caso solo 2.4 GHz con larghezza compatibile Wi-Fi ---
    solo24 = (set(bands_seen) == {"24"})
    if fam is FAMS["WIFI"] and solo24 and 21.0 <= bw_p95 <= 24.0:
        base_score -= 0.4   # abbassa costo Wi-Fi → più probabile

    return base_score


class BandMemory:
    """
    Ricorda quali bande sono state viste di recente (utile perché l'RFE scansiona
    58 -> 52 -> 24 in tempi diversi). Una banda resta 'vista' per ttl_s secondi.
    """
    def __init__(self, ttl_s=8.0):
        self.ttl = float(ttl_s)
        self.last = {}  # "24"/"52"/"58" -> ts

    def mark(self, band_key):
        if band_key:
            self.last[str(band_key)] = time.time()

    def get_seen(self):
        now = time.time()
        return {b for b,ts in self.last.items() if (now - ts) <= self.ttl}

class CapabilityClassifier:
    def __init__(self, margin=1.0, hold_s=2.0, fps=1.0):
        self.current = None
        self.margin = float(margin)
        self.hold_needed = max(1, int(hold_s*fps))
        self.hold_cnt = 0
        self.last_scores = None

    def classify(self, bw_series_mhz, bands_seen):
        bw_p95 = _robust_p95(bw_series_mhz)
        bw_max = max(bw_series_mhz) if bw_series_mhz else 0.0
        scores = {name: _score_family(bw_p95, bw_max, bands_seen, fam) for name,fam in FAMS.items()}
        best = min(scores, key=scores.get)
        best_s = scores[best]

        if self.current is None:
            self.current, self.hold_cnt = best, 0
        else:
            cur_s = scores[self.current]
            if best_s + self.margin < cur_s:
                self.hold_cnt += 1
                if self.hold_cnt >= self.hold_needed:
                    self.current = best
                    self.hold_cnt = 0
            else:
                self.hold_cnt = 0

        self.last_scores = scores
        return dict(family=self.current, scores=scores, bw_p95=bw_p95, bw_max=bw_max, bands_seen=sorted(bands_seen))


# --- Analisi multi‑picchi (opzionale, leggera, niente SciPy) ---
def count_multi_peaks_from_psd(freqs_mhz, psd_db, center_mhz, bw_mhz,
                               prom_db=4.0, min_spacing_mhz=None):
    """
    Conta i picchi 'prominenti' dentro la finestra video.
    prominenza approssimata e minima distanza tra picchi.
    """
    try:
        import numpy as np
    except Exception:
        return dict(n_peaks=0, peakiness=0.0)

    if freqs_mhz is None or psd_db is None or len(freqs_mhz) < 8: 
        return dict(n_peaks=0, peakiness=0.0)

    freqs = np.asarray(freqs_mhz, float)
    psd   = np.asarray(psd_db, float)
    lo = center_mhz - bw_mhz/2.0
    hi = center_mhz + bw_mhz/2.0
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m): 
        return dict(n_peaks=0, peakiness=0.0)

    f = freqs[m]
    p = psd[m] - np.median(psd[m])

    # distanza minima tra picchi (in MHz)
    if min_spacing_mhz is None:
        min_spacing_mhz = max(2.0, bw_mhz/10.0)

    n = len(f)
    n_peaks, peakiness = 0, 0.0
    peak_pos = None
    for k in range(1, n-1):
        if p[k] > p[k-1] and p[k] > p[k+1] and p[k] >= prom_db:
            if peak_pos is not None and abs(f[k] - peak_pos) < min_spacing_mhz:
                continue
            n_peaks += 1
            peakiness += float(p[k])
            peak_pos = float(f[k])

    return dict(n_peaks=int(n_peaks), peakiness=float(peakiness))


def load_latest_rfe_csv_for_band(band_key):
    """
    Ritorna (freqs_mhz, psd_dbm) dall'ultimo CSV RFE della banda richiesta.
    Atteso: symlink /tmp/rfe/scan/latest_XX.csv (XX in 24/52/58)
    """
    import os, csv
    path = f"/tmp/rfe/scan/latest_{band_key}.csv"
    if not os.path.exists(path):
        return None, None
    freqs, amps = [], []
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                fm = row.get("freq_mhz") or row.get("frequency") or row.get("freq")
                am = row.get("power_dbm") or row.get("dbm") or row.get("amp_dbm") or row.get("amp")
                try:
                    fm = float(fm); am = float(am)
                except Exception:
                    continue
                freqs.append(fm); amps.append(am)
    except Exception:
        return None, None
    return (freqs or None), (amps or None)
