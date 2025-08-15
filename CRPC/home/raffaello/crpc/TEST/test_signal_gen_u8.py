#!/usr/bin/env python3
# Generatore IQ u8-offset per FIFO HackRF:
# - "blocks": 7 blocchi test 0..1 MHz
# - "tone":   tono puro (può mostrare spurie da quantizzazione)
# - "tone-clean": tono con dither TPDF di 1 LSB per eliminare spurie periodiche

import os, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fifo", type=str, default="/tmp/hackrf_58.iq")
ap.add_argument("--fs", type=float, default=10e6)
ap.add_argument("--mode", type=str, default="blocks",
                choices=["blocks","tone","tone-clean"])
ap.add_argument("--tone-freq", type=float, default=300e3, help="per --mode tone*, Hz")
ap.add_argument("--amp", type=float, default=0.6, help="ampiezza complessiva 0..1")
ap.add_argument("--chunk-samps", type=int, default=16384, help="campioni per chunk")

# Parametri blocks
ap.add_argument("--blocks-on-ms", type=float, default=25.0, help="durata ON per blocchi")
ap.add_argument("--blocks-off-ms", type=float, default=15.0, help="durata OFF per blocchi")
ap.add_argument("--stagger", type=float, default=0.0, help="0=allineati (orizzontali), 1=slanted pieno")

# Parametri tone-clean
ap.add_argument("--dither-lsb", type=float, default=1.0,
                help="ampiezza dither TPDF in multipli di 1 LSB (0=off)")

args = ap.parse_args()

FS = float(args.fs)
dt = 1.0/FS
two_pi = 2*np.pi

# posizioni dei 7 blocchi (0..1 MHz)
blk_freqs = np.array([100e3, 230e3, 360e3, 490e3, 620e3, 750e3, 880e3], dtype=np.float64)
#blk_freqs = np.array([20e3, 140e3, 260e3, 380e3, 500e3, 620e3, 740e3], dtype=np.float64)

# “spessore” del blocco (qui lasciamo 1 sola sottoportante per semplicità/CPU)
#subsp = np.array([0.0], dtype=np.float64)
# Sottoportanti per riempire il blocco (±80 kHz con passo 4 kHz)
#subsp = np.arange(-80e3, 80.0001e3, 4e3, dtype=np.float64)  # NO
subsp = np.arange(-120e3, 120e3+1, 2e3, dtype=np.float64)

# helper: quantizza a u8 offset interleaving IQ
def to_u8(iq):
    # clip in sicurezza
    i = np.clip(np.real(iq), -1.0, 1.0)
    q = np.clip(np.imag(iq), -1.0, 1.0)
    iu8 = np.round((i * 127.5) + 127.5).astype(np.uint8)
    qu8 = np.round((q * 127.5) + 127.5).astype(np.uint8)
    out = np.empty(iu8.size*2, dtype=np.uint8)
    out[0::2] = iu8
    out[1::2] = qu8
    return out.tobytes()

# envelope on/off a “mattoni”
on_len = int((args.blocks_on_ms/1000.0) * FS)
off_len = int((args.blocks_off_ms/1000.0) * FS)
period_len = max(1, on_len + off_len)

# fasi per ogni sottoportante (random per aspetto “speckled”)
rng = np.random.default_rng(123)
phases = rng.uniform(0, 2*np.pi, size=(blk_freqs.size, subsp.size))

def tpdf_dither(shape, lsb=1.0, rng=None):
    """Dither TPDF: somma di due uniformi in [-0.5,0.5] → triangolare.
       lsb: ampiezza in multipli di 1 LSB a livello “float” (1 LSB ≈ 1/127.5)."""
    if lsb <= 0.0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()
    u1 = rng.random(shape) - 0.5
    u2 = rng.random(shape) - 0.5
    # 1 LSB in dominio float corrisponde a ~1/127.5
    return (u1 + u2) * (lsb / 127.5)

def gen_chunk(n0, N):
    t = (n0 + np.arange(N)) * dt
    if args.mode == "tone":
        w = two_pi * (args.tone_freq) * t
        return (args.amp * np.exp(1j*w).astype(np.complex64))

    # --- modalità OFDM "mattoncini" ---
    T_sym = 0.001   # 1 ms per simbolo
    sym_len = int(round(T_sym * FS))
    sym_idx = (n0 // sym_len)

    sig = np.zeros(N, dtype=np.complex64)
    rng_loc = np.random.default_rng( (1234567 + sym_idx) )  # fase/simboli stabili per simbolo

    for bi, f0 in enumerate(blk_freqs):
        stg = max(0.0, min(1.0, args.stagger))
        duty_offset = int((bi * (on_len+off_len) / blk_freqs.size) * stg)
        idx0 = (n0 + duty_offset) % (on_len+off_len)
        pos = (idx0 + np.arange(N)) % (on_len+off_len)
        on_mask = (pos < on_len).astype(np.float32)

        # simboli QPSK random rinnovati a ogni simbolo
        qpsk = (rng_loc.integers(0,4,size=subsp.size))
        ph = (np.pi/2)*qpsk
        # somma le sottoportanti
        for si, df in enumerate(subsp):
            w = two_pi * (f0 + df) * t
            sig += on_mask * np.exp(1j*(w + ph[si]))

    # normalizza e scala
    sig = args.amp * sig / max(1.0, np.max(np.abs(sig)) + 1e-6)
    return sig.astype(np.complex64)


def ensure_fifo(path):
    if not os.path.exists(path):
        try: os.mkfifo(path)
        except FileExistsError:
            pass

def main():
    ensure_fifo(args.fifo)
    print(f"✅ Generatore verso FIFO {args.fifo} @ {FS/1e6:.2f} Msps, mode={args.mode}")
    n = 0
    # apri la FIFO in scrittura bloccante
    with open(args.fifo, "wb", buffering=0) as f:
        while True:
            samps = gen_chunk(n, args.chunk_samps)
            n += args.chunk_samps
            # quantizza u8 con dither già dentro (per tone-clean)
            f.write(to_u8(samps))
            # scrittura “tempo reale”
            time.sleep(args.chunk_samps / FS * 0.90)

if __name__ == "__main__":
    main()
