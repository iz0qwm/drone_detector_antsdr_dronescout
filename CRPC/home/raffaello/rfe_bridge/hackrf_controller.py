#!/usr/bin/env python3
# /home/raffaello/hackrf_controller.py
import os, json, time, subprocess, signal, sys
from pathlib import Path

# IPC trigger
TRIGGER_FIFO = Path("/tmp/hackrf_trigger.fifo")

# Mutual exclusion
LOCKFILE = Path("/tmp/hackrf_controller.lock")

# Output / tools
LOGDIR  = Path("/tmp/hackrf_captures")
RUN_SH  = Path("/home/raffaello/crpc/run_hackrf_iq.sh")  # script cattura IQ on-demand

# Symlink “sorgente live” per la waterfall
LIVE_LINK = Path("/tmp/hackrf_live.iq")

# Policy
MIN_ON_S           = 6      # durata minima cattura (s)
GLOBAL_COOLDOWN_S  = 3      # pausa tra capture back-to-back (s)

def ensure_trigger_fifo():
    if not TRIGGER_FIFO.exists():
        try:
            os.mkfifo(TRIGGER_FIFO)
        except FileExistsError:
            pass

def take_lock() -> bool:
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock():
    try:
        LOCKFILE.unlink(missing_ok=True)
    except Exception:
        pass

def band_out_path(band: str) -> Path:
    # percorso dove scrive run_hackrf_iq.sh (FIFO o file)
    return Path(f"/tmp/hackrf_{band}.iq")

def update_live_symlink(target: Path):
    """Punta /tmp/hackrf_live.iq alla sorgente (file/FIFO) della banda attiva."""
    try:
        if LIVE_LINK.is_symlink() or LIVE_LINK.exists():
            LIVE_LINK.unlink()
        os.symlink(str(target), str(LIVE_LINK))
        print(f"↪ live → {target}")
    except Exception as e:
        print("⚠️  Symlink live non aggiornato:", e, file=sys.stderr)

def run_capture(band: str, f0_mhz: float, bw_hz: int, hold_s: int):
    LOGDIR.mkdir(parents=True, exist_ok=True)

    # scegli SPS in base a bw: ~2.5x la BW, quantizzato alle velocità “classiche”
    desired = max(10_000_000, min(20_000_000, int(2.5 * bw_hz)))  # 10–20 MS/s
    choices = [10_000_000, 12_000_000, 16_000_000, 20_000_000]
    sps = min(choices, key=lambda c: abs(c - desired))

    # durata proporzionale alla BW (clip un po' più lunga per video grossi)
    duration = max(MIN_ON_S, min(12, int(max(6, (bw_hz/1e6)*0.8))))

    # Aggiorna symlink per la waterfall prima di partire
    out_path = band_out_path(band)
    update_live_symlink(out_path)

    # Assicurati che lo script sia eseguibile
    try:
        RUN_SH.chmod(RUN_SH.stat().st_mode | 0o111)
    except Exception:
        pass

    cmd = [
        str(RUN_SH),
        "--band", band,
        "--f0", f"{f0_mhz}",
        "--bw", f"{int(bw_hz)}",
        "--sps", f"{sps}",
        "--seconds", f"{duration}",
        # "--out", str(out_path),  # usa se vuoi forzare il path (lo script ha già default coerenti)
    ]

    print("▶ Avvio HackRF:", " ".join(cmd))
    # Avvia in suo process group per un SIGINT pulito
    p = subprocess.Popen(cmd, preexec_fn=os.setsid)
    t0 = time.time()
    # Attendi la durata richiesta; lo script ha già un timeout, ma qui tieni il controllo
    while True:
        if p.poll() is not None:
            break
        if time.time() - t0 >= duration:
            break
        time.sleep(0.2)

    # Stop “gentile”
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    # Chiudi entro poco, poi forza
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("⏹️  HackRF fermato.")

def main():
    print("🎛️  HackRF controller in ascolto…")
    ensure_trigger_fifo()

    # Loop lettura trigger (una riga JSON per evento)
    while True:
        # Apri in lettura “bloccante”: il writer (trigger) aprirà in write
        with open(TRIGGER_FIFO, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    band = str(evt["band"])
                    f0   = float(evt["f0_mhz"])
                    bw   = int(evt.get("bw_hz", 0))
                    hold = int(evt.get("hold_s", MIN_ON_S))
                except Exception as e:
                    print("⚠️  Trigger malformato:", e, line, file=sys.stderr)
                    continue

                if band not in ("24", "58"):
                    print("⚠️  Banda non valida nel trigger:", band, file=sys.stderr)
                    continue

                if not take_lock():
                    print("…ignoro trigger: HackRF occupato.")
                    time.sleep(GLOBAL_COOLDOWN_S)
                    continue

                try:
                    run_capture(band, f0, bw, hold)
                finally:
                    release_lock()
                    time.sleep(GLOBAL_COOLDOWN_S)

if __name__ == "__main__":
    main()
