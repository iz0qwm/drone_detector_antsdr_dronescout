#!/usr/bin/env python3
import argparse, json, subprocess, pathlib, datetime as dt, time, os, sys, shutil, glob

APP_ROOT = pathlib.Path("/home/raffaello/apprendimento")
TILES_DAEMON = "/home/raffaello/crpc/iq_to_tiles_cmap_arg.py"
FIFO_BY_BAND = { "24":"/tmp/hackrf_24.iq", "52":"/tmp/hackrf_52.iq", "58":"/tmp/hackrf_58.iq" }
CENTER_FILE  = { "24":"/tmp/center_24.txt", "52":"/tmp/center_52.txt", "58":"/tmp/center_58.txt" }
OUT_TILES_DIR = "/tmp/tiles"  # dove il tuo script salva *_live.png / *_cum.png
HACKRF = shutil.which("hackrf_transfer") or "/usr/bin/hackrf_transfer"

def run0(cmd, timeout=None, shell=False):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout, shell=shell)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
    except Exception as e:
        return {"rc": 127, "out": "", "err": f"{type(e).__name__}: {e}"}

def ensure_fifo(path: str):
    p = pathlib.Path(path)
    try:
        if p.exists():
            if not p.is_fifo():
                p.unlink(missing_ok=True)
                os.mkfifo(path, 0o666)
        else:
            os.mkfifo(path, 0o666)
        return True, ""
    except Exception as e:
        return False, str(e)

def tiles_daemon_running() -> bool:
    # c'è già un processo del tuo script che gira?
    r = run0(["pgrep", "-f", TILES_DAEMON])
    return r["rc"] == 0

def maybe_start_tiles_daemon(fs_hz: int):
    """Avvia il tuo script se non è già in esecuzione. Usa fs==sample-rate della registrazione."""
    if tiles_daemon_running():
        return {"started": False, "note": "daemon già attivo"}
    # avvio con parametri ragionevoli; lo script apre tutte e 3 le FIFO
    #fs_view = min(1_200_000, fs_hz)
    #cmd = f"nohup python3 {TILES_DAEMON} --fs {fs_hz} --fs-view {fs_view} --nfft 65536 --hop-div 8 --mode EWMA --gamma 1.3 --dc-notch >/tmp/iq_to_tiles.log 2>&1 &"
    fs_view = min(1_200_000, fs_hz)
    cmd = (
    f"nohup python3 {TILES_DAEMON} "
    f"--fs {fs_hz} --fs-view {fs_view} "
    f"--nfft 65536 --hop-div 8 "
    f"--mode EWMA --alpha 0.20 "
    f"--norm bgsub --bg-alpha 0.02 "
    f"--fast-settle-cols 120 --fast-settle-alpha 0.25 "
    f"--delta-floor -10 --delta-ceil 10 "
    f"--gamma 1.3 "
    f"--cmap turbo "
    f"--dc-notch "
    f">/tmp/iq_to_tiles.log 2>&1 &"
    )

    
    r = run0(cmd, shell=True)
    time.sleep(0.6)  # piccolo grace per inizializzare threads/FIFO
    return {"started": True, "rc": r["rc"], "err": r["err"]}

def write_center_file(band: str, center_mhz: float):
    try:
        pathlib.Path(CENTER_FILE[band]).write_text(str(center_mhz*1e6))
        return True, ""
    except Exception as e:
        return False, str(e)

def copy_latest_tiles(band: str, sess_tiles_dir: pathlib.Path):
    """Copia live/cum correnti e il più recente 'timbrato' come best.png; ignora se mancanti."""
    sess_tiles_dir.mkdir(parents=True, exist_ok=True)
    live_src = pathlib.Path(OUT_TILES_DIR)/f"{band}_live.png"
    cum_src  = pathlib.Path(OUT_TILES_DIR)/f"{band}_cum.png"
    copied = []
    if live_src.exists():
        shutil.copy2(live_src, sess_tiles_dir/"live.png"); copied.append("live.png")
    if cum_src.exists():
        shutil.copy2(cum_src,  sess_tiles_dir/"cum.png");  copied.append("cum.png")
    stamped = sorted(glob.glob(str(pathlib.Path(OUT_TILES_DIR)/f"{band}_cum_*.png")), key=os.path.getmtime)
    if stamped:
        shutil.copy2(stamped[-1], sess_tiles_dir/"best.png"); copied.append("best.png")
    return copied

def pick_sample_rate(bw_mhz: float) -> int:
    # HackRF: meglio 8–10 MS/s; minimo 2 MS/s
    sr = int(bw_mhz * 1.5e6)        # un po' > BW
    sr = max(2_000_000, min(10_000_000, sr))
    # arrotonda a 200 kS/s (facilita PLL)
    sr = int(round(sr / 200_000)) * 200_000
    return sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["24","52","58"], required=True)
    ap.add_argument("--drone-id", required=True)
    ap.add_argument("--center-mhz", type=float, required=True)
    ap.add_argument("--bw-mhz", type=float, default=20.0)    # usiamo SR ≈ BW
    ap.add_argument("--duration-s", type=int, default=15)
    ap.add_argument("--out-root", default=str(APP_ROOT/"data"))
    args = ap.parse_args()

    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    sess_dir = pathlib.Path(args.out_root)/"recordings"/args.band/args.drone_id/f"session_{ts}"
    tiles_dir = sess_dir/"tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    sr = pick_sample_rate(args.bw_mhz)
    iq_path = sess_dir / f"iq_{args.center_mhz:.3f}MHz_{args.bw_mhz:.1f}MHz.iq"
    fifo_path = FIFO_BY_BAND[args.band]

    # META iniziale
    meta = {
        "band": args.band,
        "center_mhz": args.center_mhz,
        "bw_mhz": args.bw_mhz,
        "ts_utc": ts,
        "drone_id": args.drone_id,
        "hw": {"hackrf":"pending","rfe":"pending","tiles":"pending"},
        "paths": {"iq": str(iq_path), "tiles": str(tiles_dir)},
    }
    (sess_dir/"meta.json").write_text(json.dumps(meta, indent=2))

    # 0) assicura daemon tiles e FIFO
    _start_info = maybe_start_tiles_daemon(sr)
    okF, errF = ensure_fifo(fifo_path)
    (sess_dir/"tiles_prepare.log").write_text(json.dumps({"start_info":_start_info, "fifo_ok":okF, "fifo_err":errF}, indent=2))

    # 1) acquisizione HackRF su FILE
    num_samples = int(args.duration_s * sr)
    hackrf_cmd = [
        HACKRF,
        "-r", str(iq_path),
        "-s", str(sr),
        "-f", str(int(args.center_mhz*1e6)),
        "-n", str(num_samples)
    ]
    r1 = run0(hackrf_cmd)
    (sess_dir/"hackrf_transfer.log").write_text((r1["out"] or "") + (("\nERR:\n"+r1["err"]) if r1["err"] else ""))
    meta["hw"]["hackrf"] = "ok" if r1["rc"]==0 else f"err:{r1['rc']}"

    # 2) snapshot RFE se presente
    rfe_src = pathlib.Path(f"/tmp/rfe/scan/latest_{args.band}.csv")
    if rfe_src.exists():
        rfe_dst = sess_dir / f"rfe_{args.band}_{ts}.csv"
        try:
            rfe_dst.write_bytes(rfe_src.read_bytes())
            meta["hw"]["rfe"] = "snapshot"
        except Exception as e:
            meta["hw"]["rfe"] = f"err:{e}"
    else:
        meta["hw"]["rfe"] = "missing"

    # 3) scrivi la center frequency per le etichette del tuo renderer
    okC, errC = write_center_file(args.band, args.center_mhz)
    if not okC:
        (sess_dir/"center_write.err").write_text(errC)

    # 4) REPLAY: cat dell’IQ verso la FIFO della band per generare le PNG
    # (bloccante per la durata del copy; va bene)
    if iq_path.exists() and okF:
        # Per evitare saturazioni, puoi aggiungere 'pv -L' se installato; qui usiamo 'cat' semplice.
        r_cat = run0(["bash","-lc", f"cat {shlex_quote(str(iq_path))} > {shlex_quote(fifo_path)}"], timeout=None)
        (sess_dir/"tiles_replay.log").write_text(json.dumps(r_cat, indent=2))
    else:
        (sess_dir/"tiles_replay.log").write_text(json.dumps({"skip":"iq o fifo mancante","iq_exists":iq_path.exists(),"fifo_ok":okF}, indent=2))

    # 5) copia le immagini prodotte dal daemon nelle tiles della sessione
    copied = copy_latest_tiles(args.band, tiles_dir)
    meta["hw"]["tiles"] = "ok" if copied else "missing"
    meta["tiles_copied"] = copied
    (sess_dir/"meta.json").write_text(json.dumps(meta, indent=2))

    ok = (r1["rc"] == 0)
    print(json.dumps({"ok": ok, "session": str(sess_dir), "hackrf_rc": r1["rc"], "tiles": copied}))
    sys.exit(0)

# --- util per quoting in shell ---
def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"

if __name__ == "__main__":
    main()
