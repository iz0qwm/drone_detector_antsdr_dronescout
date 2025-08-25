#!/usr/bin/env python3
import argparse, json, subprocess, pathlib, datetime as dt, time, os, sys, shutil, glob

APP_ROOT = pathlib.Path("/home/raffaello/apprendimento")
TILES_DAEMON = "/home/raffaello/crpc/iq_to_tiles_cmap_arg.py"
FIFO_BY_BAND = { "24":"/tmp/hackrf_24.iq", "52":"/tmp/hackrf_52.iq", "58":"/tmp/hackrf_58.iq" }
CENTER_FILE  = { "24":"/tmp/center_24.txt", "52":"/tmp/center_52.txt", "58":"/tmp/center_58.txt" }
OUT_TILES_DIR = "/tmp/tiles"
HACKRF = shutil.which("hackrf_transfer") or "/usr/bin/hackrf_transfer"
PV = shutil.which("pv")  # opzionale

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

def tiles_daemon_pids():
    r = run0(["pgrep", "-f", TILES_DAEMON])
    if r["rc"] != 0: return []
    return [int(x) for x in r["out"].split() if x.strip().isdigit()]

def kill_tiles_daemon():
    pids = tiles_daemon_pids()
    if not pids: return {"killed":0}
    run0(["bash","-lc", "kill -9 " + " ".join(str(pid) for pid in pids)])
    time.sleep(0.2)
    return {"killed":len(pids)}

def maybe_start_tiles_daemon(fs_hz: int, force_restart=True):
    """
    Avvia il renderer con GLI STESSI PARAMETRI del live CRPC.
    Forziamo il riavvio per garantire fs/parametri identici alla sessione.
    """
    info = {}
    if force_restart:
        info["kill"] = kill_tiles_daemon()
    fs_view = min(1_200_000, fs_hz)
    # Parametri allineati al profilo che hai confermato “va benissimo”
    #cmd = (
    #    f"nohup python3 {TILES_DAEMON} "
    #    f"--fs {fs_hz} --fs-view {fs_view} "
    #    f"--nfft 65536 --hop-div 8 "
    #    f"--mode EWMA --alpha 0.20 "
    #    f"--norm bgsub --bg-alpha 0.02 "
    #    f"--fast-settle-cols 120 --fast-settle-alpha 0.25 "
    #    f"--delta-floor -10 --delta-ceil 10 "
    #    f"--gamma 1.3 "
    #    f"--cmap turbo "
    #    f"--dc-notch "
    #    f">/tmp/iq_to_tiles.log 2>&1 &"
    #)
    cmd = (
        f"nohup python3 {TILES_DAEMON} "
        f"--fs 10e6 --fs-view 1.2e6 "
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
    time.sleep(0.8)
    info["start_rc"] = r["rc"]; info["start_err"] = r["err"]
    return info

def write_center_file(band: str, center_mhz: float):
    try:
        pathlib.Path(CENTER_FILE[band]).write_text(str(center_mhz*1e6))
        return True, ""
    except Exception as e:
        return False, str(e)

def copy_latest_tiles_for_session(band: str, sess_tiles_dir: pathlib.Path, ts_utc: str):
    """
    Copia live/cum correnti e, come best.png, il cum con timestamp che combacia con la sessione.
    Se non trovato, fallback all’ultimo.
    """
    sess_tiles_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    live_src = pathlib.Path(OUT_TILES_DIR)/f"{band}_live.png"
    cum_src  = pathlib.Path(OUT_TILES_DIR)/f"{band}_cum.png"
    if live_src.exists():
        shutil.copy2(live_src, sess_tiles_dir/"live.png"); copied.append("live.png")
    if cum_src.exists():
        shutil.copy2(cum_src,  sess_tiles_dir/"cum.png");  copied.append("cum.png")

    # match stampato per questa sessione
    # ts_utc es: 2025-08-24T18-07-12Z -> prefisso file: 2025-08-24_18-07-12
    pref = ts_utc.replace("T","_").replace("Z","")
    pattern = str(pathlib.Path(OUT_TILES_DIR)/f"{band}_cum_{pref}*.png")
    stamped = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not stamped:
        # fallback: ultimo disponibile (ma almeno della stessa band)
        stamped = sorted(glob.glob(str(pathlib.Path(OUT_TILES_DIR)/f"{band}_cum_*.png")), key=os.path.getmtime)
    if stamped:
        shutil.copy2(stamped[-1], sess_tiles_dir/"best.png"); copied.append("best.png")
    return copied

def pick_sample_rate(bw_mhz: float) -> int:
    sr = int(bw_mhz * 1.5e6)
    sr = max(2_000_000, min(10_000_000, sr))
    sr = int(round(sr / 200_000)) * 200_000
    return sr

def shlex_quote(s: str) -> str:  # evita import shlex
    return "'" + s.replace("'", "'\\''") + "'"

def replay_iq_rate_limited(iq_path: pathlib.Path, fifo_path: str, sr: int, log_path: pathlib.Path):
    """
    Riproduce l’IQ nella FIFO limitando la velocità a ~2*SR byte/s (I8+Q8).
    Usa pv se disponibile, altrimenti fallback Python.
    """
    try:
        if PV:
            rate = 2*sr  # bytes/s
            cmd = f"pv -q -L {rate} {shlex_quote(str(iq_path))} > {shlex_quote(fifo_path)}"
            r = run0(["bash","-lc", cmd])
            log_path.write_text(json.dumps({"mode":"pv","rc":r["rc"],"err":r["err"]}, indent=2))
            return
        # Fallback Python
        chunk = 512*1024
        bytes_per_sec = 2*sr
        t0 = time.time()
        sent = 0
        with open(iq_path, "rb") as f, open(fifo_path, "wb", buffering=0) as w:
            while True:
                buf = f.read(chunk)
                if not buf: break
                w.write(buf)
                sent += len(buf)
                elapsed = time.time() - t0
                target = sent / bytes_per_sec
                if target > elapsed:
                    time.sleep(min(0.05, target - elapsed))
        log_path.write_text(json.dumps({"mode":"python","sent":sent,"sr":sr}, indent=2))
    except Exception as e:
        log_path.write_text(json.dumps({"mode":"err","err":str(e)}, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["24","52","58"], required=True)
    ap.add_argument("--drone-id", required=True)
    ap.add_argument("--center-mhz", type=float, required=True)
    ap.add_argument("--bw-mhz", type=float, default=20.0)
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

    # 0) renderer coerente con il live (FORZA riavvio con fs corretto)
    start_info = maybe_start_tiles_daemon(sr, force_restart=True)
    okF, errF = ensure_fifo(fifo_path)
    (sess_dir/"tiles_prepare.log").write_text(json.dumps({"start_info":start_info, "fifo_ok":okF, "fifo_err":errF}, indent=2))

    # 1) acquisizione IQ
    num_samples = int(args.duration_s * sr)
    hackrf_cmd = [
        HACKRF, "-r", str(iq_path),
        "-s", str(sr),
        "-f", str(int(args.center_mhz*1e6)),
        "-n", str(num_samples)
    ]
    r1 = run0(hackrf_cmd)
    (sess_dir/"hackrf_transfer.log").write_text((r1["out"] or "") + (("\nERR:\n"+r1["err"]) if r1["err"] else ""))
    meta["hw"]["hackrf"] = "ok" if r1["rc"]==0 else f"err:{r1['rc']}"

    # 2) snapshot RFE
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

    # 3) center file per etichette
    okC, errC = write_center_file(args.band, args.center_mhz)
    if not okC:
        (sess_dir/"center_write.err").write_text(errC)

    # 4) REPLAY con rate‑limit
    if iq_path.exists() and okF:
        replay_iq_rate_limited(iq_path, fifo_path, sr, sess_dir/"tiles_replay.log")
    else:
        (sess_dir/"tiles_replay.log").write_text(json.dumps({"skip":"iq o fifo mancante","iq_exists":iq_path.exists(),"fifo_ok":okF}, indent=2))

    # 5) attendi chiusura cum “stampato” e copia PNG
    time.sleep(0.8)
    copied = copy_latest_tiles_for_session(args.band, tiles_dir, ts)
    meta["hw"]["tiles"] = "ok" if copied else "missing"
    meta["tiles_copied"] = copied
    (sess_dir/"meta.json").write_text(json.dumps(meta, indent=2))

    ok = (r1["rc"] == 0)
    print(json.dumps({"ok": ok, "session": str(sess_dir), "hackrf_rc": r1["rc"], "tiles": copied}))
    sys.exit(0)

if __name__ == "__main__":
    main()
