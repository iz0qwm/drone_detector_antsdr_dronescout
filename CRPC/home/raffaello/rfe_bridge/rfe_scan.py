#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, sys, argparse, math
import RFExplorer

PORT = "/dev/ttyUSB0"
BAUD = 500000

# Frame binari per cambio modulo (come nel tuo script originale)
CM_6G    = b'#\x05CM\x00'  # mainboard 6G (usato per 58 nel tuo file)
CM_WSUB3 = b'#\x05CM\x01'  # expansion WSUB3G (usato per 24 nel tuo file)

def wait_active_model(rfe, timeout=6.0):
    t0 = time.time()
    while time.time()-t0 < timeout:
        rfe.ProcessReceivedString(True)
        if rfe.ActiveModel != RFExplorer.RFE_Common.eModel.MODEL_NONE:
            return True
        time.sleep(0.02)
    return False

def wait_sweep_start(rfe, start_mhz, timeout=8.0, tol=0.6):
    """Ritorna l’ultimo sweep con StartFrequencyMHZ≈start_mhz oppure None."""
    t0 = time.time()
    last_count = -1
    while time.time()-t0 < timeout:
        rfe.ProcessReceivedString(True)
        count = rfe.SweepData.Count
        if count > 0 and count != last_count:
            last_count = count
            sw = rfe.SweepData.GetData(count - 1)
            try:
                if abs(float(sw.StartFrequencyMHZ) - float(start_mhz)) <= tol:
                    return sw
            except Exception:
                pass
        time.sleep(0.03)
    return None

def _get_amp_table(rfe):
    """Prova a recuperare una amplitude table; se non disponibile crea un placeholder."""
    # Tentativo: rfe.Configuration.AmplitudeTable
    try:
        cfg = getattr(rfe, "Configuration", None)
        if cfg is not None and hasattr(cfg, "AmplitudeTable") and cfg.AmplitudeTable is not None:
            return cfg.AmplitudeTable
    except Exception:
        pass
    # Fallback: istanzia una tabella vuota (nessuna correzione)
    try:
        ATmod = getattr(RFExplorer, "RFEAmplitudeTableData", None)
        if ATmod is not None:
            return ATmod.RFEAmplitudeTableData()
    except Exception:
        pass
    return None

def _total_points(sw):
    for name in ("TotalSteps", "DataPoints", "DataPointCount"):
        if hasattr(sw, name):
            try:
                val = int(getattr(sw, name))
                if val > 0:
                    return val
            except Exception:
                pass
    # Ultimo tentativo: cerca array interni
    for name in ("AmplitudeData", "m_arrAmplitudeDataDBM", "arrAmplitudeDataDBM"):
        if hasattr(sw, name):
            try:
                return len(getattr(sw, name))
            except Exception:
                pass
    # Default RF Explorer ha 112 punti per sweep, ma mettiamo un fallback safe
    return 112

def _freq_step(sw, total):
    # Preferisci StepFrequencyMHZ; se non c'è, ricava da Stop/Start
    try:
        step = float(sw.StepFrequencyMHZ)
        if step > 0:
            return step
    except Exception:
        pass
    try:
        start = float(sw.StartFrequencyMHZ)
        stop  = float(sw.StopFrequencyMHZ)
        if total > 1:
            return (stop - start) / (total - 1)
    except Exception:
        pass
    # fallback prudente
    return 0.0

def _get_amp_accessor(sw, rfe):
    """Restituisce una callable(i)->dBm con fallback multipli."""
    amp_table = _get_amp_table(rfe)

    # 1) Metodo canonico a 3 argomenti
    def try_get_amp3(i, use_corr=True):
        return float(sw.GetAmplitudeDBM(i, amp_table, bool(use_corr)))

    # 2) Metodo a 2 argomenti (alcune build)
    def try_get_amp2(i, use_corr=True):
        return float(sw.GetAmplitudeDBM(i, amp_table))

    # 3) Metodo a 1 argomento (rare)
    def try_get_amp1(i, use_corr=True):
        return float(sw.GetAmplitudeDBM(i))

    # 4) Metodo senza correzione esplicito
    def try_uncorr(i):
        return float(sw.GetUncorrectedAmplitudeDBM(i))

    # 5) Array interni noti
    arr_candidates = []
    for name in ("AmplitudeData", "m_arrAmplitudeDataDBM", "arrAmplitudeDataDBM"):
        if hasattr(sw, name):
            try:
                arr = getattr(sw, name)
                _ = arr[0] if len(arr) > 0 else None
                arr_candidates.append(arr)
            except Exception:
                pass

    def from_array_factory(arr):
        def _f(i):
            return float(arr[i])
        return _f

    # Prova a stabilire quale funziona davvero con una sonda su i=0
    tests = []
    tests.append(lambda i: try_get_amp3(i, True))
    tests.append(lambda i: try_get_amp3(i, False))
    tests.append(lambda i: try_get_amp2(i, True))
    tests.append(lambda i: try_get_amp1(i, True))
    tests.append(lambda i: try_uncorr(i))
    for arr in arr_candidates:
        tests.append(from_array_factory(arr))

    # Scegli il primo che non alza eccezioni
    for fn in tests:
        try:
            _ = fn(0)
            return fn
        except Exception:
            continue

    # Ultimo fallback: restituisce NaN
    return lambda i: float("nan")

def write_csv_from_sweep(sw, rfe, out_path):
    """Scrive CSV: freq_mhz,amp_dbm con massima compatibilità tra versioni API."""
    start = float(getattr(sw, "StartFrequencyMHZ"))
    total = _total_points(sw)
    step  = _freq_step(sw, total)
    get_amp = _get_amp_accessor(sw, rfe)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("freq_mhz,amp_dbm\n")
        for i in range(int(total)):
            freq = start + step * i if step > 0 else start
            try:
                amp  = get_amp(i)
                if amp is None or (isinstance(amp, float) and math.isnan(amp)):
                    f.write(f"{freq:.6f},\n")
                else:
                    f.write(f"{freq:.6f},{amp:.2f}\n")
            except Exception:
                f.write(f"{freq:.6f},\n")

def main():
    parser = argparse.ArgumentParser(description="Switch banda RF Explorer + (opzionale) sweep→CSV")
    parser.add_argument("--band", required=True, choices=["24", "58"], help="Banda target: 24 o 58")
    parser.add_argument("--port", default=PORT, help="Porta seriale (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=BAUD, help=f"Baudrate (default: {BAUD})")
    # Opzioni SCAN (se presenti fa lo sweep e salva CSV)
    parser.add_argument("-s", "--start", type=float, help="Start MHz")
    parser.add_argument("-e", "--end",   type=float, help="End MHz")
    parser.add_argument("--out", help="CSV di output (es. /tmp/rfe/scan/latest_24.csv)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log verboso")
    args = parser.parse_args()

    # Mapping come nel tuo script originale
    if args.band == "58":
        CM = CM_6G
        DEF_START, DEF_STOP = 5700.0, 5900.0
    else:
        CM = CM_WSUB3
        DEF_START, DEF_STOP = 2400.0, 2483.0

    do_scan = (args.start is not None and args.end is not None and args.out is not None)
    START = float(args.start) if args.start is not None else DEF_START
    STOP  = float(args.end)   if args.end   is not None else DEF_STOP

    try:
        rfe = RFExplorer.RFECommunicator()
        rfe.AutoConfigure = False
        rfe.GetConnectedPorts()
        assert rfe.ConnectPort(args.port, args.baud), f"Connect fallita su {args.port}@{args.baud}"

        # Reset + handshake (come nel tuo file che “va benissimo”)
        rfe.SendCommand("r")
        t0 = time.time()
        while rfe.IsResetEvent and time.time()-t0 < 7.0:
            pass
        time.sleep(2.0)

        rfe.SendCommand_RequestConfigData()
        wait_active_model(rfe, 6.0)

        # Invia frame CM raw
        ser = rfe.m_objSerialPort
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        ser.write(CM)
        ser.flush()
        time.sleep(0.3)

        # Richiedi di nuovo la config
        rfe.SendCommand_RequestConfigData()
        wait_active_model(rfe, 3.0)

        # Imposta range
        if args.verbose:
            print(f"UpdateDeviceConfig: {START}-{STOP} MHz")
        rfe.UpdateDeviceConfig(START, STOP)
        rfe.ProcessReceivedString(True)

        # Se non voglio lo scan/CSV, mi fermo dopo aver verificato che parta uno sweep
        sw = wait_sweep_start(rfe, START, timeout=8.0)
        if not sw:
            print("ATTENZIONE: sweep non partito dallo START atteso; verifica sul device.", file=sys.stderr)
            sys.exit(2)

        if not do_scan:
            print("ActiveModel:", rfe.ActiveModel)
            print(f"OK: switch '{args.band}' + range {START}-{STOP} MHz impostato su {args.port}@{args.baud}")
            return

        # Salva CSV con fallback multipli di ampiezza
        write_csv_from_sweep(sw, rfe, args.out)
        if args.verbose:
            print(f"OK: sweep salvato → {args.out}")

    finally:
        try:
            rfe.Close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

