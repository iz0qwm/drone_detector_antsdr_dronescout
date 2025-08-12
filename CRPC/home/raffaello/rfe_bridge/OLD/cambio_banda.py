#!/usr/bin/env python3
import time, sys, io, contextlib
import RFExplorer  # stessa lib dello scanner

PORT = "/dev/ttyUSB0"
BAUD = 500000

# "58" → 5700–5900 MHz, "24" → 2400–2483 MHz (interi in MHz)
BAND = "58"
if BAND == "58":
    START, STOP = 5700, 5900
else:
    START, STOP = 2400, 2483

def eprint(*a, **k): print(*a, file=sys.stderr, **k)

# stdout/stderr “muto” per zittire la lib
_sink = io.StringIO()
mute = contextlib.redirect_stdout(_sink)
mute_err = contextlib.redirect_stderr(_sink)

try:
    with mute, mute_err:
        rfe = RFExplorer.RFECommunicator()
        rfe.AutoConfigure = False
        rfe.GetConnectedPorts()

        if not rfe.ConnectPort(PORT, BAUD):
            raise RuntimeError(f"Non connesso a {PORT}@{BAUD}")

        # Pulisci buffer seriale (se esposto)
        try:
            rfe.m_objSerialPort.reset_input_buffer()
            rfe.m_objSerialPort.reset_output_buffer()
        except Exception:
            pass

        # Reset completo come fa lo scanner
        rfe.SendCommand("r")
        t0 = time.time()
        while rfe.IsResetEvent and time.time() - t0 < 5.0:
            pass
        time.sleep(2.5)

        # Richiedi config e attendi modello
        rfe.SendCommand_RequestConfigData()
        while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE:
            rfe.ProcessReceivedString(True)

        if not rfe.IsAnalyzer():
            raise RuntimeError("Il dispositivo non è un Spectrum Analyzer")

        # Set START/STOP → forza il modulo
        rfe.UpdateDeviceConfig(START, STOP)
        rfe.ProcessReceivedString(True)

        # Attendi finché l’ultimo sweep ha StartFrequency == START
        objSweep = None
        t0 = time.time()
        while (objSweep is None or objSweep.StartFrequencyMHZ != START) and (time.time() - t0 < 6.0):
            rfe.ProcessReceivedString(True)
            if rfe.SweepData.Count > 0:
                objSweep = rfe.SweepData.GetData(rfe.SweepData.Count - 1)

        ok = objSweep is not None and objSweep.StartFrequencyMHZ == START

finally:
    try:
        rfe.Close()
    except Exception:
        pass

if ok:
    print(f"OK: banda impostata {START}-{STOP} MHz su {PORT}@{BAUD}")
else:
    eprint("Attenzione: non ho ricevuto sweep con START atteso; il cambio potrebbe non essere avvenuto.")
    sys.exit(2)

