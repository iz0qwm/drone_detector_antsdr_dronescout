#!/usr/bin/env python3
import time, sys
import RFExplorer  # stessa libreria del tuo rfexplorerDetailedScan

PORT = "/dev/ttyUSB0"
BAUD = 500000

# "58" → 5700–5900 MHz, "24" → 2400–2483 MHz (interi MHz come nel tuo tool)
BAND = "58"
if BAND == "58":
    START, STOP = 5700, 5900
else:
    START, STOP = 2400, 2483

def log(*a): print(*a, flush=True)
def die(msg, code=1):
    print("ERRORE:", msg, file=sys.stderr, flush=True)
    sys.exit(code)

try:
    log("→ Istanzio RFECommunicator()")
    rfe = RFExplorer.RFECommunicator()
    rfe.AutoConfigure = False

    log("→ GetConnectedPorts()")
    rfe.GetConnectedPorts()

    log(f"→ ConnectPort({PORT}, {BAUD})")
    if not rfe.ConnectPort(PORT, BAUD):
        die(f"ConnectPort fallita su {PORT}@{BAUD}")

    # (opzionale) pulisci i buffer se disponibile
    try:
        rfe.m_objSerialPort.reset_input_buffer()
        rfe.m_objSerialPort.reset_output_buffer()
        log("→ Buffer seriale pulito")
    except Exception as e:
        log(f"(Info) reset buffer non disponibile: {e}")

    # Vai in Spectrum Analyzer mode (alcuni fw lo gradiscono esplicito)
    try:
        log("→ SendCommand_SpectrumAnalyzerMode()")
        rfe.SendCommand_SpectrumAnalyzerMode()
        time.sleep(0.2)
    except Exception as e:
        log(f"(Info) SA mode: {e}")

    # Reset intero come fa il tuo tool
    log("→ Reset 'r' e attesa stabilizzazione")
    rfe.SendCommand("r")
    t0 = time.time()
    while rfe.IsResetEvent and time.time() - t0 < 7.0:
        pass
    time.sleep(2.5)

    # Richiedi config e attendi ActiveModel
    log("→ RequestConfig e attesa ActiveModel")
    rfe.SendCommand_RequestConfigData()
    t0 = time.time()
    while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE and time.time() - t0 < 5.0:
        rfe.ProcessReceivedString(True)
    log("ActiveModel:", rfe.ActiveModel)

    if not rfe.IsAnalyzer():
        die("Il dispositivo connesso NON è uno Spectrum Analyzer")

    # Logga la config corrente se disponibile
    try:
        log("Model string:", rfe.ModelString)
        log("FW Version:", rfe.FirmwareVersion)
        log("SweepData.Count iniziale:", rfe.SweepData.Count)
    except Exception as e:
        log(f"(Info) dettagli modello non disponibili: {e}")

    # Imposta START/STOP
    log(f"→ UpdateDeviceConfig({START}, {STOP})")
    rfe.UpdateDeviceConfig(START, STOP)
    rfe.ProcessReceivedString(True)

    # Attendi che arrivi uno sweep con StartFrequency == START
    log("→ Attendo sweep con StartFrequency ==", START)
    objSweep = None
    t0 = time.time()
    while time.time() - t0 < 8.0:
        rfe.ProcessReceivedString(True)
        if rfe.SweepData.Count > 0:
            objSweep = rfe.SweepData.GetData(rfe.SweepData.Count - 1)
            log("  Sweep:", "steps=", objSweep.TotalSteps,
                "start=", objSweep.StartFrequencyMHZ,
                "stop=", objSweep.GetFrequencyMHZ(objSweep.TotalSteps-1))
            if objSweep.StartFrequencyMHZ == START:
                log("✓ Riconfigurazione avvenuta.")
                break
        time.sleep(0.1)

    if not objSweep or objSweep.StartFrequencyMHZ != START:
        die("Non ho ricevuto sweep col START atteso (il modulo potrebbe NON accettare lo switch via seriale).", 2)

    log(f"OK: banda impostata {START}-{STOP} MHz su {PORT}@{BAUD}")

except Exception as e:
    die(f"Eccezione: {e}")
finally:
    try:
        rfe.Close()
    except Exception:
        pass

