#!/usr/bin/env python3
import time
import sys
# importa la stessa libreria usata dal tuo scanner
import RFExplorer  # type: ignore

# === incolla qui la classe DetailedScanner presa dal tuo file,
#     oppure fai un import se l'hai in un modulo separato. ===
# Per comodità, qui definisco solo i metodi che servono:
from typing import Any, Dict, TextIO, Optional

class DetailedScanner:
    def __init__(self, serialport: Optional[str], baudrate: int, verbose: bool):
        self.verbose = verbose
        self.goodState = False
        try:
            self.objRFE: Any = RFExplorer.RFECommunicator()
            self.objRFE.AutoConfigure = False
            self.objRFE.GetConnectedPorts()
        except Exception as obEx:
            print("Error:", obEx, file=sys.stderr)

        if self.objRFE.ConnectPort(serialport, baudrate):
            # Reset come nello scanner
            self.objRFE.SendCommand("r")
            while self.objRFE.IsResetEvent:
                pass
            time.sleep(3)
            # Request config e attesa ActiveModel
            self.objRFE.SendCommand_RequestConfigData()
            while self.objRFE.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE:
                self.objRFE.ProcessReceivedString(True)
            # Verifica Analyzer
            if self.objRFE.IsAnalyzer():
                self.goodState = True
        else:
            print("Not Connected", file=sys.stderr)

    def __del__(self) -> None:
        try:
            self.goodState = False
            self.objRFE.Close()
        except Exception:
            pass

    def _updateFreqRange(self, startFreq: int, stopFreq: int) -> None:
        # è esattamente il metodo del tuo tool
        self.objRFE.UpdateDeviceConfig(startFreq, stopFreq)
        self.objRFE.ProcessReceivedString(True)
        # attendo finché arriva uno sweep con lo start desiderato
        objSweep = None
        t0 = time.time()
        while True:
            self.objRFE.ProcessReceivedString(True)
            if self.objRFE.SweepData.Count > 0:
                objSweep = self.objRFE.SweepData.GetData(self.objRFE.SweepData.Count - 1)
                if objSweep.StartFrequencyMHZ == startFreq:
                    break
            if time.time() - t0 > 10:
                break

# ======== parametri ========
PORT = "/dev/ttyUSB0"
BAUD = 500000

# "58" per 5.8 GHz, "24" per 2.4 GHz
BAND = "58"
if BAND == "58":
    START, STOP = 5700, 5900   # int MHz, come nel tuo tool
else:
    START, STOP = 2400, 2483

# ======== esecuzione ========
ds = DetailedScanner(PORT, BAUD, verbose=False)
if not ds.goodState:
    print("Handshake fallito (MODEL_NONE): riprovo con attesa più lunga...", file=sys.stderr)
    # a volte serve più tempo dopo il reset
    time.sleep(2)
    ds = DetailedScanner(PORT, BAUD, verbose=False)

if ds.goodState:
    ds._updateFreqRange(START, STOP)
    print(f"OK: impostato {START}-{STOP} MHz su {PORT}@{BAUD}")
else:
    print("Errore: non sono riuscito a inizializzare l’Analyzer.", file=sys.stderr)
    sys.exit(2)

