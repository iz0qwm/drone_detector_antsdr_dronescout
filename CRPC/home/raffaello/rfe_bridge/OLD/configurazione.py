#!/usr/bin/env python3
import time, sys, inspect
import RFExplorer

PORT="/dev/ttyUSB0"; BAUD=500000

rfe = RFExplorer.RFECommunicator()
rfe.AutoConfigure = False
rfe.GetConnectedPorts()
ok = rfe.ConnectPort(PORT, BAUD)
print("Connect:", ok)

# reset + handshake come fa rfexplorerDetailedScan
rfe.SendCommand("r")
t0=time.time()
while rfe.IsResetEvent and time.time()-t0<7: pass
time.sleep(2.5)
rfe.SendCommand_RequestConfigData()
t0=time.time()
while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE and time.time()-t0<5:
    rfe.ProcessReceivedString(True)

cfg = getattr(rfe, "Configuration", None)
print("ActiveModel:", rfe.ActiveModel)
print("Has Configuration:", cfg is not None)

if cfg:
    # Elenco attributi “utili” della config
    attrs = [a for a in dir(cfg) if not a.startswith("_")]
    print("\n-- RFEConfiguration attributes --")
    for a in attrs:
        try:
            val = getattr(cfg, a)
            if isinstance(val, (int, float, bool, str)):
                print(f"{a} = {val}")
        except Exception as e:
            pass

    # Prova a stampare anche RFExplorer.RFE_Common per enum/moduli
    print("\n-- RFE_Common enums present --")
    for a in dir(RFExplorer.RFE_Common):
        if a.startswith("e"):
            print(a, "=>", getattr(RFExplorer.RFE_Common, a))

rfe.Close()

