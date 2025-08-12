#!/usr/bin/env python3
import time, sys, re
import RFExplorer

PORT="/dev/ttyUSB0"; BAUD=500000

def scan_names(obj, pat):
    rx=re.compile(pat, re.I)
    return [n for n in dir(obj) if rx.search(n)]

rfe = RFExplorer.RFECommunicator()
rfe.AutoConfigure = False
rfe.GetConnectedPorts()
assert rfe.ConnectPort(PORT, BAUD), "Connect fallita"

# reset + handshake come fa il tuo tool
rfe.SendCommand("r")
t0=time.time()
while rfe.IsResetEvent and time.time()-t0<7: pass
time.sleep(2.0)
rfe.SendCommand_RequestConfigData()
t0=time.time()
while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE and time.time()-t0<6:
    rfe.ProcessReceivedString(True)

print("ActiveModel:", rfe.ActiveModel)
cfg = getattr(rfe, "Configuration", None)
print("Has Configuration:", cfg is not None)

# 1) metodi della classe che “odorano” di modulo/espansione
print("\n-- RFECommunicator methods matching (Module|Expansion|Board) --")
for n in scan_names(rfe, r"(Module|Expansion|Board)"):
    attr=getattr(rfe, n)
    if callable(attr):
        print("  ", n, "()")
    else:
        print("  ", n, "=", attr)

# 2) attributi utili di Configuration (se presente)
if cfg:
    print("\n-- RFEConfiguration attrs --")
    for n in scan_names(cfg, r"."):
        if n.startswith("_"): continue
        try:
            v=getattr(cfg,n)
            if isinstance(v,(int,float,bool,str)):
                print("  ", n, "=", v)
        except: pass

# 3) Enum possibili in RFE_Common
print("\n-- RFE_Common enums (potrebbero includere moduli) --")
for n in dir(RFExplorer.RFE_Common):
    if n.startswith("e"):
        print(" ", n, getattr(RFExplorer.RFE_Common, n))

rfe.Close()

