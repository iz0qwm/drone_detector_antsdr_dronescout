#!/usr/bin/env python3
import re, sys, time, inspect
import RFExplorer

PORT="/dev/ttyUSB0"; BAUD=500000

def matches(name):
    keys = ["select","set","use","switch","mode","main","expan","board","module"]
    s=name.lower()
    return any(k in s for k in keys)

rfe = RFExplorer.RFECommunicator()
rfe.AutoConfigure=False
rfe.GetConnectedPorts()
assert rfe.ConnectPort(PORT, BAUD), "Connect fallita"

# reset + handshake
rfe.SendCommand("r")
t0=time.time()
while rfe.IsResetEvent and time.time()-t0<7: pass
time.sleep(2.0)
rfe.SendCommand_RequestConfigData()
t0=time.time()
while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE and time.time()-t0<6:
    rfe.ProcessReceivedString(True)

print("ActiveModel:", rfe.ActiveModel)
print("\n--- RFECommunicator: metodi/attributi utili ---")
for name in dir(rfe):
    if name.startswith("_"): continue
    if matches(name):
        obj=getattr(rfe,name)
        kind="() " if callable(obj) else "= "
        print(f"{name}{kind}{'' if callable(obj) else obj!r}")

print("\n--- RFE_Common: enum e costanti ---")
for name in dir(RFExplorer.RFE_Common):
    if name.startswith("e") or "Module" in name or "Model" in name:
        print(name, getattr(RFExplorer.RFE_Common, name))

# stampiamo anche eventuale docstring del communicator
print("\nDoc RFECommunicator:\n", inspect.getdoc(RFExplorer.RFECommunicator) or "<no doc>")

rfe.Close()

