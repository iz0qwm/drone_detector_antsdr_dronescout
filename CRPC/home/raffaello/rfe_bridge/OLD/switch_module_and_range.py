#!/usr/bin/env python3
import time, sys
import RFExplorer

PORT = "/dev/ttyUSB0"
BAUD = 500000
BAND = "58"  # "58" => mainboard 6G (5.7–5.9), "24" => expansion WSUB3G (2.4–2.483)

if BAND == "58":
    TARGET_MODEL = RFExplorer.RFE_Common.eModel.MODEL_6G
    START, STOP = 5700, 5900
else:
    TARGET_MODEL = RFExplorer.RFE_Common.eModel.MODEL_WSUB3G
    START, STOP = 240, 2483  # NB: qui puoi usare 2400–2483 se preferisci in MHz interi

def wait_active_model(rfe, timeout=6.0):
    t0 = time.time()
    while rfe.ActiveModel == RFExplorer.RFE_Common.eModel.MODEL_NONE and time.time()-t0 < timeout:
        rfe.ProcessReceivedString(True)

def wait_sweep_start(rfe, start_mhz, timeout=8.0):
    t0 = time.time()
    while time.time()-t0 < timeout:
        rfe.ProcessReceivedString(True)
        if rfe.SweepData.Count > 0:
            sw = rfe.SweepData.GetData(rfe.SweepData.Count - 1)
            if sw.StartFrequencyMHZ == start_mhz:
                return True
        time.sleep(0.05)
    return False

rfe = RFExplorer.RFECommunicator()
rfe.AutoConfigure = False
rfe.GetConnectedPorts()
assert rfe.ConnectPort(PORT, BAUD), "Connect fallita"

# reset + handshake come il tuo tool
rfe.SendCommand("r")
t0 = time.time()
while rfe.IsResetEvent and time.time()-t0 < 7.0: pass
time.sleep(2.0)
rfe.SendCommand_RequestConfigData()
wait_active_model(rfe)
print("ActiveModel iniziale:", rfe.ActiveModel)

# ⚠️ HACK: forziamo l'active model lato client alla board desiderata
rfe.m_eActiveModel = TARGET_MODEL
# opzionale ma utile per coerenza interna:
if TARGET_MODEL == RFExplorer.RFE_Common.eModel.MODEL_6G:
    rfe.m_bExpansionBoardActive = False
else:
    rfe.m_bExpansionBoardActive = True

# ora inviamo la config per la board "simulata"
rfe.UpdateDeviceConfig(START, STOP)
rfe.ProcessReceivedString(True)

ok = wait_sweep_start(rfe, START)
print("Esito sweep start:", ok)
print("ActiveModel lato client:", rfe.ActiveModel, " | forzato:", rfe.m_eActiveModel)

rfe.Close()
if not ok:
    sys.exit(2)
print(f"OK: impostato {START}-{STOP} MHz su {PORT}@{BAUD} (hack ActiveModel)")

