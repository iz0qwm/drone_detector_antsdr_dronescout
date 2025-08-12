#!/usr/bin/env python3
import time, sys, argparse
import RFExplorer

PORT = "/dev/ttyUSB0"
BAUD = 500000

# Frame binari per cambio modulo (da tuo log Windows)
CM_6G    = b'#\x05CM\x00'  # mainboard 6G
CM_WSUB3 = b'#\x05CM\x01'  # expansion WSUB3G

def wait_active_model(rfe, timeout=6.0):
    t0 = time.time()
    while time.time()-t0 < timeout:
        rfe.ProcessReceivedString(True)
        if rfe.ActiveModel != RFExplorer.RFE_Common.eModel.MODEL_NONE:
            return True
        time.sleep(0.02)
    return False

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

def main():
    parser = argparse.ArgumentParser(description="Switch banda RF Explorer")
    parser.add_argument("--band", required=True, choices=["24", "58"], help="Banda target: 24 o 58 GHz")
    parser.add_argument("--port", default=PORT, help="Porta seriale (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=BAUD, help=f"Baudrate (default: {BAUD})")
    args = parser.parse_args()

    if args.band == "58":
        CM = CM_6G
        START, STOP = 5700, 5900   # restringiamo sul 5.8
    else:
        CM = CM_WSUB3
        START, STOP = 2400, 2483

    try:
        rfe = RFExplorer.RFECommunicator()
        rfe.AutoConfigure = False
        rfe.GetConnectedPorts()
        assert rfe.ConnectPort(args.port, args.baud), f"Connect fallita su {args.port}@{args.baud}"

        # reset + handshake (come il tool)
        rfe.SendCommand("r")
        t0 = time.time()
        while rfe.IsResetEvent and time.time()-t0 < 7.0:
            pass
        time.sleep(2.0)

        rfe.SendCommand_RequestConfigData()
        if not wait_active_model(rfe, 6.0):
            print("WARN: ActiveModel ancora NONE, continuo…", file=sys.stderr)

        # invia frame binario CM raw
        ser = rfe.m_objSerialPort
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass

        ser.write(CM)
        ser.flush()
        time.sleep(0.3)

        # richiedi di nuovo la config e processa finché cambia modello/range
        rfe.SendCommand_RequestConfigData()
        wait_active_model(rfe, 3.0)

        # imposta lo start/stop desiderato
        rfe.UpdateDeviceConfig(START, STOP)
        rfe.ProcessReceivedString(True)

        ok = wait_sweep_start(rfe, START)
        print("ActiveModel:", rfe.ActiveModel)
        if ok:
            print(f"OK: switch '{args.band}' e range {START}-{STOP} MHz impostato su {args.port}@{args.baud}")
        else:
            print("ATTENZIONE: sweep non partito dallo START atteso; verifica sul device.", file=sys.stderr)
            sys.exit(2)
    finally:
        try:
            rfe.Close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

