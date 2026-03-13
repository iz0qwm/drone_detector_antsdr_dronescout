from datetime import datetime
import time
import socket, json
from state import receiver_status
from coverage import set_receiver_position


ROMA_FALLBACK = (41.9776, 12.6154, 20)

def set_status(lat, lon, alt, ok, source):
    receiver_status.update({
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "fix_ok": ok,
        "source": source,
        "ts_iso": datetime.utcnow().isoformat() + "Z",
        "receiver_alive": True,          # 👈 NUOVO
        "position_mode": "gps" if ok else "fallback"  # 👈 NUOVO
    })

    # aggiorna anche la posizione per il coverage
    set_receiver_position(lat, lon)

    if ok:
        receiver_status["last_ok_iso"] = receiver_status["ts_iso"]
        receiver_status["last_lat"] = lat
        receiver_status["last_lon"] = lon
        receiver_status["last_alt"] = alt


def gps_thread():
    last_fix_time = time.time()
    # appena parte il thread
    set_status(*ROMA_FALLBACK, False, "fallback-roma")

    set_receiver_position(ROMA_FALLBACK[0], ROMA_FALLBACK[1])
    
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", 2947), timeout=3)
            f = s.makefile("rw", buffering=1)
            f.write('?WATCH={"enable":true,"json":true}\n')
            f.flush()

            for line in f:
                j = json.loads(line)
                if j.get("class") == "TPV":
                    lat = j.get("lat")
                    lon = j.get("lon")
                    alt = j.get("alt")
                    mode = j.get("mode", 0)

                    if mode >= 2 and lat is not None and lon is not None:
                        set_status(lat, lon, alt, True, "gpsd")
                        last_fix_time = time.time()
                    else:
                        # niente fix valido
                        if time.time() - last_fix_time > 10:
                            set_status(*ROMA_FALLBACK, False, "fallback-roma")
        except Exception:
            if not receiver_status["last_lat"]:
                set_status(*ROMA_FALLBACK, False, "fallback-roma")
            time.sleep(2)

