# bridge_core.py
import threading
import time
import json
import os
from flask import Flask, jsonify

from state import drones, log_entries, stats, lock
from gps import gps_thread

from sources.dji import listen_dji
from sources.remoteid import listen_remoteid

from sinks.firestore import send_firestore
from sinks.dsc import send_dsc

from sources.flarm_ogn import listen_flarm_ogn

from sources.adsb_dump1090 import listen_adsb

from sinks.aprs_client import send_frame
from sinks.ogn_formatter import build_frame

from sinks.dsc_aircraft import send_aircraft_live


RUNTIME_DIR = "/home/pi/bridge/runtime"
RUNTIME_STATS = os.path.join(RUNTIME_DIR, "stats.json")
RUNTIME_TMP = os.path.join(RUNTIME_DIR, "stats.tmp")

ROUTES = {
    "DJI":      {"dsc": True,  "firestore": True},
    "RemoteID": {"dsc": True,  "firestore": True},
    "FLARM":    {"dsc": False, "firestore": False},
    "ADSB":     {"dsc": False, "firestore": False},
}

app = Flask("bridge_core_api")

@app.route("/api/drones")
def api_drones():
    with lock:
        return jsonify(drones)

@app.route("/api/logs")
def api_logs():
    with lock:
        return jsonify(log_entries)

@app.route("/api/receiver")
def api_receiver():
    from state import receiver_status
    return jsonify(receiver_status)

@app.route("/api/sources")
def api_sources():
    from state import services
    return jsonify({k: {"running": v} for k, v in services.items()})



DRONE_TTL_SECONDS = 30   # puoi tararlo

def cleanup_loop():
    while True:
        time.sleep(2)
        now = time.time()

        with lock:
            before = len(drones)
            drones[:] = [
                d for d in drones
                if now - d.get("last_seen", now) < DRONE_TTL_SECONDS
            ]
            after = len(drones)

        if before != after:
            print(f"[CLEANUP] Removed {before - after} stale drones")

def handle_drone(drone):
    drone = normalize_id(drone)
    update_state(drone)

    # NON forwardare ciò che viene da OGN
    if drone.get("source") != "FLARM":
        frame = build_frame(drone, "IZ0QWM")
        if frame:
            print("[BRIDGE] Sending APRS frame:", frame)
            send_frame(frame)

    # ✅ ADSB: se è UAV, trattalo come DRONE, non come aereo
    if drone.get("source") == "ADSB":
        emitter = (drone.get("emitter_type") or drone.get("emitter") or "").upper()
        is_uav = (emitter == "UAV") or (drone.get("category") in ("UAV", "DRONE"))
        if not is_uav:
            send_aircraft_live(drone)
            return
        # else: continua sotto (routing droni)

    # === DA QUI IN POI SOLO DRONI ===
    if drone.get("encrypted"):
        return

    route = ROUTES.get(drone["source"], {})
    if route.get("firestore"):
        send_firestore(drone)
    if route.get("dsc"):
        send_dsc(drone)



def normalize_id(drone):
    raw_id = drone.get("id", "")
    clean = raw_id.replace("ICA", "").lower()
    drone["id"] = clean
    return drone

def write_runtime_stats():
    try:
        os.makedirs(RUNTIME_DIR, exist_ok=True)

        data = {
            "drones_seen": stats["drones_seen"],
            "last_drone_iso": stats["last_drone_iso"],
            "active_drones": len(drones)
        }

        # Scrittura atomica
        with open(RUNTIME_TMP, "w") as f:
            json.dump(data, f)

        os.replace(RUNTIME_TMP, RUNTIME_STATS)

    except Exception as e:
        print("[STATS] write error:", e)



def update_state(drone):
    now = time.time()
    drone["last_seen"] = now

    with lock:
        for i, d in enumerate(drones):
            if d["id"] == drone["id"]:

                existing_source = d.get("source")
                new_source = drone.get("source")

                # 1️⃣ Se abbiamo già ADSB e arriva APRS → ignora
                if existing_source == "ADSB" and new_source != "ADSB":
                    return

                # 2️⃣ Se arriva ADSB → sostituisce sempre
                if new_source == "ADSB":
                    drones[i] = drone
                    break

                # 3️⃣ Se arriva APRS ma non abbiamo ADSB → merge intelligente
                for k, v in drone.items():
                    if v is not None:
                        d[k] = v

                drones[i] = d
                break
        else:
            drones.append(drone)

        log_entries.append(drone.copy())
        stats["drones_seen"] += 1
        stats["last_drone_iso"] = drone["timestamp"]
        write_runtime_stats()



def main():
    print("🟢 Bridge core starting")

    threading.Thread(target=listen_dji, args=(handle_drone,), daemon=True).start()
    threading.Thread(target=listen_remoteid, args=(handle_drone,), daemon=True).start()
    threading.Thread(target=gps_thread, daemon=True).start()

    threading.Thread(
        target=app.run,
        kwargs={
            "host": "127.0.0.1",
            "port": 8090,
            "debug": False,
            "use_reloader": False
        },
        daemon=True
    ).start()

    threading.Thread(
        target=listen_flarm_ogn,
        args=(handle_drone,),
        daemon=True
    ).start()

    threading.Thread(
        target=listen_adsb,
        args=(handle_drone,),
        daemon=True
    ).start()

    threading.Thread(
        target=cleanup_loop,
        daemon=True
    ).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
