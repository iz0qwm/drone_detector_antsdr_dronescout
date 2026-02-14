# sources/dji.py
import zmq
import json
import time
from state import services


def listen_dji(on_drone):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:4221")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    services["DJI"] = True
    print("[DJI] Listener avviato (attesa dati su tcp://127.0.0.1:4221)")

    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)

            drone = {
                "source": "DJI",
                "id": None,
                "model": None,
                "lat": None,
                "lon": None,
                "altitude": None,
                "speed": None,
                "heading": None,
                "rssi": None,
                "encrypted": False,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            # --- Parse DJI ZMQ messages ---
            for item in data:
                if "Basic ID" in item:
                    drone["id"] = item["Basic ID"].get("id")
                    drone["model"] = item["Basic ID"].get("description")
                    drone["rssi"] = item["Basic ID"].get("RSSI")
                elif "Location/Vector Message" in item:
                    loc = item["Location/Vector Message"]
                    drone["lat"] = loc.get("latitude")
                    drone["lon"] = loc.get("longitude")
                    drone["altitude"] = loc.get("geodetic_altitude")
                    drone["speed"] = loc.get("speed")

            # --- Detect encrypted / O4 legacy ---
            if drone["id"] and (
                drone["id"] in ("unknown", "9999999999") or
                "Encrypted" in (drone.get("model") or "")
            ):
                drone["encrypted"] = True
                # IMPORTANT: encrypted drones must NOT have real coordinates
                drone["lat"] = None
                drone["lon"] = None

            on_drone(drone)

            print(f"[DJI] {json.dumps(drone, indent=2)}")

        except Exception as e:
            print(f"[DJI] Error: {e}")
