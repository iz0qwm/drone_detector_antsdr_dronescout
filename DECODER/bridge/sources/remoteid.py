# sources/remoteid.py
import zmq
import time
import json
from state import services

REMOTEID_PORT = 5556

def listen_remoteid(on_drone):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    services["RemoteID"] = True
    socket.bind(f"tcp://127.0.0.1:{REMOTEID_PORT}")
    print(f"[RemoteID] Listener avviato (attesa dati su tcp://127.0.0.1:{REMOTEID_PORT})")

    while True:
        try:
            message = socket.recv_json()

            drone = {
                "source": "RemoteID",
                "id": message.get("icao"),
                "model": None,
                "lat": message.get("lat"),
                "lon": message.get("lon"),
                "altitude": message.get("alt"),
                "speed": message.get("hor_velocity"),
                "heading": message.get("heading"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            on_drone(drone)
            print(f"[RemoteID] {json.dumps(drone, indent=2)}")

        except Exception as e:
            print(f"[RemoteID] Error: {e}")
