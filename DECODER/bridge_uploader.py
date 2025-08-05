import zmq
import json
import time
import threading
from flask import Flask, jsonify
from flask_cors import CORS

# Porta ZMQ RemoteID
REMOTEID_PORT = 5556

# Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Liste globali
drones = []
log_entries = []
MAX_LOGS = 50
drones_lock = threading.Lock()

def update_drones(drone_info):
    """Aggiorna lista droni + aggiunge log separato"""
    global drones, log_entries
    with drones_lock:
        # aggiorna drone esistente o aggiunge nuovo
        for i, d in enumerate(drones):
            if d["id"] == drone_info["id"] and d["source"] == drone_info["source"]:
                drones[i] = drone_info
                break
        else:
            drones.append(drone_info)

        # aggiunge log
        log_entries.append(drone_info.copy())
        if len(log_entries) > MAX_LOGS:
            log_entries = log_entries[-MAX_LOGS:]

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/drones")
def get_drones():
    """Restituisce lista droni unici"""
    with drones_lock:
        return jsonify(drones)

@app.route("/logs")
def get_logs():
    """Restituisce log degli ultimi N messaggi"""
    with drones_lock:
        return jsonify(log_entries)

def listen_zmq_dji():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:4221")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        try:
            message = socket.recv_string()
            data = json.loads(message)

            drone_info = {
                "source": "DJI",
                "id": None,
                "model": None,
                "lat": None,
                "lon": None,
                "altitude": None,
                "speed": None,
                "heading": None,
                "rssi": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            for item in data:
                if "Basic ID" in item:
                    drone_info["id"] = item["Basic ID"].get("id")
                    drone_info["model"] = item["Basic ID"].get("description")
                    drone_info["rssi"] = item["Basic ID"].get("RSSI")
                elif "Location/Vector Message" in item:
                    loc = item["Location/Vector Message"]
                    drone_info["lat"] = loc.get("latitude")
                    drone_info["lon"] = loc.get("longitude")
                    drone_info["altitude"] = loc.get("geodetic_altitude")
                    drone_info["speed"] = loc.get("speed")

            update_drones(drone_info)
            print(f"[DJI] {json.dumps(drone_info, indent=2)}")

        except Exception as e:
            print(f"[DJI] Error: {e}")
            continue

def listen_remoteid():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://127.0.0.1:{REMOTEID_PORT}")
    print(f"[RemoteID] Listening on port {REMOTEID_PORT}...")

    while True:
        try:
            message = socket.recv_json()
            drone_info = {
                "source": "RemoteID",
                "id": message.get("icao"),
                "model": None,
                "lat": message.get("lat"),
                "lon": message.get("lon"),
                "altitude": message.get("alt"),
                "speed": message.get("hor_velocity"),
                "heading": message.get("heading"),
                "rssi": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            update_drones(drone_info)
            print(f"[RemoteID] {json.dumps(drone_info, indent=2)}")

        except Exception as e:
            print(f"[RemoteID] Error receiving message: {e}")

def main():
    dji_thread = threading.Thread(target=listen_zmq_dji)
    dji_thread.daemon = True
    dji_thread.start()

    remoteid_thread = threading.Thread(target=listen_remoteid)
    remoteid_thread.daemon = True
    remoteid_thread.start()

    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8080})
    flask_thread.daemon = True
    flask_thread.start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

