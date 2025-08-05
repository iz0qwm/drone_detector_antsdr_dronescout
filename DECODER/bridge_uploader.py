import zmq
import json
import time
import threading
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
from flask_cors import CORS
from math import radians, cos, sin, sqrt, atan2

# Porta ZMQ RemoteID
REMOTEID_PORT = 5556

# Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

cred = credentials.Certificate("/home/pi/firebase_key.json")  # o il percorso corretto
firebase_admin.initialize_app(cred)
db = firestore.client()

# Liste globali
drones = []
log_entries = []
MAX_LOGS = 50
drones_lock = threading.Lock()
# filtriamo i dati
last_sent = {}  # drone_id: (lat, lon, timestamp)
SEND_INTERVAL = 10  # secondi
MIN_DISTANCE = 5  # metri


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

    # Invia su firestore
    send_to_firestore(drone_info)


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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371 * 1000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def should_send(drone_id, lat, lon):
    now = time.time()
    if not lat or not lon:
        return False
    key = drone_id or "unknown"
    if key not in last_sent:
        last_sent[key] = (lat, lon, now)
        return True
    old_lat, old_lon, old_ts = last_sent[key]
    dist = haversine(lat, lon, old_lat, old_lon)
    if dist > MIN_DISTANCE or now - old_ts > SEND_INTERVAL:
        last_sent[key] = (lat, lon, now)
        return True
    return False

def send_to_firestore(drone_info):
    try:
        drone_id = drone_info.get("id") or "unknown"
        model = drone_info.get("model") or "-"
        lat = drone_info.get("lat")
        lon = drone_info.get("lon")
        alt = drone_info.get("altitude")
        speed = drone_info.get("speed")

        if None in (lat, lon): return
        if not should_send(drone_id, lat, lon): return

        # Conversione da km/h → m/s se necessario
        if speed and speed > 10:
            speed = round(speed / 3.6, 1)

        now = int(time.time() * 1000)

        data = {
            "lat": lat,
            "lon": lon,
            "altitude": alt,
            "speed": speed,
            "model": model,
            "timestamp": now
        }

        # Aggiorna documento principale
        db.collection("detected_drones").document(drone_id).set(data)

        # Aggiungi punto nella traiettoria
        trajectory_point = {
            "lat": lat,
            "lon": lon,
            "timestamp": now
        }
        db.collection("trajectories").document(drone_id).collection("points").add(trajectory_point)

        print(f"📤 Inviato {drone_id} a Firestore")

    except Exception as e:
        print(f"🔥 Errore invio Firestore: {e}")




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

