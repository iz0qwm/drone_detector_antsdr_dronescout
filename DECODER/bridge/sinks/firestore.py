# sinks/firestore.py
import time
import requests
from math import radians, cos, sin, sqrt, atan2

PROJECT_ID = "tutto-sui-droni-community"
API_KEY = ""
FIRESTORE_BASE = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"

last_sent = {}
SEND_INTERVAL = 10
MIN_DISTANCE = 5

def doc_url(collection, doc_id):
    return f"{FIRESTORE_BASE}/{collection}?documentId={doc_id}&key={API_KEY}"

def subcollection_url(collection, doc_id, subcollection):
    return f"{FIRESTORE_BASE}/{collection}/{doc_id}/{subcollection}?key={API_KEY}"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 * 1000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def should_send(drone_id, lat, lon):
    now = time.time()
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

def send_firestore(drone):
    try:
        drone_id = str(drone.get("id") or "unknown")
        lat = drone.get("lat")
        lon = drone.get("lon")
        alt = drone.get("altitude")
        speed = drone.get("speed")

        if lat is None or lon is None:
            return
        if not should_send(drone_id, lat, lon):
            return

        if speed and speed > 10:
            speed = round(speed / 3.6, 1)

        now = int(time.time() * 1000)

        doc = {
            "fields": {
                "lat": {"doubleValue": lat},
                "lon": {"doubleValue": lon},
                "altitude": {"doubleValue": alt or 0},
                "speed": {"doubleValue": speed or 0},
                "model": {"stringValue": drone.get("model") or f"Sconosciuto {drone_id}"},
                "timestamp": {"integerValue": now}
            }
        }

        r = requests.post(doc_url("detected_drones", drone_id), json=doc)
        if r.status_code == 409:
            r = requests.patch(
                f"{FIRESTORE_BASE}/detected_drones/{drone_id}?key={API_KEY}",
                json=doc
            )

        pt = {
            "fields": {
                "lat": {"doubleValue": lat},
                "lon": {"doubleValue": lon},
                "timestamp": {"integerValue": now}
            }
        }

        requests.post(subcollection_url("trajectories", drone_id, "points"), json=pt)

    except Exception as e:
        print(f"🔥 Firestore error: {e}")
