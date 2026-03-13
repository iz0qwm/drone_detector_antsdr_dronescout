import math
import json
import os
import threading
import time

# =========================
# CONFIG
# =========================

SECTOR_SIZE = 5               # gradi per settore (72 settori)
MIN_DISTANCE = 300            # metri minimi per considerare il punto
SAVE_INTERVAL = 60            # secondi

RUNTIME_DIR = "/home/pi/bridge/runtime"
COVERAGE_FILE = os.path.join(RUNTIME_DIR, "coverage.json")

# protocolli supportati
PROTOCOLS = ["DJI", "RemoteID", "ADSB", "FLARM"]

drone_max_distance = {
    "DJI": {},
    "RemoteID": {},
    "ADSB_AIRCRAFT": {},
    "ADSB_UAV": {},
    "FLARM": {}
}


# =========================
# STATE
# =========================

coverage = {
    "DJI": {},
    "RemoteID": {},
    "ADSB_AIRCRAFT": {},
    "ADSB_UAV": {},
    "FLARM": {}
}
rx_lat = None
rx_lon = None

lock = threading.Lock()

# =========================
# RECEIVER POSITION
# =========================

def set_receiver_position(lat, lon):
    global rx_lat, rx_lon
    rx_lat = lat
    rx_lon = lon


# =========================
# GEO MATH
# =========================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def bearing(lat1, lon1, lat2, lon2):

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - \
        math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)

    b = math.degrees(math.atan2(x, y))

    return (b + 360) % 360


# =========================
# COVERAGE UPDATE
# =========================

def update_coverage(drone):

    global coverage

    if rx_lat is None or rx_lon is None:
        return

    lat = drone.get("lat")
    lon = drone.get("lon")

    print("RX POS:", rx_lat, rx_lon)
    print("DRONE:", drone.get("source"), drone.get("lat"), drone.get("lon"))

    if lat is None or lon is None:
        return

    source = drone.get("source")
    obj = drone.get("object_type")

    if source == "ADSB":
        if obj == "UAV":
            source = "ADSB_UAV"
        else:
            source = "ADSB_AIRCRAFT"


    if source not in coverage:
        return

    dist = haversine(rx_lat, rx_lon, lat, lon)

    if dist < MIN_DISTANCE:
        return

    drone_id = drone.get("id")
    if not drone_id:
        return

    # non bloccare aggiornamenti dello stesso drone

    #prev_drone_max = drone_max_distance[source].get(drone_id, 0)

    #if dist <= prev_drone_max:
    #    return

    #drone_max_distance[source][drone_id] = dist




    az = bearing(rx_lat, rx_lon, lat, lon)

    sector = int(az // SECTOR_SIZE) * SECTOR_SIZE

    print("SECTOR UPDATE:", source, sector, dist)
    
    with lock:

        prev = coverage[source].get(sector, 0)

        if dist > prev:
            coverage[source][sector] = round(dist, 1)


# =========================
# SAVE / LOAD
# =========================

def save():

    try:

        os.makedirs(RUNTIME_DIR, exist_ok=True)

        with lock:
            data = {k: v.copy() for k, v in coverage.items()}

        tmp = COVERAGE_FILE + ".tmp"

        with open(tmp, "w") as f:
            json.dump(data, f)

        os.replace(tmp, COVERAGE_FILE)

    except Exception as e:
        print("[COVERAGE] save error:", e)


def load():

    global coverage

    if not os.path.isfile(COVERAGE_FILE):
        return

    try:
        with open(COVERAGE_FILE) as f:
            raw = json.load(f)

            coverage = {
                proto: {int(k): v for k, v in sectors.items()}
                for proto, sectors in raw.items()
            }

    except Exception as e:
        print("[COVERAGE] load error:", e)


# =========================
# AUTO SAVE THREAD
# =========================

def autosave_loop():

    while True:
        time.sleep(SAVE_INTERVAL)
        save()


def start():

    load()

    threading.Thread(
        target=autosave_loop,
        daemon=True
    ).start()