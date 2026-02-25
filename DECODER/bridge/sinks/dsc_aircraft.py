import requests
import time

AIRCRAFT_INGEST_URL = "https://ingestaircraftlive-32dg4v266a-oc.a.run.app"

NODE_ID = "IZ0QWM_001"

_last_sent = {}  # rate limit per ICAO

MIN_INTERVAL = 5  # secondi tra invii stesso ICAO
MAX_ALT = 1000    # metri
MIN_SPEED = 20    # km/h

def send_aircraft_live(drone):
    try:
        if drone.get("source") != "ADSB":
            return

        icao = drone.get("id")
        lat = drone.get("lat")
        lon = drone.get("lon")
        alt = drone.get("altitude")
        speed = drone.get("speed")

        if not icao or lat is None or lon is None:
            return

        # filtro quota
        if alt is not None and alt > MAX_ALT:
            return

        # filtro velocità
        if speed is not None and speed < MIN_SPEED:
            return

        now = time.time()

        # rate limit per ICAO
        last = _last_sent.get(icao)
        if last and now - last < MIN_INTERVAL:
            return

        _last_sent[icao] = now

        payload = {
            "nodeId": NODE_ID,
            "aircraft": [{
                "id": icao,
                "lat": lat,
                "lon": lon,
                "altitude": alt,
                "speed": speed,
                "heading": drone.get("heading"),
                "source": "ADSB",
                "quality": "mlat" if drone.get("mlat") else "adsb",
                "category": drone.get("category") 
            }]
        }

        r = requests.post(AIRCRAFT_INGEST_URL, json=payload, timeout=5)

        if r.status_code == 200:
            print(f"✈️ Aircraft live OK: {icao}")
        else:
            print(f"❌ Aircraft live error {r.status_code}: {r.text}")

    except Exception as e:
        print(f"🔥 Aircraft send error: {e}")
