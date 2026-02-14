import requests
import time
from state import services

URL = "http://127.0.0.1:9090/data/aircraft.json"

def listen_adsb(on_drone):
    services["ADSB"] = True

    while True:
        try:
            r = requests.get(URL, timeout=5)
            data = r.json()

            for ac in data.get("aircraft", []):

                hex_id = ac.get("hex")
                lat = ac.get("lat")
                lon = ac.get("lon")

                if not hex_id or lat is None or lon is None:
                    continue

                raw_feet = ac.get("alt_baro")

                try:
                    feet = float(raw_feet)
                    alt_m = int(feet * 0.3048)   # tronca i metri
                except (TypeError, ValueError):
                    alt_m = None


                drone = {
                    "source": "ADSB",
                    "id": hex_id,
                    "model": ac.get("type") or "",
                    "category": ac.get("category") or "",
                    "lat": lat,
                    "lon": lon,
                    "altitude": alt_m,
                    "speed": ac.get("gs"),
                    "heading": ac.get("track"),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }


                on_drone(drone)


        except Exception as e:
            print(f"[ADSB] error: {e}")

        time.sleep(2)
