# sinks/dsc.py
import requests

DSC_INGEST_URL = "https://ingesttrafficobject-32dg4v266a-oc.a.run.app"
DSC_SOURCE = "airsense"

def send_dsc(drone):
    try:
        if not drone.get("id"):
            return

        lat = drone.get("lat")
        lon = drone.get("lon")
        if lat is None or lon is None:
            return


        if drone.get("encrypted"):
            payload = {
                "source": DSC_SOURCE,
                "objectId": "dji-encrypted",
                "type": "alert",
                "lat": RECEIVER_LAT,
                "lon": RECEIVER_LON,
                "model": "DJI Encrypted (possible O4)",
            }
            requests.post(DSC_INGEST_URL, json=payload, timeout=5)
            return

        payload = {
            "source": DSC_SOURCE,
            "objectId": str(drone["id"]),
            "type": "drone",
            "lat": lat,
            "lon": lon,
            "altitude": drone.get("altitude"),
            "speed": drone.get("speed"),
            "heading": drone.get("heading"),
            "model": drone.get("model"),
        }

        r = requests.post(DSC_INGEST_URL, json=payload, timeout=5)

        if r.status_code == 200:
            print(f"🛰️ DSC ingest OK: {drone['id']}")
        else:
            print(f"❌ DSC ingest error {r.status_code}: {r.text}")

    except Exception as e:
        print(f"🔥 DSC send error: {e}")
