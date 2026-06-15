# services/dsc_bridge.py

import requests

from services.logger import log
from services.network import has_internet
from services.dsc_settings import get_dsc_settings
import time
import socket

_last_sent = {}

DSC_INGEST_URL = (
    "https://ingesttrafficobject-32dg4v266a-oc.a.run.app"
)

DSC_SOURCE = "tracker_mini"


def build_observer():

    cfg = get_dsc_settings()

    return {
        "id": cfg.get("node_id") or socket.gethostname(),

        "name": cfg.get(
            "node_name",
            "Portable Air Node"
        ),

        "type": "tracker-mini",

        "lat": cfg.get("lat"),
        "lon": cfg.get("lon"),

        "capabilities": [
            "remoteid",
            "adsb",
            "missions"
        ]
    }


def send_detected_drone_to_dsc(drone):

    #
    # niente Internet
    #
    if not has_internet():
        return False

    try:

        drone_id = (
            drone.get("id")
            or drone.get("serial")
        )

        if not drone_id:
            return False

        lat = drone.get("lat")
        lon = drone.get("lon")

        if lat is None or lon is None:
            return False

        observer = build_observer()

        payload = {
            "source": DSC_SOURCE,
            "trackerId": observer["id"],
            "objectId": str(drone_id),

            "type": "drone",

            "lat": lat,
            "lon": lon,

            "altitude": drone.get(
                "altitude"
            ),

            "speed": drone.get(
                "speed"
            ),

            "heading": drone.get(
                "heading"
            ),

            "model": drone.get(
                "model"
            ),

            "observer": observer
        }

        now = time.time()

        last = _last_sent.get(
            drone_id,
            0
        )

        if now - last < 5:
            return False

        _last_sent[drone_id] = now

        r = requests.post(
            DSC_INGEST_URL,
            json=payload,
            timeout=5
        )

        if r.status_code == 200:

            log(
                "DSC",
                "Drone sent",
                drone_id
            )

            return True

        log(
            "DSC",
            f"Drone send failed HTTP {r.status_code}",
            level="ERROR"
        )

        return False

    except Exception as e:

        log(
            "DSC",
            f"Drone send error: {e}",
            level="ERROR"
        )

        return False