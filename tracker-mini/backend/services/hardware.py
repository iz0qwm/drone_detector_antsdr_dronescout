import os
import time

from services.ds110 import is_alive
from config import SETTINGS
import services.meshtastic_service as meshtastic

READSB_JSON = "/run/readsb/aircraft.json"


def adsb_local_alive():

    if not os.path.exists(
        READSB_JSON
    ):
        return False

    try:

        age = (
            time.time()
            - os.path.getmtime(
                READSB_JSON
            )
        )

        return age < 30

    except Exception:
        return False


def get_hardware_status():

    return {

        "wifi_client": os.path.exists(
            "/sys/class/net/wlan1"
        ),

        "ds110": os.path.exists(
            SETTINGS["ds110"]["device"]
        ),

        "ds110_alive": is_alive(),

        "meshtastic":
            meshtastic.running,

        "meshtastic_alive":
            meshtastic.is_alive(),

        "adsb_receiver":
            os.path.exists(
                READSB_JSON
            ),

        "adsb_decoder":
            adsb_local_alive()
    }