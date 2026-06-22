import os
import time

from services.network import (
    get_network_status
)

import services.ds110 as ds110
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


def get_services_status():

    network = get_network_status()

    internet = network["internet"]

    return {

        "internet": internet,

        "ads_local":
            adsb_local_alive(),

        "ads_network":
            internet,

        "remote_id":
            ds110.running,
            
        "meshtastic_enabled":
            meshtastic.running,

        "meshtastic_alive":
            meshtastic.is_alive(),

        "ogn":
            internet,

        "dsc":
            internet

    }