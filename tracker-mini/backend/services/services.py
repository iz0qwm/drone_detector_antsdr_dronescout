from services.network import (
    get_network_status
)
from services.ds110 import is_alive

def get_services_status():

    network = get_network_status()

    internet = network["internet"]

    return {

        "internet": internet,

        # per ora fake
        "ads_local": False,

        # se c'è internet
        "ads_network": internet,

        # futuro bridge DSC
        "remote_id": is_alive(),

        # futuro OGN
        "ogn": internet,

        # futuro DSC
        "dsc": internet

    }