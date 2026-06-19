from services.network import (
    get_network_status
)

import services.ds110 as ds110

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
        "remote_id": ds110.running,

        # futuro OGN
        "ogn": internet,

        # futuro DSC
        "dsc": internet

    }