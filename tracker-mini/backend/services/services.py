from services.network import (
    get_network_status
)

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
        "remote_id": False,

        # futuro OGN
        "ogn": internet,

        # futuro DSC
        "dsc": internet

    }