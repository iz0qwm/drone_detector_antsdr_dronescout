import os
from services.ds110 import is_alive
from config import SETTINGS

def get_hardware_status():

    return {

        "wifi_client": os.path.exists(
            "/sys/class/net/wlan1"
        ),

        "ds110": os.path.exists(
            SETTINGS["ds110"]["device"]
        ),

        "ds110_alive": is_alive()

    }