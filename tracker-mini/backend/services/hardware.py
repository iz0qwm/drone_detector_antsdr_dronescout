import os
from services.ds110 import is_alive


def get_hardware_status():

    return {

        "wifi_client": os.path.exists(
            "/sys/class/net/wlan1"
        ),

        "ds110": os.path.exists(
            "/dev/ttyACM0"
        ),

        "ds110_alive": is_alive()

    }