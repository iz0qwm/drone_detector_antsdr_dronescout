import os


def get_hardware_status():

    return {

        "wifi_client": os.path.exists(
            "/sys/class/net/wlan1"
        )

    }