import subprocess
import threading
import time

from services.network_manager import (
    start_hotspot,
    stop_hotspot,
    hotspot_status
)


def network_available():
    try:
        result = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "DEVICE,STATE",
                "device",
                "status"
            ]
        ).decode().splitlines()

        eth_connected = False
        wifi_connected = False

        for line in result:
            parts = line.split(":", 1)

            if len(parts) != 2:
                continue

            device, state = parts

            if device == "eth0" and state == "connected":
                eth_connected = True

            if device == "wlan0" and state == "connected":
                wifi_connected = True

        # hotspot NON conta
        if hotspot_status()["active"]:
            wifi_connected = False

        return eth_connected or wifi_connected

    except Exception as e:
        print(f"network_available error: {e}")
        return False


def infrastructure_transition_worker():
    try:
        timeout = 30
        interval = 2
        waited = 0
        print("INFRA worker started")

        while waited < timeout:
            print("Checking network...")
            print(network_available())

            if network_available():
                print("Network available, stopping hotspot")
                stop_hotspot()
                return

            time.sleep(interval)
            waited += interval

        print("Timeout expired, keeping hotspot active")
        # timeout scaduto → lasciamo hotspot acceso

    except Exception as e:
        print(f"INFRA WORKER ERROR: {e}")


def set_field_mode():
    try:
        subprocess.run(
            [
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "device",
                "disconnect",
                "wlan0"
            ],
            stderr=subprocess.DEVNULL
        )

        result = start_hotspot()

        if result["success"]:
            return {
                "success": True,
                "mode": "FIELD",
                "message": "Field mode activated"
            }

        return result

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def set_client_mode():
    try:
        worker = threading.Thread(
            target=infrastructure_transition_worker,
            daemon=True
        )
        worker.start()

        return {
            "success": True,
            "mode": "TRANSITION",
            "message": "Infrastructure transition started. Hotspot will remain active until another network is available."
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def get_mode_status():
    hotspot = hotspot_status()

    if hotspot["active"]:
        return {
            "mode": "FIELD"
        }

    if network_available():
        return {
            "mode": "CLIENT"
        }

    return {
        "mode": "UNKNOWN"
    }