import subprocess

from services.logger import log
from config import SETTINGS, save_settings
from pathlib import Path
import time


SERVICE_NAME = "readsb-local.service"


def is_receiving():

    try:

        aircraft = Path(
            "/run/readsb/aircraft.json"
        )

        if not aircraft.exists():
            return False

        age = (
            time.time() -
            aircraft.stat().st_mtime
        )

        return age < 30

    except Exception:

        return False
    
def is_config_enabled():

    return SETTINGS.get(
        "traffic",
        {}
    ).get(
        "adsb_local_enabled",
        False
    )



def is_enabled():

    try:

        result = subprocess.run(
            [
                "/usr/bin/systemctl",
                "is-active",
                SERVICE_NAME
            ],
            capture_output=True,
            text=True
        )

        return (
            result.stdout.strip()
            == "active"
        )

    except Exception as e:

        log(
            "READSB",
            f"Status check failed: {e}",
            level="ERROR"
        )

        return False


def set_enabled(enabled):

    SETTINGS["traffic"]["adsb_local_enabled"] = enabled

    save_settings()

    try:

        if enabled:

            log(
                "READSB",
                "Starting ADS-B receiver"
            )

            subprocess.run(
                [
                    "/usr/bin/sudo",
                    "/usr/bin/systemctl",
                    "start",
                    SERVICE_NAME
                ],
                check=True
            )

        else:

            log(
                "READSB",
                "Stopping ADS-B receiver"
            )

            subprocess.run(
                [
                    "/usr/bin/sudo",
                    "/usr/bin/systemctl",
                    "stop",
                    SERVICE_NAME
                ],
                check=True
            )

        state = is_enabled()

        log(
            "READSB",
            f"ADS-B receiver state = {state}"
        )

        return state

    except Exception as e:

        log(
            "READSB",
            f"Unable to change state: {e}",
            level="ERROR"
        )

        return False
    

def start():

    return set_enabled(True)


def stop():

    return set_enabled(False)