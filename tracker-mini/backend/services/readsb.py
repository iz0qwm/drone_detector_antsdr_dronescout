import subprocess

from services.logger import log


SERVICE_NAME = "readsb-local.service"


def is_enabled():

    try:

        result = subprocess.run(
            [
                "/usr/bin/sudo",
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