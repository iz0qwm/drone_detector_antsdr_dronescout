import time
import threading
import requests
import socket
from services.logger import log
from services.network import has_internet
from services.dsc_settings import get_dsc_settings
from services.gps import get_gps_status



HEARTBEAT_URL = (
    "https://europe-west8-droneskycheck-d0136.cloudfunctions.net/"
    "ingestTrackerHeartbeat"
)

ONLINE_INTERVAL = 60
OFFLINE_INTERVAL = 5

_status = {
    "enabled": True,
    "last_ok": None,
    "last_error": None,
    "last_attempt": None,
    "online": False
}


def build_payload():

    cfg = get_dsc_settings()

    #
    # Non devo trasmettere per configurazione?
    #
    if not cfg.get(
        "sync_enabled",
        True
    ):
        return


    if cfg.get("position_source") == "gps":

        gps = get_gps_status()

        if not gps.get("fix"):
            return None

        lat = gps.get("lat")
        lon = gps.get("lon")

    else:

        lat = cfg.get("lat")
        lon = cfg.get("lon")

    if lat is None or lon is None:
        return None

    return {
        "trackerId": cfg.get("node_id") or socket.gethostname(),

        "name": cfg.get(
            "node_name",
            "Portable Air Node"
        ),

        "version": "1.0.0",

        "lat": lat,
        "lon": lon,

        "capabilities": [
            "missions",
            "remoteid",
            "adsb",
            "meshtastic"
        ]
    }


def send_heartbeat():

    payload = build_payload()

    if not payload:

        _status["last_error"] = (
            "Missing tracker position"
        )

        return False

    _status["last_attempt"] = int(
        time.time()
    )

    try:

        r = requests.post(
            HEARTBEAT_URL,
            json=payload,
            timeout=10
        )

        if r.status_code == 200:

            _status["last_ok"] = int(
                time.time()
            )

            _status["last_error"] = None
            _status["online"] = True

            log(
                "DSC",
                "Heartbeat sent",
                payload["trackerId"]
            )

            return True

        _status["online"] = False

        _status["last_error"] = (
            f"HTTP {r.status_code}"
        )

        log(
            "DSC",
            f"Heartbeat failed HTTP {r.status_code}",
            level="ERROR"
        )

        return False

    except Exception as e:

        _status["online"] = False
        _status["last_error"] = str(e)

        log(
            "DSC",
            f"Heartbeat error: {e}",
            level="ERROR"
        )

        return False


def heartbeat_loop():

    internet_was_up = False

    while True:

        try:

            internet_now = has_internet()

            #
            # OFF -> ON
            #
            if (
                internet_now and
                not internet_was_up
            ):

                log(
                    "DSC",
                    "Internet available"
                )

                internet_was_up = True

                send_heartbeat()

                time.sleep(
                    ONLINE_INTERVAL
                )

                continue

            #
            # ON -> OFF
            #
            if (
                not internet_now and
                internet_was_up
            ):

                log(
                    "DSC",
                    "Internet lost",
                    level="WARNING"
                )

                internet_was_up = False

            #
            # Heartbeat periodico
            #
            if internet_now:

                send_heartbeat()

                time.sleep(
                    ONLINE_INTERVAL
                )

            else:

                time.sleep(
                    OFFLINE_INTERVAL
                )

        except Exception as e:

            log(
                "DSC",
                f"Loop error: {e}",
                level="ERROR"
            )

            time.sleep(10)


def start_dsc_heartbeat():

    t = threading.Thread(
        target=heartbeat_loop,
        daemon=True
    )

    t.start()

    log(
        "DSC",
        "Heartbeat service started"
    )


def get_dsc_heartbeat_status():

    return _status