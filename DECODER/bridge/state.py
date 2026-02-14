
from threading import Lock
from datetime import datetime

drones = []
log_entries = []
stats = {
    "start_iso": datetime.utcnow().isoformat() + "Z",
    "drones_seen": 0,
    "last_drone_iso": None
}

receiver_status = {
    "lat": None, "lon": None, "alt": None,
    "fix_ok": False, "source": "unknown",
    "ts_iso": None, "last_ok_iso": None,
    "last_lat": None, "last_lon": None, "last_alt": None
}

services = {
    "DJI": False,
    "RemoteID": False,
    "FLARM": False,
    "ADSB": False
}

lock = Lock()
