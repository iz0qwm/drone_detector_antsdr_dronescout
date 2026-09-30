"""Read local readsb state for the private field scene without changing its public API."""
import json
from pathlib import Path

from services.air_local import READSB_JSON

MAX_READSB_BYTES = 2 * 1024 * 1024


def read_local_readsb():
    path = Path(READSB_JSON)
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_READSB_BYTES + 1)
        if len(raw) > MAX_READSB_BYTES:
            return {"now": None, "aircraft": []}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"now": None, "aircraft": []}
    except (OSError, ValueError, UnicodeError):
        # Loss of readsb must not stop mission-area and RID synchronization.
        return {"now": None, "aircraft": []}
