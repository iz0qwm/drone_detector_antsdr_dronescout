import requests
import time
from services.logger import log

OGN_URL = "https://solarmonitor.kwos.org/api/ogn/traffic"

VALID_SOURCES = {
    "FLARM",
    "SAFESKY",
    "FREEFLIGHT",
    "FANET"
}

MAX_AGE_SECONDS = 20


def get_bounds_args(request):
    return {
        "minLat": float(request.args.get("minLat")),
        "maxLat": float(request.args.get("maxLat")),
        "minLon": float(request.args.get("minLon")),
        "maxLon": float(request.args.get("maxLon")),
    }


def in_bounds(lat, lon, bounds):
    return (
        bounds["minLat"] <= lat <= bounds["maxLat"]
        and bounds["minLon"] <= lon <= bounds["maxLon"]
    )


def get_ogn_traffic(bounds):
    try:
        res = requests.get(
            OGN_URL,
            timeout=5
        )
        res.raise_for_status()
        data = res.json()

    except Exception as e:
        log(
            "OGN",
            "Traffic source unavailable",
            level="WARNING"
        )
        return {
            "success": False,
            "objects": [],
            "count": 0
        }

    now = time.time()
    objects = []

    for obj in data.get("objects", []):
        src = (
            obj.get("source") or ""
        ).upper()

        if src not in VALID_SOURCES:
            continue

        lat = obj.get("lat")
        lon = obj.get("lon")

        if lat is None or lon is None:
            continue

        if not in_bounds(lat, lon, bounds):
            continue

        last_seen = obj.get("last_seen", 0)

        if now - last_seen > MAX_AGE_SECONDS:
            continue

        objects.append({
            "id": obj.get("id") or obj.get("callsign") or "unknown",
            "callsign": obj.get("callsign") or obj.get("id") or "N/A",
            "lat": lat,
            "lon": lon,
            "alt_m": obj.get("alt_m"),
            "heading": obj.get("heading") or 0,
            "speed": obj.get("speed"),
            "source": src,
            "last_seen": last_seen,
            "updatedAt": int(last_seen * 1000)
        })

    return {
        "success": True,
        "count": len(objects),
        "objects": objects
    }