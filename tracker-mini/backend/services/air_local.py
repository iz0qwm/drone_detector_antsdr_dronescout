import json
import os
import time

from services.logger import log

READSB_JSON = "/run/readsb/aircraft.json"

MAX_ALTITUDE_METERS = 1000

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


def get_local_aircraft(bounds, show_all=False):

    if not os.path.exists(READSB_JSON):

        log(
            "ADSB",
            "readsb aircraft.json not found",
            level="WARNING"
        )

        return {
            "success": False,
            "aircraft": []
        }

    try:

        with open(
            READSB_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        log(
            "ADSB",
            f"Unable to read aircraft.json: {e}",
            level="ERROR"
        )

        return {
            "success": False,
            "aircraft": []
        }

    aircraft = []

    for a in data.get("aircraft", []):

        lat = a.get("lat")
        lon = a.get("lon")

        if lat is None or lon is None:
            continue

        if not in_bounds(
            lat,
            lon,
            bounds
        ):
            continue

        alt_ft = (
            a.get("alt_geom")
            or a.get("alt_baro")
            or 0
        )

        try:
            alt_ft = float(alt_ft)
        except Exception:
            alt_ft = 0

        altitude = alt_ft * 0.3048

        is_heli = (
            a.get("category") == "A7"
        )

        if (
            not show_all
            and not is_heli
            and altitude > MAX_ALTITUDE_METERS
        ):
            continue

        speed = a.get("gs")

        try:
            speed = float(speed) * 0.514444
        except Exception:
            speed = None

        log(
            "ADSB",
            f"LOCAL {a.get('hex')} alt={altitude:.0f}m "
            f"show_all={show_all}"
        )

        aircraft.append({

            "icao":
                a.get("hex"),

            "callsign":
                (
                    a.get("flight")
                    or a.get("hex")
                    or "N/A"
                ).strip(),

            "lat":
                lat,

            "lon":
                lon,

            "altitude":
                altitude,

            "speed":
                speed,

            "heading":
                a.get("track") or 0,

            "category":
                a.get("category"),

            "isHelicopter":
                is_heli,

            "source":
                "LOCAL_ADSB",

            "updatedAt":
                int(time.time() * 1000)
        })

    log(
        "ADSB",
        f"LOCAL ADS-B aircraft: {len(aircraft)}"
    )

    return {
        "success": True,
        "aircraft": aircraft
    }