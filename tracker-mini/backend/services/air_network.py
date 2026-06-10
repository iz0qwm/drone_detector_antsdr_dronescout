import requests
import time
from services.logger import log

MAX_ALTITUDE_METERS = 1000

SOLARMONITOR_ADSB_URL = (
    "https://solarmonitor.kwos.org/api/adsb/aircraft.json"
)

OGN_URL = (
    "https://solarmonitor.kwos.org/api/ogn/traffic"
)

OPENSKY_URL = (
    "https://opensky-network.org/api/states/all"
)


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


def fetch_solarmonitor(bounds):
    try:
        res = requests.get(
            SOLARMONITOR_ADSB_URL,
            timeout=4
        )
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        log(
            "ADSB",
            "SolarMonitor unavailable",
            level="WARNING"
        )
        return []

    aircraft = []

    for a in data.get("aircraft", []):
        lat = a.get("lat")
        lon = a.get("lon")
        alt_ft = a.get("alt_geom") or a.get("alt_baro")

        if alt_ft is None:
            continue

        try:
            alt_ft = float(alt_ft)
        except Exception:
            log(
                "ADSB",
                f"BAD SOLAR ALT: {alt_ft} "
                f"type={type(alt_ft)} "
                f"hex={a.get('hex')}"
            )
            continue

        if lat is None or lon is None or alt_ft is None:
            continue

        if not in_bounds(lat, lon, bounds):
            continue

        if (a.get("seen") or 999) > 30:
            continue

        altitude = alt_ft * 0.3048
        is_heli = a.get("category") == "A7"

        if not is_heli and altitude > MAX_ALTITUDE_METERS:
            continue

        speed = a.get("gs")
        if isinstance(speed, (int, float)):
            speed = speed * 0.514444

        aircraft.append({
            "icao": a.get("hex"),
            "callsign": (a.get("flight") or a.get("hex") or "N/A").strip(),
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "speed": speed,
            "heading": a.get("track") or 0,
            "category": a.get("category"),
            "isHelicopter": is_heli,
            "source": "SOLARMONITOR_ADSB",
            "updatedAt": int(time.time() * 1000)
        })

    return aircraft


def fetch_ogn(bounds):
    try:
        res = requests.get(
            OGN_URL,
            timeout=4
        )
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        log(
            "ADSB",
            "OGN ADS-B unavailable",
            level="WARNING"
        )
        return []

    now = time.time()
    aircraft = []

    for obj in data.get("objects", []):
        if obj.get("source") != "ADSB":
            continue

        lat = obj.get("lat")
        lon = obj.get("lon")
        alt = obj.get("alt_m")

        if lat is None or lon is None or alt is None:
            continue

        if not in_bounds(lat, lon, bounds):
            continue

        if now - obj.get("last_seen", 0) > 15:
            continue

        category = obj.get("category")
        is_heli = category == "A7"

        if not is_heli and alt > MAX_ALTITUDE_METERS:
            continue

        icao = obj.get("icao")

        if not icao and obj.get("id", "").startswith("ICA"):
            icao = obj["id"][3:].lower()

        if not icao:
            icao = "ogn_" + str(obj.get("id", "unknown")).lower()

        speed = obj.get("speed")

        try:
            speed = float(speed)
        except Exception:
            speed = None
            
        heading = obj.get("heading")

        if not isinstance(heading, (int, float)) and heading is not None:
            log(
                "ADSB",
                f"BAD OGN HEADING: {heading} "
                f"type={type(heading)} "
                f"id={obj.get('id')}"
            )

        aircraft.append({
            "icao": icao,
            "callsign": obj.get("callsign") or obj.get("id") or "N/A",
            "lat": lat,
            "lon": lon,
            "altitude": alt,
            "speed": speed,
            "heading": heading or 0,
            "category": category,
            "isHelicopter": is_heli,
            "source": "OGN_ADSB",
            "updatedAt": int(obj.get("last_seen", now) * 1000)
        })

    return aircraft


def fetch_opensky(bounds):
    try:
        res = requests.get(
            OPENSKY_URL,
            params={
                "lamin": bounds["minLat"],
                "lomin": bounds["minLon"],
                "lamax": bounds["maxLat"],
                "lomax": bounds["maxLon"]
            },
            timeout=6
        )
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        log(
            "ADSB",
            "OpenSky unavailable",
            level="WARNING"
        )
        return []

    aircraft = []

    for s in data.get("states", []) or []:
        icao = s[0]
        callsign = (s[1] or icao or "N/A").strip()
        lon = s[5]
        lat = s[6]
        baro_alt = s[7]
        on_ground = s[8]
        velocity = s[9]
        heading = s[10]
        geo_alt = s[13]
        category = s[17] if len(s) > 17 else None

        altitude = geo_alt if geo_alt is not None else baro_alt

        if lat is None or lon is None or altitude is None:
            continue

        if on_ground:
            continue

        is_heli = (
            category == 6
            or callsign.startswith("POLI")
            or callsign.startswith("PS")
            or callsign.startswith("CC")
            or callsign.startswith("VF")
            or callsign.startswith("HELI")
        )

        if not is_heli and altitude > MAX_ALTITUDE_METERS:
            continue

        aircraft.append({
            "icao": icao,
            "callsign": callsign,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "speed": velocity,
            "heading": heading or 0,
            "category": category,
            "isHelicopter": is_heli,
            "source": "OPENSKY",
            "updatedAt": int(time.time() * 1000)
        })

    return aircraft


def merge_aircraft(*lists):
    merged = {}

    for source_list in lists:
        for ac in source_list:
            icao = ac.get("icao")
            if not icao:
                continue

            old = merged.get(icao)

            if not old:
                merged[icao] = ac
                continue

            if ac.get("updatedAt", 0) >= old.get("updatedAt", 0):
                merged[icao] = ac

    return list(merged.values())


def get_network_aircraft(bounds):
    solarmonitor = fetch_solarmonitor(bounds)
    ogn = fetch_ogn(bounds)
    opensky = fetch_opensky(bounds)

    log(
        "ADSB",
        f"Sources: SOLAR={len(solarmonitor)} "
        f"OGN={len(ogn)} "
        f"OPENSKY={len(opensky)}"
    )

    return {
        "success": True,
        "sources": {
            "solarmonitor": len(solarmonitor),
            "ogn": len(ogn),
            "opensky": len(opensky)
        },
        "aircraft": merge_aircraft(
            solarmonitor,
            ogn,
            opensky
        )
    }