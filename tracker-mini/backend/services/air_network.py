import requests
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.logger import log

MAX_ALTITUDE_METERS = 1000
MAX_POINT_RADIUS_NM = 250
READSB_STALE_SECONDS = 45
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "MiniTracker/1.0 (+https://github.com/iz0qwm/drone_detector_antsdr_dronescout)",
}

SOLARMONITOR_ADSB_URL = (
    "https://solarmonitor.kwos.org/api/adsb/aircraft.json"
)

OGN_URL = (
    "https://solarmonitor.kwos.org/api/ogn/traffic"
)

OPENSKY_URL = (
    "https://opensky-network.org/api/states/all"
)

AIRPLANES_LIVE_BASE_URL = (
    "https://api.airplanes.live"
)

ADSB_LOL_BASE_URL = (
    "https://api.adsb.lol"
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


def to_float(value):
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except Exception:
        return None
    if not math.isfinite(n):
        return None
    return n


def normalize_heading(value):
    n = to_float(value)
    if n is None:
        return 0
    return n % 360


def is_readsb_helicopter(category):
    return str(category or "").upper() == "A7"


def within_altitude_filter(altitude, is_heli, show_all):
    if show_all or is_heli:
        return True
    if altitude is None:
        return True
    return altitude <= MAX_ALTITUDE_METERS


def haversine_meters(lat1, lon1, lat2, lon2):
    r = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_query_from_bounds(bounds):
    center_lat = (bounds["minLat"] + bounds["maxLat"]) / 2
    center_lon = (bounds["minLon"] + bounds["maxLon"]) / 2

    corners = [
        (bounds["minLat"], bounds["minLon"]),
        (bounds["minLat"], bounds["maxLon"]),
        (bounds["maxLat"], bounds["minLon"]),
        (bounds["maxLat"], bounds["maxLon"]),
    ]

    radius_m = max(
        haversine_meters(center_lat, center_lon, lat, lon)
        for lat, lon in corners
    )
    radius_nm = int(math.ceil(radius_m / 1852))
    radius_nm = max(1, min(MAX_POINT_RADIUS_NM, radius_nm))

    return center_lat, center_lon, radius_nm


def aircraft_from_readsb_item(item, provider_source, bounds, show_all,
                              response_now_seconds=None):
    if item.get("alt_baro") == "ground":
        return None

    icao = item.get("hex")
    if not icao:
        return None
    icao = str(icao).strip().lower()

    lat = to_float(item.get("lat"))
    lon = to_float(item.get("lon"))
    if lat is None or lon is None:
        return None

    if not in_bounds(lat, lon, bounds):
        return None

    alt_ft = to_float(item.get("alt_geom"))
    if alt_ft is None:
        alt_ft = to_float(item.get("alt_baro"))
    altitude = alt_ft * 0.3048 if alt_ft is not None else None

    category = item.get("category")
    is_heli = is_readsb_helicopter(category)

    if not within_altitude_filter(altitude, is_heli, show_all):
        return None

    seen_seconds = to_float(item.get("seen"))
    if seen_seconds is not None and seen_seconds > READSB_STALE_SECONDS:
        return None

    if response_now_seconds and seen_seconds is not None:
        updated_at = int((response_now_seconds - seen_seconds) * 1000)
    else:
        updated_at = int(time.time() * 1000)

    speed = to_float(item.get("gs"))
    if speed is not None:
        speed = speed * 0.514444

    heading = normalize_heading(
        item.get("track")
        or item.get("true_heading")
        or item.get("mag_heading")
        or item.get("nav_heading")
    )

    return {
        "icao": icao,
        "callsign": (item.get("flight") or icao or "N/A").strip(),
        "lat": lat,
        "lon": lon,
        "altitude": altitude,
        "speed": speed,
        "heading": heading,
        "category": category,
        "isHelicopter": is_heli,
        "source": provider_source,
        "updatedAt": updated_at
    }


def fetch_solarmonitor(bounds, show_all=False):
    try:
        res = requests.get(
            SOLARMONITOR_ADSB_URL,
            headers=REQUEST_HEADERS,
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

    response_now_seconds = to_float(data.get("now"))
    return [
        ac for ac in (
            aircraft_from_readsb_item(
                item,
                "SOLARMONITOR_ADSB",
                bounds,
                show_all,
                response_now_seconds
            )
            for item in data.get("aircraft", [])
        )
        if ac
    ]


def fetch_readsb_point_provider(base_url, provider_source, bounds, show_all=False):
    center_lat, center_lon, radius_nm = point_query_from_bounds(bounds)
    url = (
        f"{base_url}/v2/point/"
        f"{center_lat:.5f}/{center_lon:.5f}/{radius_nm}"
    )

    try:
        res = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=6
        )
        res.raise_for_status()
        data = res.json()
    except Exception:
        log(
            "ADSB",
            f"{provider_source} unavailable",
            level="WARNING"
        )
        return []

    items = data.get("aircraft")
    if not isinstance(items, list):
        items = data.get("ac")
    if not isinstance(items, list):
        items = []

    response_now_seconds = to_float(data.get("now"))

    return [
        ac for ac in (
            aircraft_from_readsb_item(
                item,
                provider_source,
                bounds,
                show_all,
                response_now_seconds
            )
            for item in items
        )
        if ac
    ]


def fetch_airplanes_live(bounds, show_all=False):
    return fetch_readsb_point_provider(
        AIRPLANES_LIVE_BASE_URL,
        "AIRPLANES_LIVE",
        bounds,
        show_all
    )


def fetch_adsb_lol(bounds, show_all=False):
    return fetch_readsb_point_provider(
        ADSB_LOL_BASE_URL,
        "ADSB_LOL",
        bounds,
        show_all
    )


def fetch_ogn(bounds, show_all=False):
    try:
        res = requests.get(
            OGN_URL,
            headers=REQUEST_HEADERS,
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

        if (
            not show_all
            and not is_heli
            and alt > MAX_ALTITUDE_METERS
        ):
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


def fetch_opensky(bounds, show_all=False):
    try:
        res = requests.get(
            OPENSKY_URL,
            params={
                "lamin": bounds["minLat"],
                "lomin": bounds["minLon"],
                "lamax": bounds["maxLat"],
                "lomax": bounds["maxLon"],
                "extended": 1
            },
            headers=REQUEST_HEADERS,
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
            category == 8
            or callsign.startswith("POLI")
            or callsign.startswith("PS")
            or callsign.startswith("CC")
            or callsign.startswith("VF")
            or callsign.startswith("HELI")
        )

        if (
            not show_all
            and not is_heli
            and altitude > MAX_ALTITUDE_METERS
        ):
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
                ac["source"] = ac.get("source") or "UNKNOWN"
                merged[icao] = ac
                continue

            old_sources = set(str(old.get("source", "")).split("+"))
            old_sources.discard("")
            old_sources.add(ac.get("source", "UNKNOWN"))

            if (
                ac.get("callsign")
                and ac.get("callsign") != ac.get("icao")
                and old.get("callsign") == old.get("icao")
            ):
                old["callsign"] = ac["callsign"]

            if ac.get("updatedAt", 0) >= old.get("updatedAt", 0):
                for key, value in ac.items():
                    if value is not None:
                        old[key] = value

            for key in ("altitude", "speed", "heading", "category"):
                if old.get(key) is None and ac.get(key) is not None:
                    old[key] = ac[key]

            old["isHelicopter"] = bool(
                old.get("isHelicopter")
                or ac.get("isHelicopter")
            )
            old["source"] = "+".join(sorted(old_sources))

    return list(merged.values())


def fetch_provider(name, fetcher, bounds, show_all):
    try:
        return name, fetcher(bounds, show_all)
    except Exception as e:
        log(
            "ADSB",
            f"{name} fetch error: {e}",
            level="WARNING"
        )
        return name, []


def get_network_aircraft(bounds, show_all=False):
    providers = [
        # SolarMonitor ADS-B is intentionally paused. Keep the fetcher above
        # available for a controlled re-enable without changing parsing logic.
        ("airplanes_live", fetch_airplanes_live),
        ("adsb_lol", fetch_adsb_lol),
        ("ogn", fetch_ogn),
        ("opensky", fetch_opensky),
    ]

    results = {}

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = [
            executor.submit(fetch_provider, name, fetcher, bounds, show_all)
            for name, fetcher in providers
        ]

        for future in as_completed(futures):
            name, aircraft = future.result()
            results[name] = aircraft

    log(
        "ADSB",
        "Sources: "
        + " ".join(
            f"{name.upper()}={len(results.get(name, []))}"
            for name, _fetcher in providers
        )
    )

    return {
        "success": True,
        "sources": {
            name: len(results.get(name, []))
            for name, _fetcher in providers
        },
        "aircraft": merge_aircraft(
            *(results.get(name, []) for name, _fetcher in providers)
        )
    }
