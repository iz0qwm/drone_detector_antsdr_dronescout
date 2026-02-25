# sinks/ogn_formatter.py

import time
import math

# ===============================
# Rate limit per target
# ===============================

_last_sent = {}
MIN_INTERVAL_SECONDS = 3  # 1 frame ogni 3s per stesso ID


# ===============================
# Coordinate conversion
# ===============================

def dec_to_aprs_lat(lat):
    hemi = "N" if lat >= 0 else "S"
    lat = abs(lat)
    deg = int(lat)
    minutes = (lat - deg) * 60
    return f"{deg:02d}{minutes:05.2f}{hemi}"


def dec_to_aprs_lon(lon):
    hemi = "E" if lon >= 0 else "W"
    lon = abs(lon)
    deg = int(lon)
    minutes = (lon - deg) * 60
    return f"{deg:03d}{minutes:05.2f}{hemi}"


def meters_to_feet(m):
    if m is None:
        return 0
    return int(m * 3.28084)


# ===============================
# Aircraft type (OGN tttt)
# ===============================

def determine_aircraft_type(drone):
    src = drone.get("source")
    emitter = (drone.get("emitter_type") or "").upper()

    # UAV anche se arriva via ADSB
    if emitter == "UAV":
        return 0xD

    if src == "ADSB":
        cat = drone.get("category")
        if cat == "A7":
            return 0x3
        if cat in ["A3", "A5"]:
            return 0x9
        return 0x8

    if src in ["DJI", "RemoteID"]:
        return 0xD

    return 0x8



# ===============================
# Address type (aa bits)
# ===============================

def determine_address_type(drone):
    src = drone.get("source")

    if src == "ADSB":
        return 0b01  # ICAO sempre

    if src in ["DJI", "RemoteID"]:
        return 0b11  # OGN tracker

    # Default sicuro: ICAO
    return 0b01



# ===============================
# APRS symbol
# ===============================

def determine_symbol(drone):
    emitter = (drone.get("emitter_type") or "").upper()

    # UAV via qualsiasi sorgente
    if emitter == "UAV":
        return "/D"

    src = drone.get("source")

    if src == "ADSB":
        if drone.get("category") == "A7":
            return "/X"
        return "/^"

    if src in ["DJI", "RemoteID"]:
        return "/'"

    return "/^"



# ===============================
# OGN id builder
# ===============================

def build_ogn_id(address_hex, aircraft_type, address_type):
    stealth = 0
    no_tracking = 0

    value = (
        (stealth << 7)
        | (no_tracking << 6)
        | (aircraft_type << 2)
        | address_type
    )

    return f"id{value:02X}{address_hex.upper()}"


# ===============================
# Filtro serio
# ===============================

def should_forward(drone):
    lat = drone.get("lat")
    lon = drone.get("lon")
    alt = drone.get("altitude")

    if lat is None or lon is None:
        return False

    # ADSB solo traffico basso
    if drone.get("source") == "ADSB":
        if alt and alt > 10000:
            return False

    return True


# ===============================
# Frame builder principale
# ===============================
def build_precision_extension(drone):
    nic = drone.get("nic")
    nac_p = drone.get("nac_p")

    if nic is None or nac_p is None:
        return ""

    # compressione semplice in due cifre
    w1 = min(max(nic, 0), 9)
    w2 = min(max(nac_p, 0), 9)

    return f"!W{w1}{w2}!"


def build_frame(drone, station_callsign):

    if not should_forward(drone):
        return None

    raw_id = drone.get("id")
    if not raw_id:
        return None

    now = time.time()

    # rate limit per target
    last = _last_sent.get(raw_id)
    if last and now - last < MIN_INTERVAL_SECONDS:
        return None

    _last_sent[raw_id] = now

    clean_id = raw_id.replace("0x", "").upper()

    # Se già inizia con ICA, togli prefisso
    if clean_id.startswith("ICA"):
        clean_id = clean_id[3:]

    # prendi solo 6 caratteri ICAO
    clean_id = clean_id[:6]

    address = "ICA" + clean_id


    lat = dec_to_aprs_lat(drone["lat"])
    lon = dec_to_aprs_lon(drone["lon"])

    alt = meters_to_feet(drone.get("altitude"))
    
    heading = int(drone.get("heading") or 0)
    speed = int(drone.get("speed") or 0)

    precision = build_precision_extension(drone)
    aircraft_type = determine_aircraft_type(drone)
    address_type = determine_address_type(drone)
    symbol = determine_symbol(drone)

    ogn_id = build_ogn_id(clean_id, aircraft_type, address_type)

    extra = ""

    vs = drone.get("vertical_speed")
    if vs is not None:
        extra += f" {int(vs)}fpm"

    flight = drone.get("flight")
    if flight:
        extra += f" {flight}"

    sq = drone.get("squawk")
    if sq:
        extra += f" Sq{sq}"

    alt_ft = meters_to_feet(drone.get("altitude"))
    fl = alt_ft / 100
    extra += f" FL{fl:.2f}"
    
    timestamp = time.strftime("%H%M%S", time.gmtime())

    # symbol[0] = table
    # symbol[1] = symbol code
    symbol_table = symbol[0]
    symbol_code = symbol[1]

    frame = (
        f"{address}>OGADSB,qAS,{station_callsign}:"
        f"/{timestamp}h"
        f"{lat}{symbol_table}{lon}"
        f"{symbol_code}{heading:03d}/{speed:03d}"
        f"/A={alt:06d} {precision} "
        f"{ogn_id}"
        f"{extra}"
    )

    return frame
