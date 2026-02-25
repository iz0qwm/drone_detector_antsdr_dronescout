import socket
import time
import re
from state import services

HOST = "127.0.0.1"
PORT = 14580

# Regex per posizione APRS (lat/lon standard OGN)
POS_REGEX = re.compile(
    r'(\d{6}h)?'                 # timestamp opzionale
    r'(\d{4}\.\d{2}[NS])'        # lat DDMM.mmN
    r'[\/\\]'                    # separatore / o \
    r'(\d{5}\.\d{2}[EW])'        # lon DDDMM.mmE
)

ALT_REGEX = re.compile(r'/A=(\d{6})')
COURSE_SPEED_REGEX = re.compile(r'(\d{3})/(\d{3})')


# =========================================================
# LISTENER
# =========================================================

def listen_flarm_ogn(on_drone):
    services["FLARM"] = True

    while True:
        try:
            with socket.create_connection((HOST, PORT), timeout=10) as s:
                s.settimeout(60)

                # LOGIN APRS-IS
                s.sendall(b"user FLARM pass -1 vers DSC-FLARM 0.2\r\n")

                f = s.makefile("r")

                for line in f:
                    line = line.strip()
                    #print("[FLARM RAW]", line)
                    if not line or line.startswith("#"):
                        continue

                    drone = parse_aprs(line)
                    if drone:
                        on_drone(drone)

        except Exception as e:
            print(f"[FLARM] APRS error: {e}")
            time.sleep(5)


# =========================================================
# PARSER APRS GENERICO OGN
# =========================================================

def parse_aprs(line):
    try:
        if ":" not in line:
            return None

        header, payload = line.split(":", 1)

        # CALLSIGN
        sender = header.split(">")[0]

        aircraft_type = None
        object_type = None

        if " id" in payload:
            try:
                id_part = payload.split(" id")[1].split()[0]  # es: id21XXXXXX
                type_byte = int(id_part[0:2], 16)
                aircraft_type = (type_byte >> 2) & 0x0F

                #print("DEBUG OGN ID:", id_part, "type_byte:", hex(type_byte), "aircraft_type:", aircraft_type)

                if aircraft_type is not None:
                    object_type = "UAV" if aircraft_type == 0xD else "AIRCRAFT"


            except Exception:
                pass

        category = None

        if aircraft_type == 0xD:
            category = "UAV"
        elif aircraft_type == 3:
            category = "A7"  # helicopter
        elif aircraft_type == 8:
            category = "A3"
        elif aircraft_type == 9:
            category = "A5"

        # TOCALL (tipo sorgente)
        try:
            tocall = header.split(">")[1].split(",")[0]
        except Exception:
            tocall = "UNKNOWN"

        # Filtriamo solo messaggi con posizione
        match = POS_REGEX.search(payload)
        if not match:
            return None

        lat_raw = match.group(2)
        lon_raw = match.group(3)

        lat = aprs_to_decimal(lat_raw)
        lon = aprs_to_decimal(lon_raw)

        altitude = parse_altitude(payload)
        course, speed = parse_course_speed(payload)

        source_type = classify_source(tocall)

        return {
            "source": source_type,
            "id": sender,
            "model": source_type,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "speed": speed,
            "heading": course,
            "category": category,
            "object_type": object_type,   # <-- AGGIUNGI QUESTO
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }


    except Exception:
        return None


# =========================================================
# UTILITIES
# =========================================================

def aprs_to_decimal(v):
    """
    Converte DDMM.mmN o DDDMM.mmE in decimale
    """
    if len(v) == 8:  # lat
        deg = int(v[0:2])
        minutes = float(v[2:7])
    else:  # lon
        deg = int(v[0:3])
        minutes = float(v[3:8])

    sign = -1 if v[-1] in ("S", "W") else 1
    return sign * (deg + minutes / 60.0)


def parse_altitude(payload):
    match = ALT_REGEX.search(payload)
    if match:
        # Altitudine in feet → convertiamo in metri
        feet = int(match.group(1))
        return round(feet * 0.3048, 1)
    return None


def parse_course_speed(payload):
    match = COURSE_SPEED_REGEX.search(payload)
    if match:
        course = int(match.group(1))
        speed = int(match.group(2))
        return course, speed
    return None, None


def classify_source(tocall):
    """
    Classifica la sorgente in modo leggibile
    """
    mapping = {
        "OGNTRK": "OGN_TRACKER",
        "OGADSB": "ADS-B",
        "OGFLR": "FLARM",
        "OGFLR6": "FLARM",
        "OGFLR7": "FLARM",
        "OGNSKY": "SAFESKY",
        "OGNINRE": "INREACH",
        "OGNDVS": "WEATHER",
        "OGAPIK": "APIK",
        "OGNMTK": "MICROTRAK"
    }

    return mapping.get(tocall, tocall)
    