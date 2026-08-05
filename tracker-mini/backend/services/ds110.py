# services/ds110.py

import threading
import time
from pymavlink import mavutil
import struct
from datetime import datetime, timezone
import glob
from services.logger import log
from config import SETTINGS
from services.dsc_bridge import (
    send_detected_drone_to_dsc
)


log("DS110", "Available serials: {}".format(glob.glob("/dev/ttyACM*")))

remoteid_aircraft = {}

running = False
thread = None
master = None
last_serial = None
ODID_MESSAGE_SIZE = 25
ODID_ID_SIZE = 20
DEBUG_DS110 = False
last_heartbeat = 0
REMOTEID_STALE_MS = 15000
REMOTEID_RETENTION_GRACE_MS = 60000


def _remoteid_stale_ms():
    return (
        SETTINGS
        .get("proximity", {})
        .get("drone_stale_ms", REMOTEID_STALE_MS)
    )


def _remoteid_retention_ms():
    return (
        _remoteid_stale_ms() +
        SETTINGS
        .get("proximity", {})
        .get("target_retention_ms", REMOTEID_RETENTION_GRACE_MS)
    )


def _last_seen_timestamp(aircraft):
    last_seen = aircraft.get("last_seen")

    if not last_seen:
        return None

    try:
        dt = datetime.fromisoformat(
            last_seen.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.timestamp()
    except Exception:
        return None


def _annotate_freshness(aircraft, now=None):
    now = now if now is not None else time.time()
    copy = dict(aircraft)
    seen_ts = _last_seen_timestamp(copy)

    if seen_ts is None:
        copy["updatedAt"] = None
        copy["age_ms"] = None
        copy["stale"] = False
        return copy

    age_ms = max(0, int((now - seen_ts) * 1000))
    copy["updatedAt"] = int(seen_ts * 1000)
    copy["age_ms"] = age_ms
    copy["stale"] = age_ms > _remoteid_stale_ms()
    return copy


def _is_expired(aircraft, now=None):
    now = now if now is not None else time.time()
    seen_ts = _last_seen_timestamp(aircraft)

    if seen_ts is None:
        return False

    age_ms = (now - seen_ts) * 1000
    return age_ms > _remoteid_retention_ms()

def is_valid_position(lat, lon):
    if lat is None or lon is None:
        return False

    if lat == 0.0 and lon == 0.0:
        return False

    return -90 <= lat <= 90 and -180 <= lon <= 180


def clean_string(value):
    return value.replace("\x00", "").replace("\t", "").replace("\r", "").replace("\n", "").strip()

def decode_odid_pack(payload, size):
    data = {
        "source": "RemoteID",
        "serial": None,
        "vendor": None,
        "model": None,
        "id_type": None,
        "ua_type": None,
        "lat": None,
        "lon": None,
        "altitude": None,
        "height": None,
        "speed": None,
        "heading": None,
        "operator_lat": None,
        "operator_lon": None,
        "operator_altitude": None,
        "operator_id": None,
        "last_seen": datetime.now(timezone.utc).isoformat()
    }

    raw = bytes(payload)

    for x in range(size):
        base = x * ODID_MESSAGE_SIZE
        block = raw[base:base + ODID_MESSAGE_SIZE]

        if len(block) < ODID_MESSAGE_SIZE:
            continue

        rid_type = block[0] >> 4

        # BASIC_ID
        if rid_type == 0:
            data["id_type"] = block[1] >> 4
            data["ua_type"] = block[1] & 0x0F
            uasid = block[2:2 + ODID_ID_SIZE].decode(
                "ascii",
                errors="ignore"
            )

            serial = clean_string(
                uasid
            ).replace(" ", "")

            data["serial"] = serial

            vendor, model = identify_drone(
                serial
            )

            data["vendor"] = vendor
            data["model"] = model

        # LOCATION
        elif rid_type == 1:

           
            speed_mult = block[1] & 0x01
            direction = block[2]
            speed_enc = block[3]

            lat = struct.unpack("<i", block[5:9])[0] / 10_000_000
            lon = struct.unpack("<i", block[9:13])[0] / 10_000_000

            altitude_geo = (struct.unpack("<H", block[15:17])[0] - 2000) / 2
            height = (struct.unpack("<H", block[17:19])[0] - 2000) / 2

            if speed_enc == 255:
                speed = None
            elif speed_mult == 1:
                speed = (speed_enc * 0.75) + (255 * 0.25)
            else:
                speed = speed_enc * 0.75

            data["lat"] = lat
            data["lon"] = lon
            data["altitude"] = altitude_geo
            data["height"] = height
            data["speed"] = speed
            data["heading"] = direction if 0 <= direction <= 360 else None

        # SYSTEM
        elif rid_type == 4:
            operator_lat = struct.unpack("<i", block[2:6])[0] / 10_000_000
            operator_lon = struct.unpack("<i", block[6:10])[0] / 10_000_000
            operator_alt = (struct.unpack("<h", block[18:20])[0] - 2000) / 2

            data["operator_lat"] = operator_lat
            data["operator_lon"] = operator_lon
            data["operator_altitude"] = operator_alt

        # OPERATOR_ID
        elif rid_type == 5:
            operator_id = block[2:2 + ODID_ID_SIZE].decode("ascii", errors="ignore")
            data["operator_id"] = clean_string(operator_id)

    return data



def get_aircraft():

    result = []
    expired_keys = []
    now = time.time()

    for key, aircraft in list(remoteid_aircraft.items()):

        if not aircraft.get("serial"):
            continue

        if _is_expired(aircraft, now):
            expired_keys.append(key)
            continue

        result.append(
            _annotate_freshness(aircraft, now)
        )

    for key in expired_keys:
        remoteid_aircraft.pop(key, None)

    return result

def ds110_worker():

    global running
    global last_serial

    while running:

        try:

            log("DS110", "Connecting...")

            master = mavutil.mavlink_connection(
                SETTINGS["ds110"]["device"],
                baud=SETTINGS["ds110"]["baudrate"],
                dialect="common"
            )
            log("DS110", "Waiting heartbeat...")

            hb = master.wait_heartbeat(timeout=10)

            if not hb:
                raise Exception("No heartbeat")

            log("DS110", "Heartbeat received")

            global last_heartbeat

            last_heartbeat = time.time()

            log(
                "DS110",
                "MAVLink dialect:",
                master.mav.__class__.__module__
            )

            from pymavlink.dialects.v20 import common

            log(
                "DS110",
                "OpenDroneID Message Pack ID:",
                common.MAVLINK_MSG_ID_OPEN_DRONE_ID_MESSAGE_PACK
            )

            while running:

                msg = master.recv_match(
                    blocking=True,
                    timeout=2
                )

                if not msg:
                    continue

                msg_type = msg.get_type()

                try:
                    msg_id = msg.get_msgId()
                except:
                    msg_id = "N/A"

                if msg_type == "HEARTBEAT":
                    last_heartbeat = time.time()
                    continue

                
                if msg_type.startswith("UNKNOWN"):
                    if DEBUG_DS110:
                        log("DS110", f"UNKNOWN MESSAGE TYPE={msg_type} ID={msg_id}")
                        try:
                            log("DS110", f"Message: {msg}")
                            try:
                                log("DS110", f"Message Dict: {msg.to_dict()}")
                            except:
                                pass
                            log("DS110", f"Message Vars: {vars(msg)}")
                        except Exception as e:
                            log("DS110", f"UNKNOWN ERROR: {e}")

                    continue

                    
                if msg_type == "BAD_DATA":
                    #log("DS110", "BAD_DATA RECEIVED Mybe a non-MAVLink message? DJI DroneID frames are sent as MAVLink2 BAD_DATA with the raw frame in the payload")
                    raw = bytes(msg.data)

                    #log(
                    #    "DS110",
                    #    f"BAD_DATA len={len(raw)} hex={raw.hex()[:120]}"
                    #)

                    # ci interessano solo frame MAVLink2
                    #if not raw.startswith(b"\xfd") or len(raw) < 70:
                    #    continue

                    if not raw.startswith(b"\xfd"):
                        continue

                    if len(raw) >= 12:
                        msg_id = (
                            raw[7]
                            | (raw[8] << 8)
                            | (raw[9] << 16)
                        )


                    msg_id = (
                        raw[7]
                        | (raw[8] << 8)
                        | (raw[9] << 16)
                    )


                    if msg_id != 12915:
                        continue


                    payload = raw[10:-2]

                    dji_serial = clean_string(
                        payload[26:46].decode(
                            "ascii",
                            errors="ignore"
                        )
                    )

                    is_dji = dji_serial.startswith(("1581F", "1581E"))

                    decoded = None if is_dji else decode_bad_data_odid_pack(raw)

                    if decoded:
                        serial = decoded.get("serial")

                        if serial and serial.startswith("1596"):
                            log(
                                "DS110",
                                f"DRONETAG VIA BAD_DATA ({serial})"
                            )
                        if decoded.get("serial"):
                            last_serial = decoded["serial"]

                        key = (
                            decoded.get("serial")
                            or last_serial
                            or decoded.get("operator_id")
                            or "unknown"
                        )

                        existing = remoteid_aircraft.get(key, {})

                        for field, value in decoded.items():
                            if value is None:
                                continue

                            if field in ("lat", "lon"):
                                continue

                            existing[field] = value

                        decoded_lat = decoded.get("lat")
                        decoded_lon = decoded.get("lon")

                        if is_valid_position(decoded_lat, decoded_lon):
                            existing["lat"] = decoded_lat
                            existing["lon"] = decoded_lon

                        remoteid_aircraft[key] = existing

                        if (
                            existing.get("serial")
                            and is_valid_position(
                                existing.get("lat"),
                                existing.get("lon")
                            )
                        ):
                            # Send Bad Data decoded to DSC
                            send_detected_drone_to_dsc(existing)

                        #if existing.get("serial") and is_valid_position(existing.get("lat"), existing.get("lon")):
                        #    log(
                        #        "DS110",
                        #        f"RID Aircraft: {existing.get('vendor')} {existing.get('model')} "
                        #        f"({existing.get('serial')}) @ {existing.get('lat'):.7f},{existing.get('lon'):.7f}"
                        #    )

                        continue
                        
                    try:

                        serial = clean_string(
                            payload[26:46].decode(
                                "ascii",
                                errors="ignore"
                            )
                        )

                        vendor, model = identify_drone(
                            serial
                        )

                        if not serial.startswith(("1581F", "1581E")):
                            continue

                        lat = int.from_bytes(
                            payload[54:58],
                            "little",
                            signed=True
                        ) / 10000000.0

                        lon = int.from_bytes(
                            payload[58:62],
                            "little",
                            signed=True
                        ) / 10000000.0

                        log(
                            "DJI",
                            f"Aircraft: {vendor} {model} ({serial}) @ {lat:.7f},{lon:.7f}"
                        )

                        remoteid_aircraft[serial] = {
                            "source": "DJI DroneID",
                            "serial": serial,
                            "vendor": vendor,
                            "model": model,
                            "lat": lat,
                            "lon": lon,
                            "last_seen": datetime.now(
                                timezone.utc
                            ).isoformat()
                        }

                        # invia il drone a DSC
                        drone = remoteid_aircraft[serial]
                        send_detected_drone_to_dsc(drone)


                    except Exception as e:

                        log("DS110", f"DJI decode error: {e}")

                    continue 

                if msg_type == "OPEN_DRONE_ID_MESSAGE_PACK":
                    
                    raw = bytes(msg.messages)

                    #log(
                    #    "DS110",
                    #    f"ODID RAW size={msg.msg_pack_size} data={raw.hex()}"
                    #)

                    decoded = decode_odid_pack(
                        msg.messages,
                        msg.msg_pack_size
                    )

                    serial = decoded.get("serial")

                    #if serial and serial.startswith("1596"):
                    #    log(
                    #        "DS110",
                    #        f"DRONETAG VIA OPEN_DRONE_ID_MESSAGE_PACK ({serial})"
                    #    )

                    if decoded.get("serial"):
                        last_serial = decoded["serial"]

                    key = (
                        decoded.get("serial")
                        or last_serial
                        or decoded.get("operator_id")
                        or "unknown"
                    )

                    if not key:
                        key = f"remoteid_{int(time.time())}"

                    existing = remoteid_aircraft.get(key, {})

                    for field, value in decoded.items():

                        if value is None:
                            continue

                        if field in ("lat", "lon"):
                            continue

                        existing[field] = value

                    decoded_lat = decoded.get("lat")
                    decoded_lon = decoded.get("lon")

                    if is_valid_position(decoded_lat, decoded_lon):
                        existing["lat"] = decoded_lat
                        existing["lon"] = decoded_lon

                    remoteid_aircraft[key] = existing

                    # Sending to DSC
                    send_detected_drone_to_dsc(existing)

                if (
                    existing.get("serial")
                    and is_valid_position(
                        existing.get("lat"),
                        existing.get("lon")
                    )
                ):

                    log(
                        "DS110",
                        f"Aircraft: {existing.get('vendor')} {existing.get('model')} ({existing.get('serial')}) @ {existing.get('lat'):.7f},{existing.get('lon'):.7f}"
                    )

                    send_detected_drone_to_dsc(existing)

                elif msg_type == "ADSB_VEHICLE":

                    log(
                        "DS110",
                        f"ADSB {msg.ICAO_address:06x}"
                    )

        except Exception as e:

            log("DS110", f"Error: {e}")

            try:
                master.close()
            except:
                pass

            time.sleep(5)

def start():

    global running
    global thread

    if running:
        return

    running = True

    thread = threading.Thread(
        target=ds110_worker,
        daemon=True
    )

    thread.start()

def stop():
    global running
    global master

    running = False

    try:
        if master:
            master.close()
    except:
        pass

    clear_aircraft()



def identify_drone(serial):

    if not serial:
        return None, None

    prefix_map = {
        "1581F": "DJI",
        "1581E": "DJI",
        "1748F": "Autel",
        "1748C": "Autel",
        "1596": "Dronetag",
        "2106": "TopView Pollicino"
    }

    model_map = {
            "1ZP": "Mavic 2 Pro",
            "163": "Mavic 2 Pro",
            "1KP": "Mavic 2 Zoom",
            "0ZP": "Mavic Air 2",
            "0M6": "Mavic 2 Zoom",
            "1WN": "Mavic Air 2",
            "3ZP": "Phantom 4 Pro V2.0",
            "2ZP": "Phantom 4 Advanced",
            "3N3": "Mavic Air 2",
            "5ZP": "Inspire 2",
            "446": "Agras T30",
            "4GC": "Mavic 2E",
            "4ZP": "Mavic Mini",
            "45T": "Mavic 3",
            "4QW": "Avata",
            "4QZ": "Mavic 3 Cine",
            "4XF": "Mini 3 Pro",
            "5FJ": "Mavic 3 Thermal",
            "5YH": "Mini 3",
            "574": "Agras T40",
            "6BU": "Agras T50",
            "6Z9": "Mini 4 Pro",
            "67P": "Mavic 3 Classic",
            "67Q": "Mavic 3 Pro",
            "6N8": "Air 3",
            "6W8": "Avata 2",
            "7ZP": "Air 2S",
            "3YT": "Air 2S",
            "8ZP": "Mini 2",
            "895": "Air 3S",
            "9DE": "Mini 5 Pro",
            "7FV": "Matrice 4E",
            "7K3": "Matrice 4T",
            "8HH": "Matrice 4D",
            "8HG": "Matrice 4TD",
            "986": "Mavic 4 Pro",
            "8DB": "Matrice 400",
            "BLK": "Avata 360",
            "A8J": "Avata 360",
            "87L": "Neo",
            "A6Q": "Neo 2",
            "7V2": "Flip",
            "8ZL": "Agras T100",
            "8ZX": "Agras T70P",
            "836": "Agras T25P",
            "574": "Agras T20P",

            "JD2": "Dragonfish Lite",
            "JD3": "Dragonfish Pro",
            "JD1": "Dragonfish Std",
            "EV2": "EVO II V3",
            "EV3": "EVO Max",
            "EV5": "EVO Lite",
            "V4A": "Autel Alpha",

            "F35": "Mini",
            "F33": "DRI",
            "F31": "BS",
            "A34": "Beacon",
            "A30": "Beacon gen.2"
    }

    vendor = None
    model = None

    for prefix, name in prefix_map.items():
        if serial.startswith(prefix):
            vendor = name
            break

    for code, name in model_map.items():
        if code in serial:
            model = name
            break

    return vendor, model


def is_alive(timeout=30):

    if last_heartbeat == 0:
        return False

    return (
        time.time() - last_heartbeat
    ) < timeout


def decode_bad_data_odid_pack(raw):
    if not raw.startswith(b"\xfd") or len(raw) < 36:
        return None

    msg_id = (
        raw[7]
        | (raw[8] << 8)
        | (raw[9] << 16)
    )

    if msg_id != 12915:
        return None

    payload = raw[10:-2]

    if len(payload) < 24:
        return None

    single_message_size = payload[22]
    msg_pack_size = payload[23]

    if single_message_size != ODID_MESSAGE_SIZE or msg_pack_size < 1:
        return None

    messages = payload[24:]

    expected_len = msg_pack_size * ODID_MESSAGE_SIZE

    if len(messages) < expected_len:
        messages = messages + bytes(expected_len - len(messages))

    return decode_odid_pack(messages, msg_pack_size)


def clear_aircraft():
    remoteid_aircraft.clear()
