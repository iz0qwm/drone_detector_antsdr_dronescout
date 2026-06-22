import threading
import time
from datetime import datetime, timezone

from config import SETTINGS
from services.logger import log
from services.gps import get_gps_status

from meshtastic.serial_interface import SerialInterface
from pubsub import pub


meshtastic_nodes = {}

running = False
thread = None
interface = None
last_packet_time = 0
last_position_update = 0

last_sent_lat = None
last_sent_lon = None

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def decode_position(position):
    if not position:
        return None, None, None

    lat = position.get("latitude")
    lon = position.get("longitude")
    alt = position.get("altitude")

    if lat is None and "latitudeI" in position:
        lat = position.get("latitudeI") / 10000000.0

    if lon is None and "longitudeI" in position:
        lon = position.get("longitudeI") / 10000000.0

    return lat, lon, alt


def update_node_from_meshtastic(node_id, node):
    user = node.get("user", {})
    position = node.get("position", {})
    metrics = node.get("deviceMetrics", {})

    lat, lon, alt = decode_position(position)

    existing = meshtastic_nodes.get(node_id, {})

    existing.update({
        "id": node_id,
        "num": node.get("num"),
        "longName": user.get("longName"),
        "shortName": user.get("shortName"),
        "hwModel": user.get("hwModel"),
        "macaddr": user.get("macaddr"),
        "lat": lat if lat is not None else existing.get("lat"),
        "lon": lon if lon is not None else existing.get("lon"),
        "altitude": alt if alt is not None else existing.get("altitude"),
        "batteryLevel": metrics.get("batteryLevel"),
        "voltage": metrics.get("voltage"),
        "channelUtilization": metrics.get("channelUtilization"),
        "last_seen": now_iso()
    })

    meshtastic_nodes[node_id] = existing


def on_receive(packet, interface):
    global last_packet_time

    last_packet_time = time.time()

    try:
        from_id = packet.get("fromId")
        decoded = packet.get("decoded", {})
        portnum = decoded.get("portnum")

        if not from_id:
            return

        node = interface.nodes.get(from_id)

        if node:
            update_node_from_meshtastic(from_id, node)

        if portnum in [
            "TEXT_MESSAGE_APP",
            "POSITION_APP"
        ]:
            log(
                "MESHTASTIC",
                f"Packet {portnum} from {from_id}"
            )

        #log("MESHTASTIC", f"Packet from {from_id} type={portnum}")

    except Exception as e:
        log("MESHTASTIC", f"Packet decode error: {e}")


def meshtastic_worker():

    global running
    global interface
    global last_packet_time
    global last_position_update

    device = SETTINGS.get(
        "meshtastic",
        {}
    ).get(
        "device"
    )

    while running:

        try:

            log(
                "MESHTASTIC",
                f"Connecting to {device}"
            )

            interface = SerialInterface(
                device
            )

            log(
                "MESHTASTIC",
                "Connected"
            )

            last_packet_time = time.time()

            pub.subscribe(
                on_receive,
                "meshtastic.receive"
            )

            log(
                "MESHTASTIC",
                f"Initial nodes: {len(interface.nodes)}"
            )

            while running:

                try:

                    current_nodes = list(
                        interface.nodes.items()
                    )

                    #log("MESHTASTIC", f"Current nodes: {len(current_nodes)}")

                    #log("MESHTASTIC", f"Node IDs: {list(interface.nodes.keys())}")

                    for node_id, node in current_nodes:

                        user = node.get(
                            "user",
                            {}
                        )

                        #log("MESHTASTIC",f"REFRESH node={node_id} " f"name={user.get('longName')} " f"short={user.get('shortName')}")

                        update_node_from_meshtastic(
                            node_id,
                            node
                        )

                    #log("MESHTASTIC", f"CACHE nodes={len(meshtastic_nodes)}")

                    if (
                        time.time()
                        - last_position_update
                    ) > 60:

                        log(
                            "MESHTASTIC",
                            "Updating tracker position"
                        )

                        update_tracker_position()

                        last_position_update = (
                            time.time()
                        )

                except Exception as e:

                    log(
                        "MESHTASTIC",
                        f"Node refresh error: {e}"
                    )

                time.sleep(5)

        except Exception as e:

            log(
                "MESHTASTIC",
                f"Connection error: {e}"
            )

            try:

                if interface:

                    log(
                        "MESHTASTIC",
                        "Closing interface"
                    )

                    interface.close()

            except Exception as close_error:

                log(
                    "MESHTASTIC",
                    f"Close error: {close_error}"
                )

            interface = None

            log(
                "MESHTASTIC",
                "Reconnect in 5 seconds"
            )

            time.sleep(5)

def update_tracker_position():

    global last_sent_lat
    global last_sent_lon

    if not interface:
        return

    try:

        gps = get_gps_status()

        if gps.get("available") and gps.get("fix"):

            lat = gps["lat"]
            lon = gps["lon"]
            alt = int(
                gps.get("alt") or 0
            )

            log(
                "MESHTASTIC",
                f"Using GPS position "
                f"{lat:.6f},{lon:.6f}"
            )

        else:
                
            lat = SETTINGS["dsc"]["lat"]
            lon = SETTINGS["dsc"]["lon"]
            alt = 0


            log(
                "MESHTASTIC",
                f"Using manual position "
                f"{lat:.6f},{lon:.6f}"
            )


        if (
            last_sent_lat is not None
            and
            abs(lat - last_sent_lat) < 0.0001
            and
            abs(lon - last_sent_lon) < 0.0001
        ):
            return

        interface.localNode.setFixedPosition(
            lat,
            lon,
            alt
        )


        last_sent_lat = lat
        last_sent_lon = lon

        log(
            "MESHTASTIC",
            f"Position updated "
            f"{lat:.6f}, {lon:.6f}"
        )

    except Exception as e:

        log(
            "MESHTASTIC",
            f"Position update error: {e}"
        )


        
def start():
    global running
    global thread

    if running:
        return

    if not SETTINGS.get("meshtastic", {}).get("enabled", False):
        log("MESHTASTIC", "Disabled in settings")
        return

    running = True

    thread = threading.Thread(
        target=meshtastic_worker,
        daemon=True
    )

    thread.start()


def stop():
    global running
    global interface

    running = False

    try:
        if interface:
            interface.close()
    except:
        pass

    interface = None


def get_nodes():
    return list(meshtastic_nodes.values())


def clear_nodes():
    meshtastic_nodes.clear()


def is_alive(timeout=30):
    if not running:
        return False

    if last_packet_time == 0:
        return False

    return (time.time() - last_packet_time) < timeout


