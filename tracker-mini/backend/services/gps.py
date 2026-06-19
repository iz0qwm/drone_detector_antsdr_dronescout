import gpsd
import socket
import json

_connected = False


def _ensure_connection():
    global _connected

    if _connected:
        return True

    try:
        gpsd.connect()
        _connected = True
        return True
    except Exception:
        return False

def get_sky_data():

    try:
        sock = socket.create_connection(
            ("127.0.0.1", 2947),
            timeout=2
        )
        sock.sendall(
            b'?WATCH={"enable":true,"json":true};\n'
        )

        for _ in range(20):
            line = sock.recv(4096).decode(
                "utf-8",
                errors="ignore"
            )
            for row in line.splitlines():
                try:
                    obj = json.loads(row)
                    if obj.get("class") == "SKY":
                        return {
                            "satellites":
                                obj.get("uSat"),
                            "hdop":
                                obj.get("hdop")
                        }

                except Exception:
                    pass

    except Exception:
        pass

    return {
        "satellites": None,
        "hdop": None
    }


def get_gps_status():
    if not _ensure_connection():
        return {
            "available": False,
            "fix": False,
            "error": "gpsd not reachable"
        }

    try:
        packet = gpsd.get_current()
        sky = get_sky_data()

        speed = None

        try:
            speed = packet.hspeed()
        except Exception:
            pass

        return {
            "available": True,
            "fix": packet.mode >= 2,
            "mode": packet.mode,
            "lat": packet.lat,
            "lon": packet.lon,
            "alt": packet.alt,
            "speed": speed,
            "track": packet.track,
            "satellites": sky["satellites"],
            "hdop": sky["hdop"]
        }

    except Exception as e:
        global _connected
        _connected = False

        return {
            "available": False,
            "fix": False,
            "error": str(e)
        }