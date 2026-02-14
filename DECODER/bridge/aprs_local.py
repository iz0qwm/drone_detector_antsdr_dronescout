import socket
import threading
import time
import argparse
from datetime import datetime
import atexit


HOST = "127.0.0.1"
PORT = 14580
OGN_HOST = "glidern5.glidernet.org"
OGN_PORT = 14580
OGN_CALLSIGN = "IZ0QWM"
OGN_PASSCODE = "23972"  # APRS-IS passcode

HEARTBEAT_INTERVAL = 30

BANNER = "# aprsc 2.1.14 DSC-Local-APRS\r\n"
SERVER_NAME = "DSC-LOCAL"

# Coordinate nodo (Roma Nord - Guidonia esempio)
NODE_LAT = 41.9776954
NODE_LON = 12.6154041
NODE_ALT_M = 100  # altitudine nodo in metri

BEACON_INTERVAL = 300  # 5 minuti

clients = []
lock = threading.Lock()
ogn_socket = None
ogn_lock = threading.Lock()

parser = argparse.ArgumentParser(description="Local APRS-IS server")
parser.add_argument(
    "-log",
    action="store_true",
    help="Log raw APRS packets received"
)

parser.add_argument(
    "-logfile",
    type=str,
    help="Write raw APRS packets to this file"
)

args = parser.parse_args()


LOG_ENABLED = args.log
LOG_FILE = args.logfile

log_fp = None
if LOG_ENABLED and LOG_FILE:
    log_fp = open(LOG_FILE, "a", buffering=1)

def safe_send(conn, data: bytes) -> bool:
    try:
        conn.sendall(data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def broadcast(msg: str, source_conn):
    data = msg.encode(errors="ignore")
    with lock:
        for c in clients[:]:
            if c is source_conn:
                continue
            if not safe_send(c, data):
                try:
                    clients.remove(c)
                except ValueError:
                    pass

def log_rx(callsign, line):
    if not LOG_ENABLED:
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[APRS][RX][{callsign}][{ts}] {line}"

    print(msg)

    if log_fp:
        try:
            log_fp.write(msg + "\n")
        except Exception as e:
            print(f"[APRS][LOG] file write error: {e}")


def log_tx_ogn(line):
    if not LOG_ENABLED:
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[OGN][TX][{ts}] {line}"

    print(msg)

    if log_fp:
        try:
            log_fp.write(msg + "\n")
        except Exception as e:
            print(f"[OGN][LOG] file write error: {e}")


def ogn_keepalive_loop():
    global ogn_socket
    while True:
        time.sleep(30)

        with ogn_lock:
            if ogn_socket:
                try:
                    ogn_socket.sendall(b"\n")
                    if LOG_ENABLED and log_fp:
                        log_fp.write("[OGN][KEEPALIVE] newline sent\n")
                except Exception:
                    print("[OGN] keepalive failed, reconnecting")
                    if LOG_ENABLED and log_fp:
                        log_fp.write("[OGN] keepalive failed\n")
                    try:
                        ogn_socket.close()
                    except:
                        pass
                    ogn_socket = None
                    connect_ogn()


def format_aprs_lat(lat):
    lat_dir = "N" if lat >= 0 else "S"
    lat = abs(lat)
    deg = int(lat)
    minutes = (lat - deg) * 60
    return f"{deg:02d}{minutes:05.2f}{lat_dir}"

def format_aprs_lon(lon):
    lon_dir = "E" if lon >= 0 else "W"
    lon = abs(lon)
    deg = int(lon)
    minutes = (lon - deg) * 60
    return f"{deg:03d}{minutes:05.2f}{lon_dir}"


def ogn_beacon_loop():
    global ogn_socket

    while True:
        time.sleep(BEACON_INTERVAL)

        with ogn_lock:
            if not ogn_socket:
                continue

        ts = datetime.utcnow().strftime("%H%M%S") + "h"

        lat_str = format_aprs_lat(NODE_LAT)
        lon_str = format_aprs_lon(NODE_LON)

        alt_ft = int(NODE_ALT_M / 0.3048)

        # Position beacon
        pos_beacon = (
            f"{OGN_CALLSIGN}>APRS,TCPIP*,qAC,{OGN_CALLSIGN}:"
            f"/{ts}{lat_str}/{lon_str}&/A={alt_ft:06d}"
        )

        forward_to_ogn(pos_beacon)

        # Status beacon
        status_beacon = (
            f"{OGN_CALLSIGN}>APRS,TCPIP*,qAC,{OGN_CALLSIGN}:"
            f">{ts} DSC-Bridge 0.1 HybridNode RomaNord ADSB+FLARM+DJI+RID"
        )

        forward_to_ogn(status_beacon)

def connect_ogn():
    global ogn_socket
    while True:
        try:
            print(f"[OGN] connecting to {OGN_HOST}:{OGN_PORT}")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((OGN_HOST, OGN_PORT))

            login = f"user {OGN_CALLSIGN} pass {OGN_PASSCODE} vers DSC-Bridge 0.1 filter r/41.9/12.6/100\r\n"
            print(f"[OGN] >> {login.strip()}")
            s.sendall(login.encode())

            # leggiamo risposta iniziale (fino a 2 secondi)
            s.settimeout(2)
            login_buffer = ""

            start = time.time()
            while time.time() - start < 2:
                try:
                    chunk = s.recv(4096).decode(errors="ignore")
                    if not chunk:
                        break
                    login_buffer += chunk
                except socket.timeout:
                    break

            if login_buffer:
                for line in login_buffer.splitlines():
                    print(f"[OGN] << {line}")
                    if LOG_ENABLED and log_fp:
                        log_fp.write(f"[OGN][LOGIN-RESP] {line}\n")
            else:
                print("[OGN] no login response received")


            s.settimeout(None)

            ogn_socket = s
            print("[OGN] connected and ready")
            return

        except Exception as e:
            print(f"[OGN] connection failed: {e}")
            time.sleep(10)

def ogn_reader_loop():
    global ogn_socket
    while True:
        time.sleep(0.1)

        with ogn_lock:
            s = ogn_socket

        if not s:
            continue

        try:
            s.settimeout(0.5)
            data = s.recv(4096)
            if data:
                lines = data.decode(errors="ignore").splitlines()
                for line in lines:
                    # Filtra beacon/keepalive del server
                    if line.startswith("#"):
                        if "logresp" in line:
                            print(f"[OGN][LOGIN] {line}")
                        continue

                    # Parse aircraft beacon only
                    if " id" in line:
                        try:
                            id_part = line.split(" id")[1].split()[0]
                            type_byte = int(id_part[2:4], 16)
                            aircraft_type = (type_byte >> 2) & 0x0F

                            # ignora gateway ADSB
                            if "OGADSB" in line:
                                continue

                            # solo UAV veri
                            if aircraft_type == 0xD:
                                print(f"[OGN][RX][UAV] {line}")
                                if LOG_ENABLED and log_fp:
                                    log_fp.write(f"[OGN][RX][UAV] {line}\n")

                        except Exception:
                            pass



            else:
                # server closed connection
                raise ConnectionError("OGN closed connection")

        except socket.timeout:
            continue
        except Exception:
            print("[OGN] reader detected disconnect, reconnecting")
            if LOG_ENABLED and log_fp:
                log_fp.write("[OGN] reader detected disconnect\n")

            with ogn_lock:
                try:
                    ogn_socket.close()
                except:
                    pass
                ogn_socket = None

            connect_ogn()

def forward_to_ogn(msg: str):
    global ogn_socket
    data = (msg + "\r\n").encode()

    with ogn_lock:
        if not ogn_socket:
            print("[OGN] not connected, packet dropped")
            return
        try:
            print(f"[OGN][TX] {msg}")
            log_tx_ogn(msg)
            ogn_socket.sendall(data)
        except Exception:
            print("[OGN] lost connection, reconnecting")
            if LOG_ENABLED and log_fp:
                log_fp.write("[OGN] lost connection, reconnecting\n")

            try:
                ogn_socket.close()
            except:
                pass
            ogn_socket = None
            connect_ogn()



def handle_client(conn, addr):
    buffer = ""
    logged_in = False
    callsign = None
    last_heartbeat = 0.0

    try:
        conn.settimeout(1.0)

        # banner immediato (APRS-IS style)
        if not safe_send(conn, BANNER.encode()):
            return

        while True:
            # recv
            try:
                data = conn.recv(4096)
            except socket.timeout:
                data = None
            except (ConnectionResetError, OSError):
                return

            # nessun dato → non è un client reale
            if not data:
                if not logged_in:
                    return
            else:
                buffer += data.decode(errors="ignore")

            # heartbeat SOLO dopo login
            if logged_in:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    if not safe_send(conn, b"# keepalive\r\n"):
                        return
                    last_heartbeat = now

            # processa righe complete
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    continue

                # login APRS-IS
                if not logged_in and line.lower().startswith("user "):
                    parts = line.split()
                    callsign = parts[1] if len(parts) > 1 else "UNKNOWN"
                    resp = f"# logresp {callsign} verified, server {SERVER_NAME}\r\n"
                    if not safe_send(conn, resp.encode()):
                        return
                    logged_in = True
                    last_heartbeat = time.time()
                    print(f"[APRS] session active: {callsign} {addr}")
                    if LOG_ENABLED and log_fp:
                        log_fp.write(f"[APRS] session active: {callsign} {addr}\n")

                    continue

                # commenti APRS
                if line.startswith("#"):
                    continue

                # pacchetto APRS → log + broadcast
                log_rx(callsign or "UNKNOWN", line)
                broadcast(line + "\r\n", conn)
                # evita di reinoltrare frame provenienti già da OGN
                if ",qAS," in line and f",{OGN_CALLSIGN}" in line:
                    continue

                forward_to_ogn(line)


    finally:
        with lock:
            try:
                clients.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except:
            pass
        if logged_in:
            print(f"[APRS] session closed: {callsign} {addr}")
            if LOG_ENABLED and log_fp:
                log_fp.write(f"[APRS] session closed: {callsign} {addr}\n")



def start_server():
    print(f"[APRS] local APRS-IS server listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)

        while True:
            conn, addr = s.accept()
            with lock:
                clients.append(conn)
            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            ).start()

def close_log():
    if log_fp:
        log_fp.close()

atexit.register(close_log)

if __name__ == "__main__":
    connect_ogn()

    threading.Thread(
        target=ogn_reader_loop,
        daemon=True
    ).start()

    threading.Thread(
        target=ogn_beacon_loop,
        daemon=True
    ).start()

    start_server()


